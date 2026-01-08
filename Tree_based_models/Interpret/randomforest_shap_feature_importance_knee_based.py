"""
Group-wise aggregation of SHAP attributions for feature-level interpretation.

This script:
- Loads the per-subject SHAP result CSV (across 5 repeats × 5 folds).
- Builds subject-by-(iteration, fold) matrices of correctness and label information.
- Identifies subjects that are always correctly or always incorrectly classified
  across all test appearances (per subject).
- Splits subjects into four groups:
    - correct_top:    always correctly classified, true label = 1 (top group)
    - correct_bottom: always correctly classified, true label = 0 (bottom group)
    - wrong_top:      always misclassified, true label = 1
    - wrong_bottom:   always misclassified, true label = 0
  (plus aggregate groups: all, correct, wrong)
- Averages SHAP attributions:
    1) First average across appearances per subject.
    2) Then average across subjects within each group.
- Saves group-wise feature importance (magnitude) and direction (signed effect)
  as text summary and CSV files.

Inputs:
- --variable_type: which feature subset to use
    (all / brain / health / socio / brain_health / brain_socio / health_socio)
- --data_path: path to the original UKB CSV (unused here except for column names via select_data).
- --root_path: directory that contains the SHAP CSV:
      {root_path}/{cls_type}/{variable_type}/{subfolder}/Randomforest_shap_{variable_type}_all_iters_folds.csv
- --outdir: directory to save all group-wise interpretation results.

Outputs (under {outdir}/{cls_type}/{variable_type}):
- all_groups_knee_point_convex_threshold.txt
    Summary of significant features per group (importance + direction),
    where the number of selected features is determined by a knee point (fallback: top-10).
- {group_name}_feature_importance_knee_point_convex_threshold.csv
    For each group (all, correct, wrong, correct_top, correct_bottom, wrong_top, wrong_bottom):
    - feature:    original feature name
    - importance: group-wise averaged |SHAP| (magnitude)
    - direction:  group-wise averaged signed SHAP value
    - abs_direction: |direction|
    - sign:       "positive" or "negative"
- group_meta.npz
    Numpy arrays of subject IDs (eid) for each group.
"""

import argparse
import os, glob
import torch
import numpy as np
from data_utils import select_data_gf_cls, select_data_edu_cls
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt


# ------------------- Config (argparse) -------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Group-wise SHAP aggregation on UKB fluid intelligence (top/bottom 10%)"
    )
    parser.add_argument(
        "--cls_type",
        type=str,
        choices=["gf", "edu"],
        default="gf",
        help="Type of the classification task: gf (fluid intelligence) or edu (education)",
    )
    parser.add_argument(
        "--variable_type",
        type=str,
        choices=["all", "brain", "health", "socio", "brain_health", "health_socio", "brain_socio"],
        default="all",
        help="Which variable set to use: all / brain / health / socio / brain_health / brain_socio / health_socio",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/Step5_refilter_categorical_for_deeplearning.csv",
        help="Path to input CSV data (used only to get feature lists via select_data).",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./results/Tree_based/Randomforest_shap",
        help="Directory that contains Random Forest SHAP CSVs.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./results/Tree_based/Randomforest_shap_plz/Interpret",
        help="Directory to save all group-wise interpretation results.",
    )
    return parser.parse_args()

args = parse_args()

variable_type = args.variable_type
cls_type = args.cls_type
root_path = f'{args.root_path}/{cls_type}/{variable_type}'
subfolder = 'ne800_md40_mf0.3_msl5_mss2_cwbalanced_ms0.9'
# data_tot_path = glob.glob(f'{root_path}/*')[0]
data_path = f'{root_path}/{subfolder}/Randomforest_shap_{variable_type}_all_iters_folds.csv'
save_root = f'{args.outdir}/{cls_type}/{variable_type}'
os.makedirs(save_root, exist_ok=True)

df = pd.read_csv(data_path)
if cls_type == 'gf':
    category_col, continue_col, Categories = select_data_gf_cls(variable_type)
elif cls_type == 'edu':
    category_col, continue_col, Categories = select_data_edu_cls(variable_type)

if category_col is None:
    category_col = []

# RF shap column mapping (num__ / cat__ one-hot merge)
shap_cols = [c for c in df.columns if c.startswith("shap::")]
df[shap_cols] = df[shap_cols].fillna(0.0)

possible_num_prefixes = ["shap::num__", "shap::preprocessor__num__"]
possible_cat_prefixes = ["shap::cat__", "shap::preprocessor__cat__"]

def pick_prefix(cols, candidates):
    for p in candidates:
        if any(c.startswith(p) for c in cols):
            return p
    return None

NUM_PREFIX = pick_prefix(shap_cols, possible_num_prefixes)
CAT_PREFIX = pick_prefix(shap_cols, possible_cat_prefixes)

if NUM_PREFIX is None:
    raise RuntimeError("Could not find numeric SHAP columns starting with shap::num__ ...")
if category_col and CAT_PREFIX is None:
    raise RuntimeError("Could not find categorical SHAP columns starting with shap::cat__ ...")

def match_base(rest: str, candidates):
    hit, best = None, -1
    for cand in candidates:
        if rest == cand or rest.startswith(cand + "_"):
            if len(cand) > best:
                best = len(cand)
                hit = cand
    return hit

# base feature -> shap columns list
base_to_cols = {c: [] for c in (continue_col + category_col)}

for col in shap_cols:
    if col.startswith(NUM_PREFIX):
        rest = col[len(NUM_PREFIX):]
        base = match_base(rest, continue_col)
        if base is not None:
            base_to_cols[base].append(col)

    elif CAT_PREFIX is not None and col.startswith(CAT_PREFIX):
        rest = col[len(CAT_PREFIX):]
        base = match_base(rest, category_col)
        if base is not None:
            base_to_cols[base].append(col)

base_to_cols = {k: v for k, v in base_to_cols.items() if len(v) > 0}
feature_names = list(base_to_cols.keys())
F = len(feature_names)
print(f"[INFO] Detected {F} base features from RF SHAP columns.")

N_ITER, N_FOLD = 5, 5

iter_info = df['iteration'].to_numpy()

all_eids = df['eid'].to_numpy()
uniq_eid = torch.sort(torch.unique(torch.tensor(all_eids)))[0]  # [N_id]
N_ID = uniq_eid.numel()

correct_mat = torch.zeros(N_ID, N_ITER * N_FOLD, dtype=torch.int16)
true_mat = torch.zeros(N_ID, N_ITER * N_FOLD, dtype=torch.int16)
present_mat = torch.zeros(N_ID, N_ITER * N_FOLD, dtype=torch.int16)

col = 0
for it in range(N_ITER):
    iter_idx = np.where(iter_info == it)[0]
    iter_df = df.iloc[iter_idx]

    for fd in range(N_FOLD):
        fold_info = iter_df['fold'].to_numpy()
        fold_idx = np.where(fold_info == fd)[0]
        fold_df = iter_df.iloc[fold_idx]

        eids = torch.tensor(fold_df["eid"].to_numpy()).cpu()
        true_lab = torch.tensor(fold_df["true_label"].to_numpy())
        pred_lab = torch.tensor(fold_df["pred_label"].to_numpy())

        idx_map = torch.searchsorted(uniq_eid, eids)

        true_mat[idx_map, col] = true_lab.short()
        correct_mat[idx_map, col] = (true_lab == pred_lab).short()
        present_mat[idx_map, col] = 1  # ★ 추가
        col += 1

        del fold_df
    del iter_df

# For each eid: how many times it appeared, and how many times it was correct
appears = present_mat.sum(dim=1)                     # [N_ID]
sum_correct = (correct_mat * present_mat).sum(dim=1) # [N_ID]

# eid index (0 ~ N_ID-1)
whole_correct_idx = torch.where((appears > 0) & (sum_correct == appears))[0]
whole_incorrect_idx = torch.where((appears > 0) & (sum_correct == 0))[0]

mean_true = (true_mat.float() * present_mat).sum(dim=1) / torch.clamp(appears.float(), min=1.0)
true_label_consistent = (mean_true > 0.5).long()  # 0 or 1

# Define groups in INDEX SPACE (0 ~ N_ID-1), not eid value
# g_all: at least once appeared in any fold
g_all = torch.where(appears > 0)[0]

# always correct / always wrong (Top+Bottom)
g_correct = whole_correct_idx
g_wrong = whole_incorrect_idx

# split by true label
g_correct_top = whole_correct_idx[true_label_consistent[whole_correct_idx] == 1]
g_correct_bottom = whole_correct_idx[true_label_consistent[whole_correct_idx] == 0]
g_wrong_top = whole_incorrect_idx[true_label_consistent[whole_incorrect_idx] == 1]
g_wrong_bottom = whole_incorrect_idx[true_label_consistent[whole_incorrect_idx] == 0]

print(len(g_all), len(g_correct), len(g_wrong), len(g_correct_top), len(g_correct_bottom), len(g_wrong_top), len(g_wrong_bottom))

groups = {
    "all": g_all,
    "correct": g_correct,
    "wrong": g_wrong,
    "correct_top": g_correct_top,
    "correct_bottom": g_correct_bottom,
    "wrong_top": g_wrong_top,
    "wrong_bottom": g_wrong_bottom,
}

F = len(continue_col) + len(category_col)

eid_sum_mag = torch.zeros(N_ID, F, dtype=torch.float32)
eid_sum_sign = torch.zeros(N_ID, F, dtype=torch.float32)
eid_cnt = torch.zeros(N_ID, dtype=torch.long)

for it in range(N_ITER):
    iter_idx = np.where(iter_info == it)[0]
    iter_df = df.iloc[iter_idx]

    for fd in range(N_FOLD):
        fold_info = iter_df['fold'].to_numpy()
        fold_idx = np.where(fold_info == fd)[0]
        fold_df = iter_df.iloc[fold_idx]

        eids = torch.tensor(fold_df["eid"].to_numpy()).cpu()
        idx_map = torch.searchsorted(uniq_eid, eids)  # [B]

        attr_list = []
        for base in feature_names:
            cols = base_to_cols[base]
            mat = fold_df[cols].to_numpy(dtype=np.float32)  # (B, n_cols)

            signed_sum = mat.sum(axis=1, keepdims=True)  # (B, 1)
            attr_list.append(signed_sum)

        attr_tot_np = np.concatenate(attr_list, axis=1)  # (B, F)
        attr_tot = torch.tensor(attr_tot_np)

        attr_all_mag = attr_tot.float().abs()
        attr_all_sign = attr_tot.float()

        eid_sum_mag.index_add_(0, idx_map, attr_all_mag.cpu())
        eid_sum_sign.index_add_(0, idx_map, attr_all_sign.cpu())

        eid_cnt.index_add_(0, idx_map, torch.ones(idx_map.size(0), dtype=torch.long))

        del fold_df
    del iter_df

cnt_safe = eid_cnt.clamp(min=1).unsqueeze(1)  # [N_ID, 1]
eid_mean_mag = eid_sum_mag / cnt_safe  # [N_ID, F]
eid_mean_sign = eid_sum_sign / cnt_safe  # [N_ID, F]

def group_mean(eid_idx_tensor):
    """Return (mean_magnitude, mean_signed) for a given group of subject indices."""
    # Convert numpy array → torch tensor if necessary
    if isinstance(eid_idx_tensor, np.ndarray):
        eid_idx_tensor = torch.tensor(eid_idx_tensor, dtype=torch.long)
    elif not isinstance(eid_idx_tensor, torch.Tensor):
        eid_idx_tensor = torch.tensor(eid_idx_tensor, dtype=torch.long)

    if eid_idx_tensor.numel() == 0:
        return None, None, None, None

    return (
        eid_mean_mag[eid_idx_tensor].mean(dim=0),
        eid_mean_sign[eid_idx_tensor].mean(dim=0),
        eid_mean_mag[eid_idx_tensor].std(dim=0),
        eid_mean_sign[eid_idx_tensor].std(dim=0),
    )


group_mean_mag = {}
group_mean_sign = {}
group_mean_mag_std = {}
group_mean_sign_std = {}
for name, g_idx in {
    "all": g_all,
    "correct": g_correct,
    "wrong": g_wrong,
    "correct_top": g_correct_top,
    "correct_bottom": g_correct_bottom,
    "wrong_top": g_wrong_top,
    "wrong_bottom": g_wrong_bottom,
}.items():
    gm, gs, gm_std, gs_std = group_mean(g_idx)
    group_mean_mag[name] = gm
    group_mean_sign[name] = gs
    group_mean_mag_std[name] = gm_std
    group_mean_sign_std[name] = gs_std

group_sizes = {
    "all": g_all.numel(),
    "correct": g_correct.numel(),
    "wrong": g_wrong.numel(),
    "correct_top": g_correct_top.numel(),
    "correct_bottom": g_correct_bottom.numel(),
    "wrong_top": g_wrong_top.numel(),
    "wrong_bottom": g_wrong_bottom.numel(),
}

summary_txt_path = os.path.join(save_root, f"all_groups_knee_point_convex_threshold.txt")

with open(summary_txt_path, "w") as f_txt:
    f_txt.write("Group sizes (number of eids):\n")
    for name, sz in group_sizes.items():
        f_txt.write(f"  {name}: {sz}\n")
    f_txt.write("\n")

    for group_name in ["all", "correct", "wrong",
                       "correct_top", "correct_bottom",
                       "wrong_top", "wrong_bottom"]:
        save_path = os.path.join(save_root, "visualization", group_name + "_all_importance_curves.png")
        os.makedirs(os.path.join(save_root, "visualization"), exist_ok=True)

        imp = group_mean_mag[group_name]
        dirn = group_mean_sign[group_name]
        imp_std = group_mean_mag_std[group_name]
        dirn_std = group_mean_sign_std[group_name]

        if imp is None:
            continue

        header = f"=== {group_name.upper()} ==="
        print("\n" + header)
        f_txt.write(header + "\n")

        indices = torch.argsort(imp, dim=0, descending=True)
        sorted_imp = imp[indices]
        sorted_std = imp_std[indices]
        sorted_dirn = dirn[indices]
        sorted_dirn_std = dirn_std[indices]
        feature_names_np = np.array(feature_names)
        sorted_feature_names = feature_names_np[indices]

        x = np.arange(1, len(sorted_imp) + 1)

        kneedle = KneeLocator(x, sorted_imp, curve='convex', direction='decreasing')
        knee_rank = kneedle.knee
        if knee_rank is None:
            n = 10  # top 10
        else:
            n = int(knee_rank)

        print("Detected knee at rank: ", knee_rank)

        # ============= Visualization
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, :])

        point_size = 7
        knee_color = "tab:red"
        knee_alpha = 0.7

        # Raw importance
        max_raw = int(len(x) * 0.1)
        x_raw = x[:max_raw]
        y_raw = sorted_imp[:max_raw]
        ax1.scatter(x_raw, y_raw, s=point_size)

        if knee_rank is not None and knee_rank <= max_raw:
            k = int(knee_rank)

            ax1.axvline(k, linestyle='--',
                        color=knee_color, alpha=knee_alpha)
            # knee point
            ax1.scatter([k], [sorted_imp[k - 1]],
                        s=40, color=knee_color, alpha=knee_alpha)

            xticks = list(ax1.get_xticks())
            if k not in xticks:
                xticks.append(k)
            xticks = sorted(xticks)
            ax1.set_xticks(xticks)

            for label in ax1.get_xticklabels():
                try:
                    val = float(label.get_text())
                except ValueError:
                    continue
                if np.isclose(val, k):
                    label.set_color(knee_color)
                    label.set_fontweight('bold')

        ax1.set_xlim(0, max_raw + 1)
        ax1.set_title(f"[{group_name}] Raw importance (top {max_raw}, convex/decreasing)")
        ax1.set_xlabel("Feature rank")
        ax1.set_ylabel("Importance")

        # Convex importance (Whole)
        x_whole = x
        y_whole = sorted_imp
        ax2.scatter(x_whole, y_whole, s=point_size)
        max_whole = len(x_whole)

        if knee_rank is not None and knee_rank <= max_whole:
            k = int(knee_rank)
            ax2.axvline(k, linestyle='--',
                        color=knee_color, alpha=knee_alpha)
            ax2.scatter([k], [sorted_imp[k - 1]],
                        s=40, color=knee_color, alpha=knee_alpha)

            xticks = list(ax2.get_xticks())
            if k not in xticks:
                xticks.append(k)
            xticks = sorted(xticks)
            ax2.set_xticks(xticks)

            for label in ax2.get_xticklabels():
                try:
                    val = float(label.get_text())
                except ValueError:
                    continue
                if np.isclose(val, k):
                    label.set_color(knee_color)
                    label.set_fontweight('bold')

        ax2.set_xlim(0, max_whole + 1)
        ax2.set_title(f"[{group_name}] Convex importance (top {max_whole}, convex)")
        ax2.set_xlabel("Feature rank")
        ax2.set_ylabel("Convex importance")

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
            print(f"Saved plot to {save_path}")

        top_n_indices = indices[:n]
        print("Selected feature indices: ", top_n_indices)
        selected_vals = sorted_imp[:n]

        for rank in range(n):
            global_idx = top_n_indices[rank]  # 0 ~ F-1
            fname = feature_names[global_idx]        # 원래 이름

            imp_mean_val = imp[global_idx].item()
            imp_std_val  = imp_std[global_idx].item()
            dir_mean_val = dirn[global_idx].item()
            dir_std_val  = dirn_std[global_idx].item()
            arrow = "↑" if dir_mean_val > 0 else "↓"

            line = (
                f"{rank+1:2d}. {fname:30s} | "
                f"importance={imp_mean_val:.5f} ± {imp_std_val:.5f} | "
                f"direction={dir_mean_val:+.5f} ± {dir_std_val:.5f} {arrow}"
            )

            print(line)
            f_txt.write(line + "\n")


# Save
for group_name in group_mean_mag.keys():
    if group_mean_mag[group_name] is None:
        continue

    imp_mean = group_mean_mag[group_name].cpu().numpy()
    dir_mean = group_mean_sign[group_name].cpu().numpy()
    imp_std  = group_mean_mag_std[group_name].cpu().numpy()
    dir_std  = group_mean_sign_std[group_name].cpu().numpy()

    df_group = pd.DataFrame({
        "feature": feature_names,
        "importance": imp_mean,
        "importance_std": imp_std,
        "direction": dir_mean,
        "direction_std": dir_std,
    })
    df_group["abs_direction"] = np.abs(df_group["direction"])
    df_group["sign"] = df_group["direction"].apply(lambda x: "positive" if x > 0 else "negative")

    df_group = df_group.sort_values("importance", ascending=False)
    df_group.to_csv(
        os.path.join(save_root, f"{group_name}_feature_importance_knee_point_convex_threshold.csv"),
        index=False
    )

np.savez(os.path.join(save_root, "group_meta.npz"),
         all=uniq_eid[g_all].numpy(),
         correct=uniq_eid[g_correct].numpy(),
         wrong=uniq_eid[g_wrong].numpy(),
         correct_top=uniq_eid[g_correct_top].numpy(),
         correct_bottom=uniq_eid[g_correct_bottom].numpy(),
         wrong_top=uniq_eid[g_wrong_top].numpy(),
         wrong_bottom=uniq_eid[g_wrong_bottom].numpy())
