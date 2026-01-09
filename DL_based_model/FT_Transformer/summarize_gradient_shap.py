"""
Aggregate Captum GradientShap attributions across 5 iterations × 5 folds,
compute eid-level and group-wise feature importance (mean ± std),
and export top-K features selected via KneeLocator.

- Loads attributions from: Iteration_{i}/Fold_{j}/attributions.pt (i=1..5, j=1..5).
- Builds per-subject (eid) attribution means across all appearances.
- Defines groups based on prediction consistency across runs:
  all / always-correct / always-wrong, and each split by true label (top=1, bottom=0).
- Computes group-wise mean ± std of feature attributions and selects top-K using KneeLocator
  (convex, decreasing), with a fallback to top-10 if no knee is detected.
- Exports CSV summaries.
"""

import argparse
import os
import numpy as np
import torch
from custom_utils.dataloader import select_data
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt


# ------------------- Config (argparse) -------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Captum GradientShap attributions across 5 iterations × 5 folds and export group-wise feature importance."
    )
    parser.add_argument(
        "--variable_type",
        type=str,
        choices=["all", "brain", "health", "socio", "brain_health", "health_socio", "brain_socio"],
        default="all",
        help="Which variable set to use: all / brain / health / socio / brain_health / brain_socio / health_socio",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/LocalData2/daheo/UKB_FINAL/results/DL_based",
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        default="FTTransformer_251116_ep_100_bs_256",
        help="Model experiment folder name (under root_path).",
    )
    parser.add_argument(
        "--model_sub_folder",
        type=str,
        default="01-08-13-32_Interpret_FTTF_E100B2LR0.0003LD1e-05",
        help="Model experiment sub folder name (under model_folder).",
    )
    return parser.parse_args()
args = parse_args()

variable_type = args.variable_type
root_dir_name = args.root_path
subfolder_name1 = args.model_folder
subfolder_name2 = args.model_sub_folder

N_ITER, N_FOLD = 5, 5
root_path = f'{root_dir_name}/{subfolder_name1}/{variable_type}/Interpret/{subfolder_name2}'
save_root = f'{root_dir_name}/{subfolder_name1}/Interpret_result/{variable_type}'
category_col, continue_col, Categories = select_data(variable_type)
os.makedirs(save_root, exist_ok=True)

# Collect all subject IDs (eids) across iterations and folds
all_eids = []
for it in range(1, N_ITER + 1):
    for fd in range(1, N_FOLD + 1):
        data = torch.load(os.path.join(root_path, f'Iteration_{it}/Fold_{fd}/attributions.pt'), weights_only=False)
        eids = torch.cat(data["eid"], dim=0).cpu()
        all_eids.append(eids)
uniq_eid = torch.sort(torch.unique(torch.cat(all_eids, dim=0)))[0]  # [N_id]
N_ID = uniq_eid.numel()

# Build per-eid matrices for correctness, true labels, and presence across 25 runs (5×5)
correct_mat = torch.zeros(N_ID, N_ITER*N_FOLD, dtype=torch.int16)
true_mat    = torch.zeros(N_ID, N_ITER*N_FOLD, dtype=torch.int16)
present_mat = torch.zeros(N_ID, N_ITER*N_FOLD, dtype=torch.int16)

col = 0
for it in range(1, N_ITER + 1):
    for fd in range(1, N_FOLD + 1):
        data = torch.load(os.path.join(root_path, f'Iteration_{it}/Fold_{fd}/attributions.pt'), weights_only=False)
        eids = torch.cat(data["eid"], dim=0).cpu()
        yt   = torch.cat(data["true"], dim=0)
        yp   = torch.cat(data["pred"], dim=0)

        # Convert stored true/pred values to class labels
        # True labels
        if yt.dim() == 1:  # [B] (binary labels or probabilities)
            true_lab = (yt > 0.5).long().cpu()
        else:  # [B, C] (one-hot or logits)
            true_lab = yt.argmax(dim=-1).long().cpu()

        # Predicted labels
        if yp.dim() == 1:  # [B] (logits or probabilities)
            pred_lab = (torch.sigmoid(yp) > 0.5).long().cpu()
        else:  # [B, C] (logits)
            pred_lab = yp.argmax(dim=-1).long().cpu()

        idx_map = torch.searchsorted(uniq_eid, eids)

        true_mat[idx_map, col]    = true_lab.short()
        correct_mat[idx_map, col] = (true_lab == pred_lab).short()
        present_mat[idx_map, col] = 1
        col += 1

# Identify eids that are always correct or always incorrect across all runs,
# and infer the consistent true label per eid (averaged over appearances)
appears     = present_mat.sum(dim=1)                  # [N_id]
sum_correct = (correct_mat * present_mat).sum(dim=1)  # sum over appeared runs only

# Always correct: correct in all appearances
whole_correct_idx   = torch.where((appears > 0) & (sum_correct == appears))[0]
# Always incorrect: incorrect in all appearances
whole_incorrect_idx = torch.where((appears > 0) & (sum_correct == 0))[0]

# Infer true label per eid (average over appeared runs only)
mean_true = (true_mat.float() * present_mat).sum(dim=1) / torch.clamp(appears.float(), min=1.0)
true_label_consistent = (mean_true > 0.5).long()

# Define groups in index space (0 ~ N_ID-1), not in eid values
# g_all: eids that appeared at least once in any fold
g_all = torch.where(appears > 0)[0]

# Always-correct / always-wrong groups
g_correct = whole_correct_idx
g_wrong = whole_incorrect_idx

# Split groups by consistent true label
g_correct_top    = whole_correct_idx[true_label_consistent[whole_correct_idx] == 1]
g_correct_bottom = whole_correct_idx[true_label_consistent[whole_correct_idx] == 0]
g_wrong_top      = whole_incorrect_idx[true_label_consistent[whole_incorrect_idx] == 1]
g_wrong_bottom   = whole_incorrect_idx[true_label_consistent[whole_incorrect_idx] == 0]

print(
    "Group sizes:",
    f"all={len(g_all)}, "
    f"correct={len(g_correct)}, "
    f"wrong={len(g_wrong)}, "
    f"correct_top={len(g_correct_top)}, "
    f"correct_bottom={len(g_correct_bottom)}, "
    f"wrong_top={len(g_wrong_top)}, "
    f"wrong_bottom={len(g_wrong_bottom)}"
)

# Aggregate attributions per group (magnitude and signed)
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

eid_sum_mag  = torch.zeros(N_ID, F, dtype=torch.float32)
eid_sum_sign = torch.zeros(N_ID, F, dtype=torch.float32)
eid_cnt      = torch.zeros(N_ID, dtype=torch.long)

for it in range(1, N_ITER + 1):
    for fd in range(1, N_FOLD + 1):
        data = torch.load(os.path.join(root_path, f'Iteration_{it}/Fold_{fd}/attributions.pt'), weights_only=False)
        eids = torch.cat(data["eid"], dim=0).cpu()
        idx_map = torch.searchsorted(uniq_eid, eids)  # [B]

        # Load per-sample attributions
        attr_num = torch.cat(data["attr_cont"], dim=0)         # [B, D_num]
        attr_cat  = torch.cat(data["attr_cat"], dim=0)         # [B, K_cat, D_emb]
        print('attr_num: ', attr_num.shape)
        print('attr_cat: ', attr_cat.shape)

        attr_num_sum = attr_num.sum(dim=-1)
        attr_cat_sum = attr_cat.sum(dim=-1)
        print('attr_num_sum: ', attr_num_sum.shape)
        print('attr_cat_sum: ', attr_cat_sum.shape)

        attr_tot_cat = torch.cat((attr_num_sum, attr_cat_sum), dim=-1)

        attr_all_mag  = attr_tot_cat.abs()                                # [B, F]
        attr_all_sign = attr_tot_cat                                      # [B, F]

        # Accumulate per-eid sums for macro-averaging
        eid_sum_mag.index_add_(0, idx_map, attr_all_mag.cpu())
        eid_sum_sign.index_add_(0, idx_map, attr_all_sign.cpu())
        # Each row corresponds to one appearance; increment per-eid counts
        eid_cnt.index_add_(0, idx_map, torch.ones(idx_map.size(0), dtype=torch.long))

# Compute per-eid mean attributions
cnt_safe = eid_cnt.clamp(min=1).unsqueeze(1)                            # [N_ID, 1]
eid_mean_mag  = eid_sum_mag  / cnt_safe                                 # [N_ID, F]
eid_mean_sign = eid_sum_sign / cnt_safe                                 # [N_ID, F]

# Compute group-level statistics from per-eid means
def group_mean(eid_idx_tensor):
    if eid_idx_tensor.numel() == 0:
        return None, None, None, None

    return (
        eid_mean_mag[eid_idx_tensor].mean(dim=0),   # mean magnitude
        eid_mean_sign[eid_idx_tensor].mean(dim=0),  # mean signed attribution
        eid_mean_mag[eid_idx_tensor].std(dim=0),    # std of magnitude
        eid_mean_sign[eid_idx_tensor].std(dim=0),   # std of signed attribution
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

feature_names = continue_col + category_col
variable_size = len(feature_names)

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

        # Plot raw importance for the top 10% features
        max_raw = int(len(x) * 0.1)
        x_raw = x[:max_raw]
        y_raw = sorted_imp[:max_raw]
        ax1.scatter(x_raw, y_raw, s=point_size)

        if knee_rank is not None and knee_rank <= max_raw:
            k = int(knee_rank)

            ax1.axvline(k, linestyle='--',
                        color=knee_color, alpha=knee_alpha)
            # Highlight the detected knee point
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

        # Plot the full importance curve (used for knee detection)
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

        # Select top-K features (K determined by knee rank; fallback to top-10)
        top_n_indices = indices[:n]
        print("Selected feature indices: ", top_n_indices)
        selected_vals = sorted_imp[:n]

        for rank in range(n):
            global_idx = top_n_indices[rank]   # 0 ~ F-1
            fname = feature_names[global_idx]  # original name

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