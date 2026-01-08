"""
LightGBM classification with SHAP TreeExplainer interpretation on UKB fluid intelligence (top/bottom 10%).

This script:
- Loads the UKB dataset and cross-validation splits (5 repeats × 5 folds).
- Trains a LightGBM classifier for each (iteration, fold) using different variable subsets.
- Evaluates classification performance (ACC, AUC, sensitivity, specificity).
- Computes TreeExplainer values for each test subject and feature.

Inputs:
- --variable_type: which feature subset to use
    (all / brain / health / socio / brain_health / brain_socio / health_socio)
- --json_path: path to the cross-validation split JSON file.
- --data_path: path to the input CSV file.
- --outdir: directory to save all results.

Outputs:
- Outputs are saved under: {outdir}/{cls_type}/{variable_type}/{parameter_name}/
- Lightgbm_shap_{variable_type}_final_result_value.csv
    Per-fold metrics (ACC, AUC, sensitivity, specificity) across all iterations/folds.
- Lightgbm_shap_{variable_type}_all_iters_folds.csv
    Row-wise SHAP values and predictions for all test subjects across all iterations/folds.
- shap_iter{it}_fold{fold}.csv
    SHAP values and predictions for a specific (iteration, fold).
"""

import argparse
import os, json, csv
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from data_utils import select_data_gf_cls, select_data_edu_cls
import shap
from sklearn.model_selection import StratifiedShuffleSplit

# ------------------- Config (argparse) -------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="LightGBM + SHAP on UKB fluid intelligence (top/bottom 10%)"
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
        "--json_path",
        type=str,
        default="./data/Iter_5_Folds_5.json",
        help="Path to cross-validation split JSON file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/Step5_refilter_categorical_for_deeplearning.csv",
        help="Path to input CSV data.",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="./results/Tree_based/Lightgbm_shap_plz",
        help="Directory to save all results and SHAP outputs.",
    )
    parser.add_argument(
        "--max_depth", type=int, default=3,
        help="Max tree depth. -1 means no limit. Typical: -1 or 3–8."
    )
    parser.add_argument(
        "--num_leaves", type=int, default=63,
        help="Max leaves per tree (complexity). Typical: 15–255."
    )
    parser.add_argument(
        "--min_data_in_leaf", type=int, default=20,
        help="Min samples per leaf (regularization). Typical: 5–100."
    )
    parser.add_argument(
        "--num_iterations", type=int, default=800,
        help="Number of boosting rounds. Typical: 200–2000. (Use early_stopping if possible.)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01,
        help="Shrinkage rate. Typical: 0.01–0.1 (smaller -> need more iterations)."
    )
    parser.add_argument(
        "--lambda_l1", type=float, default=0.0,
        help="L1 regularization. Typical: 0–10 (try 0, 0.1, 1.0)."
    )
    parser.add_argument(
        "--lambda_l2", type=float, default=5.0,
        help="L2 regularization. Typical: 0–10 (try 0, 0.1, 1.0)."
    )
    return parser.parse_args()

args = parse_args()

variable_type = args.variable_type
json_path = args.json_path
data_path = args.data_path
parameter_name = (f"md{args.max_depth}_nl{args.num_leaves}_lr{args.learning_rate}_"
                  f"ni{args.num_iterations}_mdl{args.min_data_in_leaf}_"
                  f"l1{args.lambda_l1}_l2{args.lambda_l2}")
outdir = os.path.join(args.outdir, args.cls_type, variable_type, parameter_name)

os.makedirs(outdir, exist_ok=True)

save_result_csv                 = f"{outdir}/Lightgbm_shap_{variable_type}_final_result_value.csv"
csv_save_all_iter_fold          = f"{outdir}/Lightgbm_shap_{variable_type}_all_iters_folds.csv"
csv_save_total_averaged_results = f"{args.outdir}/Lightgbm_shap_averaged_total_results.csv"

# ------------------- Read Data -------------------
df = pd.read_csv(data_path)
df = df.fillna(0.0)

eid_col = 'eid'
if args.cls_type == 'gf':
    label_col = "fluid_2_p10"
    category_col, continue_col, Categories = select_data_gf_cls(variable_type)
elif args.cls_type == 'edu':
    label_col = "ed_b_2"
    category_col, continue_col, Categories = select_data_edu_cls(variable_type)

# Continuous / categorical feature indices
if variable_type != 'brain' and category_col is not None and len(category_col) > 0:
    n_con = len(continue_col)
    n_cat = len(category_col)
    # categorical features come after continuous features when concatenated
    cat_idx = list(range(n_con, n_con + n_cat))
else:
    cat_idx = None

with open(json_path, "r") as f:
    ids = json.load(f)

def get_datalist(variable_type, category_col, continue_col, data, eids, mean_cont, std_cont):
    """
    Subset data by subject IDs (eid), apply z-normalization to continuous features
    using the provided mean and std (computed from the train set), and return
    numpy arrays for X (continuous and categorical), y, and sorted eids.

    Returns:
        X_cat: np.ndarray or None
            Categorical features (int64) if variable_type != 'brain', otherwise None.
        X_con: np.ndarray
            Z-normalized continuous features (float32).
        y: np.ndarray
            Binary labels (0/1).
        eids_sorted: np.ndarray
            Corresponding subject IDs, sorted by eid.
    """
    sub = data.loc[data[eid_col].isin(eids)].copy()
    sub = sub.sort_values(eid_col).reset_index(drop=True)

    X_con = sub[continue_col].to_numpy(dtype=np.float32)
    X_con = (X_con - mean_cont) / std_cont

    y = sub[label_col].astype(np.int64).to_numpy()
    eids_sorted = sub[eid_col].to_numpy()

    if variable_type != 'brain':
        X_cat = sub[category_col].to_numpy(dtype=np.int64)
        return X_cat, X_con, y, eids_sorted
    else:
        return None, X_con, y, eids_sorted


all_rows = []
all_iter_acc, all_iter_auc, all_iter_sen, all_iter_spc = [], [], [], []
val_all_iter_acc, val_all_iter_auc, val_all_iter_sen, val_all_iter_spc = [], [], [], []

for it in range(5):
    fold_acc, fold_auc, fold_sen, fold_spc = [], [], [], []
    val_fold_acc, val_fold_auc, val_fold_sen, val_fold_spc = [], [], [], []

    for fold in range(5):
        itfold = ids["iterations"][it]["folds"][fold]
        train_val_eids = itfold["train_eid"]
        test_eids = itfold["test_eid"]

        # train_val dataframe
        train_valid_df = df.loc[df[eid_col].isin(train_val_eids)].copy().sort_values(eid_col)
        train_valid_eid = train_valid_df[eid_col].to_numpy()
        train_valid_labels = train_valid_df[label_col].to_numpy()

        # Split validation (Stratified 10%)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, valid_idx = next(sss.split(train_valid_eid, train_valid_labels))

        train_eids = train_valid_eid[train_idx].tolist()
        valid_eids = train_valid_eid[valid_idx].tolist()

        train_df = df.loc[df[eid_col].isin(train_eids)].copy().sort_values(eid_col)
        train_cont = train_df[continue_col].to_numpy(dtype=np.float32)

        # Compute mean and std from the training set for z-normalization
        mean_cont = train_cont.mean(axis=0, keepdims=True)
        std_cont = train_cont.std(axis=0, ddof=0, keepdims=True)
        std_cont[std_cont == 0.0] = 1.0  # avoid division by zero

        train_cat, train_con, train_y, _ = get_datalist(
            variable_type,
            category_col,
            continue_col,
            df,
            train_eids,
            mean_cont.astype(np.float32),
            std_cont.astype(np.float32),
        )
        valid_cat, valid_con, valid_y, valid_eids_sorted = get_datalist(
            variable_type,
            category_col,
            continue_col,
            df,
            valid_eids,
            mean_cont.astype(np.float32),
            std_cont.astype(np.float32),
        )
        test_cat, test_con, test_y, test_eids_sorted = get_datalist(
            variable_type,
            category_col,
            continue_col,
            df,
            test_eids,
            mean_cont.astype(np.float32),
            std_cont.astype(np.float32),
        )

        # Concatenate features: continuous first, then categorical
        if variable_type != 'brain':
            X_train = np.concatenate((train_con, train_cat), axis=-1)
            X_valid = np.concatenate((valid_con, valid_cat), axis=-1)
            X_test = np.concatenate((test_con, test_cat), axis=-1)

            train_ds = lgb.Dataset(X_train, label=train_y, categorical_feature=cat_idx)
            valid_ds = lgb.Dataset(X_valid, label=valid_y, categorical_feature=cat_idx, reference=train_ds)
            test_ds = lgb.Dataset(X_test, label=test_y, categorical_feature=cat_idx, reference=train_ds)
        else:
            X_train, X_valid, X_test = train_con, valid_con, test_con
            train_ds = lgb.Dataset(X_train, label=train_y)
            valid_ds = lgb.Dataset(X_valid, label=valid_y, reference=train_ds)
            test_ds = lgb.Dataset(X_test, label=test_y, reference=train_ds)

        params = {
            "objective": "binary",
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_data_in_leaf": args.min_data_in_leaf,
            "num_leaves": args.num_leaves,
            "boosting": "gbdt",
            "bagging_fraction": 1.0,
            "feature_fraction": 1.0,
            "lambda_l1": args.lambda_l1,
            "lambda_l2": args.lambda_l2,
            "metric": "auc",
        }

        model = lgb.train(
            params,
            train_ds,
            num_boost_round=args.num_iterations,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "valid"],
        )

        # Predict on validation and test set
        y_pred_proba_val = model.predict(X_valid)
        y_pred_val = (y_pred_proba_val > 0.5).astype(int)
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Evaluation metrics
        acc_val = accuracy_score(valid_y, y_pred_val)
        auc_val = roc_auc_score(valid_y, y_pred_proba_val)
        acc = accuracy_score(test_y, y_pred)
        auc = roc_auc_score(test_y, y_pred_proba)

        tn_val, fp_val, fn_val, tp_val = confusion_matrix(valid_y, y_pred_val, labels=[0, 1]).ravel()
        tn, fp, fn, tp = confusion_matrix(test_y, y_pred, labels=[0, 1]).ravel()
        sensitivity_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        specificity_val = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(
            f"[iter {it} fold {fold}] acc={acc:.4f}, auc={auc:.4f}, "
            f"sen={sensitivity:.4f}, spc={specificity:.4f}"
        )

        result_row = {
            "iter": it,
            "fold": fold,
            "acc_val": acc_val,
            "auc_val": auc_val,
            "sen_val": sensitivity_val,
            "spe_val": specificity_val,
            "acc": acc,
            "auc": auc,
            "sen": sensitivity,
            "spe": specificity,
        }

        # Append per-fold metrics to CSV
        write_header = not os.path.exists(save_result_csv)
        with open(save_result_csv, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(result_row)

        val_fold_acc.append(acc_val)
        val_fold_auc.append(auc_val)
        val_fold_sen.append(sensitivity_val)
        val_fold_spc.append(specificity_val)

        fold_acc.append(acc)
        fold_auc.append(auc)
        fold_sen.append(sensitivity)
        fold_spc.append(specificity)

        if variable_type != 'brain':
            feature_names = (
                    [f"cont_{c}" for c in continue_col] + [f"cat_{c}" for c in category_col]
            )
        else:
            feature_names = [f"cont_{c}" for c in continue_col]

        # Compute SHAP values using TreeExplainer
        # model.feature_name_ = feature_names
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            # Explanations for the positive class (index 1)
            shap_values = shap_values[-1]

        # Store per-subject SHAP values and predictions
        rows = []
        for i in range(X_test.shape[0]):
            row = {
                "iteration": it,
                "fold": fold,
                "eid": int(test_eids_sorted[i]),
                "true_label": int(test_y[i]),
                "pred_label": int(y_pred[i]),
                "pred_proba": float(y_pred_proba[i]),
                "acc_fold": float(acc),
                "auc_fold": float(auc),
                "sens_fold": float(sensitivity),
                "spec_fold": float(specificity),
            }
            # Add SHAP value for each feature
            for j, fname in enumerate(feature_names):
                row[f"shap::{fname}"] = float(shap_values[i, j])
            rows.append(row)

        pd.DataFrame(rows).to_csv(f"{outdir}/shap_iter{it}_fold{fold}.csv",index=False,)

        all_rows.extend(rows)

    # Per-iteration averages across the 5 folds
    all_iter_acc.append(np.mean(fold_acc))
    all_iter_auc.append(np.mean(fold_auc))
    all_iter_sen.append(np.mean(fold_sen))
    all_iter_spc.append(np.mean(fold_spc))

    val_all_iter_acc.append(float(np.mean(val_fold_acc)))
    val_all_iter_auc.append(float(np.mean(val_fold_auc)))
    val_all_iter_sen.append(float(np.mean(val_fold_sen)))
    val_all_iter_spc.append(float(np.mean(val_fold_spc)))


# Final summary across iterations
all_iter_acc = np.array(all_iter_acc)
all_iter_auc = np.array(all_iter_auc)
all_iter_sen = np.array(all_iter_sen)
all_iter_spc = np.array(all_iter_spc)

val_all_iter_acc = np.array(val_all_iter_acc)
val_all_iter_auc = np.array(val_all_iter_auc)
val_all_iter_sen = np.array(val_all_iter_sen)
val_all_iter_spc = np.array(val_all_iter_spc)

print(
    f"Average acc={np.mean(all_iter_acc):.4f}, auc={np.mean(all_iter_auc):.4f}, "
    f"sen={np.mean(all_iter_sen):.4f}, spc={np.mean(all_iter_spc):.4f}"
)
print(
    f"STD acc={np.std(all_iter_acc, ddof=1):.4f}, auc={np.std(all_iter_auc, ddof=1):.4f}, "
    f"sen={np.std(all_iter_sen, ddof=1):.4f}, spc={np.std(all_iter_spc, ddof=1):.4f}"
)

df_all = pd.DataFrame(all_rows)
df_all.to_csv(csv_save_all_iter_fold, index=False)

tot_avg_cols = ['variable',
                'val mean auc', 'val mean acc', 'val mean sen', 'val mean spc', 'val std auc', 'val std acc', 'val std sen', 'val std spc',
                'mean auc', 'mean acc', 'mean sen', 'mean spc', 'std auc', 'std acc', 'std sen', 'std spc',
                'max_depth', 'num_leaves', 'learning_rate', 'num_iterations', 'min_data_in_leaf', 'lambda_l1', 'lambda_l2']
tot_avg_vals = [variable_type,
                np.mean(val_all_iter_auc), np.mean(val_all_iter_acc), np.mean(val_all_iter_sen), np.mean(val_all_iter_spc),
                np.std(val_all_iter_auc, ddof=1), np.std(val_all_iter_acc, ddof=1), np.std(val_all_iter_sen, ddof=1),
                np.std(val_all_iter_spc, ddof=1),
                np.mean(all_iter_auc), np.mean(all_iter_acc), np.mean(all_iter_sen), np.mean(all_iter_spc),
                np.std(all_iter_auc, ddof=1), np.std(all_iter_acc, ddof=1), np.std(all_iter_sen, ddof=1),
                np.std(all_iter_spc, ddof=1),
                args.max_depth, args.num_leaves, args.learning_rate, args.num_iterations, args.min_data_in_leaf,
                args.lambda_l1, args.lambda_l2]

df_row = pd.DataFrame([tot_avg_vals], columns=tot_avg_cols)

file_exists = os.path.isfile(csv_save_total_averaged_results)

df_row.to_csv(
    csv_save_total_averaged_results,
    mode='a' if file_exists else 'w',
    header=not file_exists,
    index=False)
