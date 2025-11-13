"""
LightGBM classification with SHAP TreeExplainer interpretation on UKB fluid intelligence (top/bottom 10%).

This script:
- Loads the UKB dataset and cross-validation splits (5 repeats × 5 folds).
- Trains a LightGBM classifier for each (iteration, fold) using different variable subsets.
- Evaluates classification performance (ACC, AUC, sensitivity, specificity).
- Computes TreeExplainer values for each test subject and feature.

Inputs:
- --variable_type: which feature subset to use (all / brain / health / socio / brain_health / brain_socio / health_socio)
- --json_path: path to the cross-validation split JSON file.
- --data_path: path to the input CSV file.
- --outdir: directory to save all results.

Outputs:
- Lightgbm_shap_<variable_type>_final_result_value.csv
    Per-fold metrics (ACC, AUC, sensitivity, specificity) across all iterations/folds.
- Lightgbm_shap_<variable_type>_all_iters_folds.csv
    Row-wise SHAP values and predictions for all test subjects across all iterations/folds.
- shap_iter<it>_fold<fold>.csv
    SHAP values and predictions for a specific (iteration, fold).
"""

import argparse
import os, json, csv
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from data_utils import select_data
import shap

# ------------------- Config (argparse) -------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="LightGBM + SHAP on UKB fluid intelligence (top/bottom 10%)"
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
        default="/LocalData1/daheo/data/BRL/UKB_new/Iter_5_Folds_5.json",
        help="Path to cross-validation split JSON file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/LocalData1/daheo/data/BRL/UKB_new/Step5_refilter_categorical_for_deeplearning.csv",
        help="Path to input CSV data.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/LocalData1/daheo/BRL/UKB_new/Model1_Tree_based/Lightgbm_shap",
        help="Directory to save all results and SHAP outputs.",
    )
    return parser.parse_args()


args = parse_args()

variable_type = args.variable_type
json_path = args.json_path
data_path = args.data_path
outdir = os.path.join(args.outdir, variable_type)


os.makedirs(outdir, exist_ok=True)

save_result_csv = os.path.join(outdir, f"Lightgbm_shap_{variable_type}_final_result_value.csv")
csv_save_all_iter_fold = f"{outdir}/Lightgbm_shap_{variable_type}_all_iters_folds.csv"


# ------------------- Read Data -------------------
df = pd.read_csv(data_path)
category_col, continue_col, Categories = select_data(variable_type)

# Continuous / categorical feature indices
if variable_type != 'brain':
    n_con = len(continue_col)
    n_cat = len(category_col)
    # categorical features come after continuous features when concatenated
    cat_idx = list(range(n_con, n_con + n_cat))

with open(json_path, "r") as f:
    ids = json.load(f)

eid_col = 'eid'
label_col = "fluid_2_p10"

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

for it in range(5):
    fold_acc, fold_auc, fold_sen, fold_spc = [], [], [], []

    for fold in range(5):
        itfold = ids["iterations"][it]["folds"][fold]
        train_eids = itfold["train_eid"]
        test_eids = itfold["valid_eid"]

        train_df = df.loc[df[eid_col].isin(train_eids)].copy().sort_values(eid_col)
        train_cont = train_df[continue_col].to_numpy(dtype=np.float32)

        # ---- Compute mean and std from the training set for z-normalization ----
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
            X_test = np.concatenate((test_con, test_cat), axis=-1)

            # categorical_feature should be indices (column positions)
            train_ds = lgb.Dataset(X_train, label=train_y, categorical_feature=cat_idx,)
            test_ds = lgb.Dataset(X_test, label=test_y, categorical_feature=cat_idx, reference=train_ds,)
        else:
            X_train = train_con
            X_test = test_con

            train_ds = lgb.Dataset(X_train, label=train_y,)
            test_ds = lgb.Dataset(X_test, label=test_y, reference=train_ds,)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "learning_rate": 0.05,
            "verbose": -1
        }

        # Use test set as validation set for monitoring
        model = lgb.train(
            params,
            train_ds,
            num_boost_round=500,
            valid_sets=[train_ds, test_ds],
            valid_names=["train", "valid"],
        )

        # --- Prediction ---
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        acc = accuracy_score(test_y, y_pred)
        auc = roc_auc_score(test_y, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(test_y, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(
            f"[iter {it} fold {fold}] acc={acc:.4f}, auc={auc:.4f}, "
            f"sen={sensitivity:.4f}, spc={specificity:.4f}"
        )

        result_row = {
            "iter": it,
            "fold": fold,
            "acc": acc,
            "auc": auc,
            "sen": sensitivity,
            "spe": specificity,
        }

        # If file does not exist → write header; otherwise append without header
        write_header = not os.path.exists(save_result_csv)
        with open(save_result_csv, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(result_row)

        fold_acc.append(acc)
        fold_auc.append(auc)
        fold_sen.append(sensitivity)
        fold_spc.append(specificity)

        # ===== SHAP =====
        if variable_type != 'brain':
            feature_names = (
                    [f"cont_{c}" for c in continue_col]
                    + [f"cat_{c}" for c in category_col]
            )
        else:
            feature_names = [f"cont_{c}" for c in continue_col]

        model.feature_name_ = feature_names
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # In some SHAP versions for binary classification, shap_values is a list [class0, class1].
        if isinstance(shap_values, list):
            # Common choice: explanations for the positive class (index 1)
            shap_values = shap_values[-1]

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
            for j, fname in enumerate(feature_names):
                row[f"shap::{fname}"] = float(shap_values[i, j])
            rows.append(row)

        pd.DataFrame(rows).to_csv(f"{outdir}/shap_iter{it}_fold{fold}.csv",index=False,)

        all_rows.extend(rows)

        # Per-iteration mean performance
    all_iter_acc.append(np.mean(fold_acc))
    all_iter_auc.append(np.mean(fold_auc))
    all_iter_sen.append(np.mean(fold_sen))
    all_iter_spc.append(np.mean(fold_spc))

# Final summary across iterations
print(
    f"Average acc={np.mean(all_iter_acc):.4f}, auc={np.mean(all_iter_auc):.4f}, "
    f"sen={np.mean(all_iter_sen):.4f}, spc={np.mean(all_iter_spc):.4f}"
)
print(
    f"STD acc={np.std(all_iter_acc):.4f}, auc={np.std(all_iter_auc):.4f}, "
    f"sen={np.std(all_iter_sen):.4f}, spc={np.std(all_iter_spc):.4f}"
)

pd.DataFrame(all_rows).to_csv(csv_save_all_iter_fold, index=False)
