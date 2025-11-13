"""
Random Forest classification with SHAP TreeExplainer interpretation on UKB fluid intelligence (top/bottom 10%).

This script:
- Loads the UKB dataset and cross-validation splits (5 repeats × 5 folds).
- Trains a Random Forest classifier for each (iteration, fold) using different variable subsets.
- Evaluates classification performance (ACC, AUC, sensitivity, specificity).
- Computes TreeExplainer values for each test subject and feature.

Inputs:
- --variable_type: which feature subset to use
    (all / brain / health / socio / brain_health / brain_socio / health_socio)
- --json_path: path to the cross-validation split JSON file.
- --data_path: path to the input CSV file.
- --outdir: directory to save all results.

Outputs:
- Randomforest_shap_<variable_type>_final_result_value.csv
    Per-fold metrics (ACC, AUC, sensitivity, specificity) across all iterations/folds.
- Randomforest_shap_<variable_type>_all_iters_folds.csv
    Row-wise SHAP values and predictions for all test subjects across all iterations/folds.
- shap_iter<it>_fold<fold>.csv
    SHAP values and predictions for a specific (iteration, fold).
"""

import os, json, csv, argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from data_utils import select_data
import shap
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ------------------- Config (argparse) -------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Random Forest + SHAP on UKB fluid intelligence (top/bottom 10%)"
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
        default="/LocalData1/daheo/BRL/UKB_new/Model1_Tree_based/Randomforest_shap",
        help="Directory to save all results and SHAP outputs.",
    )
    return parser.parse_args()


args = parse_args()


variable_type = args.variable_type
json_path = args.json_path
data_path = args.data_path
outdir = os.path.join(args.outdir, variable_type)

os.makedirs(outdir, exist_ok=True)

save_result_csv = f"{outdir}/Randomforest_shap_{variable_type}_final_result_value.csv"
csv_save_all_iter_fold = f"{outdir}/Randomforest_shap_{variable_type}_all_iters_folds.csv"

# ------------------- Read data and basic settings -------------------
df = pd.read_csv(data_path)
df = df.fillna(0.0)

# Select feature columns for the given variable_type
category_col, continue_col, Categories = select_data(variable_type)

with open(json_path, "r") as f:
    ids = json.load(f)

eid_col = 'eid'
label_col = "fluid_2_p10"


def get_datalist(category_col, continue_col, data, eids, mean_cont, std_cont):
    """
    Subset data by subject IDs (eid), apply z-normalization to continuous features
    using the provided mean and std (computed from the train set), and return
    a DataFrame for X (continuous + categorical), labels, and sorted eids.

    Args:
        category_col: list or None
            Names of categorical feature columns. If None, only continuous columns are used.
        continue_col: list
            Names of continuous feature columns.
        data: pd.DataFrame
            Full dataset.
        eids: list or array-like
            Subject IDs to subset.
        mean_cont: np.ndarray
            Mean of continuous features from the training set. Shape (1, n_features) or (n_features,).
        std_cont: np.ndarray
            Std of continuous features from the training set. Shape (1, n_features) or (n_features,).

    Returns:
        X_data: pd.DataFrame
            Feature DataFrame containing categorical and continuous columns.
        y: np.ndarray
            Binary labels (0/1) for the selected subjects.
        eids_sorted: np.ndarray
            Subject IDs sorted by eid.
    """
    # Subset by eid, sort to keep a deterministic order
    sub = data.loc[data[eid_col].isin(eids)].copy()
    sub = sub.sort_values(eid_col).reset_index(drop=True)

    if category_col is None:
        feature_cols = continue_col
    else:
        feature_cols = category_col + continue_col

    # Make sure we work on an independent DataFrame
    X_data = sub[feature_cols].copy()

    # Flatten mean and std to length-(n_features) vectors
    mean_flat = np.asarray(mean_cont).reshape(-1)
    std_flat = np.asarray(std_cont).reshape(-1)

    # Convert to Series for column-wise alignment with continue_col
    mean_series = pd.Series(mean_flat, index=continue_col)
    std_series = pd.Series(std_flat, index=continue_col)

    # Z-normalization for continuous features (categoricals remain unchanged)
    X_data[continue_col] = (X_data[continue_col] - mean_series) / std_series

    y = sub[label_col].astype(np.int64).to_numpy()
    eids_sorted = sub[eid_col].to_numpy()
    return X_data, y, eids_sorted


all_rows = []
all_iter_acc, all_iter_auc, all_iter_sen, all_iter_spc = [], [], [], []

# ------------------- Cross-validation loop (5 repeats × 5 folds) -------------------
for it in range(5):
    fold_acc, fold_auc, fold_sen, fold_spc = [], [], [], []

    for fold in range(5):
        itfold = ids["iterations"][it]["folds"][fold]
        train_eids = itfold["train_eid"]
        test_eids = itfold["valid_eid"]

        # Extract training subset to compute normalization statistics
        train_df_full = (
            df.loc[df[eid_col].isin(train_eids)]
            .copy()
            .sort_values(eid_col)
            .reset_index(drop=True)
        )
        train_cont = train_df_full[continue_col].to_numpy(dtype=np.float32)

        # ---- Compute mean and std from the training set for z-normalization ----
        mean_cont = train_cont.mean(axis=0, keepdims=True)
        std_cont = train_cont.std(axis=0, ddof=0, keepdims=True)
        std_cont[std_cont == 0.0] = 1.0  # avoid division by zero

        # ColumnTransformer for preprocessing:
        # - One-hot encode categorical features
        # - Pass through z-normalized continuous features
        if variable_type != 'brain':
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), category_col),
                    ("num", "passthrough", continue_col),
                ]
            )
        else:
            # brain-only case: only continuous (imaging) features
            preprocessor = ColumnTransformer(
                transformers=[("num", "passthrough", continue_col)]
            )

        # Random Forest classifier (classification task)
        model = RandomForestClassifier(random_state=42)

        # Pipeline = preprocessing + model
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

        # Build train / test sets with z-normalized continuous features
        train_X, train_y, _ = get_datalist(
            category_col, continue_col, df, train_eids, mean_cont, std_cont
        )
        test_X, test_y, test_eids_sorted = get_datalist(
            category_col, continue_col, df, test_eids, mean_cont, std_cont
        )

        # Fit Random Forest via pipeline
        pipeline.fit(train_X, train_y)

        # Predict on test set via pipeline
        y_pred_proba = pipeline.predict_proba(test_X)[:, 1]
        y_pred = pipeline.predict(test_X)

        # ---- Evaluation metrics ----
        acc = accuracy_score(test_y, y_pred)
        auc = roc_auc_score(test_y, y_pred_proba)

        tn, fp, fn, tp = confusion_matrix(test_y, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall / TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR

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

        # Append per-fold metrics to CSV (write header once)
        write_header = not os.path.exists(save_result_csv)
        with open(save_result_csv, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(result_row)

        fold_acc.append(acc)
        fold_auc.append(auc)
        fold_sen.append(sensitivity)
        fold_spc.append(specificity)

        # =================== SHAP interpretation per subject ===================
        # 1) Extract trained RandomForest and preprocessor from the pipeline
        rf_model = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # 2) Apply the same preprocessing to the test set
        #    (OneHot encoding for categoricals + passthrough for continuous)
        X_test_trans = preprocessor.transform(test_X)

        # OneHotEncoder often returns a sparse matrix → convert to dense
        if hasattr(X_test_trans, "toarray"):
            X_test_dense = X_test_trans.toarray()
        else:
            X_test_dense = X_test_trans

        # 3) Compute SHAP values using TreeExplainer on the trained Random Forest
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test_dense)

        # Binary classification: shap_values can be [class0, class1]
        if isinstance(shap_values, list):
            # Use SHAP values for the positive class (label 1)
            shap_values = shap_values[1]

        shap_values = np.squeeze(shap_values)

        # 4) Retrieve feature names after preprocessing
        #    e.g., 'cat__alcohol_2_3.0', 'num__age_2', ...
        feature_names = preprocessor.get_feature_names_out()

        # 5) Store per-subject SHAP values and predictions
        rows = []
        for i in range(X_test_dense.shape[0]):
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
            # Add SHAP value for each preprocessed feature
            for j, fname in enumerate(feature_names):
                row[f"shap::{fname}"] = float(shap_values[i, j])
            rows.append(row)

        df_fold = pd.DataFrame(rows)
        df_fold.to_csv(os.path.join(outdir, f"shap_iter{it}_fold{fold}.csv"), index=False)
        all_rows.extend(rows)

    # Per-iteration averages across the 5 folds
    all_iter_acc.append(float(np.mean(fold_acc)))
    all_iter_auc.append(float(np.mean(fold_auc)))
    all_iter_sen.append(float(np.mean(fold_sen)))
    all_iter_spc.append(float(np.mean(fold_spc)))

# ------------------- Final summary across iterations -------------------
all_iter_acc = np.array(all_iter_acc)
all_iter_auc = np.array(all_iter_auc)
all_iter_sen = np.array(all_iter_sen)
all_iter_spc = np.array(all_iter_spc)

print(
    f"Average acc={np.mean(all_iter_acc):.4f}, auc={np.mean(all_iter_auc):.4f}, "
    f"sen={np.mean(all_iter_sen):.4f}, spc={np.mean(all_iter_spc):.4f}"
)
print(
    f"STD acc={np.std(all_iter_acc):.4f}, auc={np.std(all_iter_auc):.4f}, "
    f"sen={np.std(all_iter_sen):.4f}, spc={np.std(all_iter_spc):.4f}"
)

df_all = pd.DataFrame(all_rows)
df_all.to_csv(csv_save_all_iter_fold, index=False)
