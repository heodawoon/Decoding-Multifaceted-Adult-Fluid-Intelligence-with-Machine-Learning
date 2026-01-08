"""
Random Forest classification with SHAP interpretation on UKB fluid intelligence (top/bottom 10%).

This script:
- Loads the UKB dataset and cross-validation splits (5 repeats × 5 folds).
- Trains a Random Forest classifier for each (iteration, fold) using different variable subsets.
- Evaluates classification performance (ACC, AUC, sensitivity, specificity).
- Computes SHAP values for each subject using a tree-based SHAP explainer.

Inputs:
- --variable_type: which feature subset to use
    (all / brain / health / socio / brain_health / brain_socio / health_socio)
- --json_path: path to the cross-validation split JSON file.
- --data_path: path to the input CSV file.
- --outdir: directory to save all results.

Outputs:
- Outputs are saved under: {outdir}/{cls_type}/{variable_type}/{parameter_name}/
- Randomforest_shap_{variable_type}_final_result_value.csv
    Per-fold metrics (ACC, AUC, sensitivity, specificity) across all iterations/folds.
- Randomforest_shap_{variable_type}_all_iters_folds.csv
    Row-wise SHAP values and predictions for all test subjects across all iterations/folds.
- shap_iter{it}_fold{fold}.csv
    SHAP values and predictions for a specific (iteration, fold).
"""

import os, json, csv, argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from data_utils import select_data_gf_cls, select_data_edu_cls
import shap
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit


# ------------------- Config (argparse) -------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Random Forest + SHAP on UKB fluid intelligence (top/bottom 10%)"
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
        "--outdir",
        type=str,
        default="./results/Tree_based/Randomforest_shap_plz",
        help="Directory to save all results and SHAP outputs.",
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=800,
        help="Number of trees. More trees reduce variance but increase time. Typical: 200–2000 (e.g., 500, 1000).",
    )
    parser.add_argument(
        "--max_depth",
        type=str,
        default=40,
        help="Maximum tree depth. None = fully grown (can overfit). Typical: 5–50 or None (e.g., 10, 20, 30).",
    )
    parser.add_argument(
        "--max_features",
        type=str,
        default=0.3,
        choices=["sqrt", "log2", "1.0", "0.5", "0.3"],
        help=(
            "Number of features to consider at each split. "
            "Use 'sqrt'/'log2' or a fraction as string (e.g., '0.5'). Strong impact on generalization."
        ),
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=5,
        help="Minimum samples required at a leaf node. Strong regularizer. Typical: 1–20 (e.g., 1, 2, 5, 10).",
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Minimum samples required to split an internal node. Typical: 2–50 (e.g., 2, 5, 10, 20).",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        choices=["None", "balanced", "balanced_subsample"],
        help="Class weighting for imbalanced data. Use 'balanced' or 'balanced_subsample' if classes are skewed.",
    )
    parser.add_argument(
        "--max_samples",
        type=str,
        default=0.9,
        help=(
            "If bootstrap=True, number of samples to draw for each tree. "
            "Float in (0,1] = fraction of training set. Typical: 0.5–1.0 (e.g., 0.7, 0.9)."
        ),
    )
    return parser.parse_args()


args = parse_args()

max_depth    = None if args.max_depth == "None" else int(args.max_depth)
class_weight = None if args.class_weight == "None" else args.class_weight
max_samples  = None if args.max_samples == "None" else float(args.max_samples)

if args.max_features in ["sqrt", "log2"]:
    max_features = args.max_features
else:
    max_features = float(args.max_features)

variable_type = args.variable_type
json_path = args.json_path
data_path = args.data_path
parameter_name = (f'ne{args.n_estimators}_md{args.max_depth}'
                  f'_mf{args.max_features}_msl{args.min_samples_leaf}'
                  f'_mss{args.min_samples_split}_cw{args.class_weight}'
                  f'_ms{max_samples}')
outdir = os.path.join(args.outdir, args.cls_type, variable_type, parameter_name)

os.makedirs(outdir, exist_ok=True)

save_result_csv                 = f"{outdir}/Randomforest_shap_{variable_type}_final_result_value.csv"
csv_save_all_iter_fold          = f"{outdir}/Randomforest_shap_{variable_type}_all_iters_folds.csv"
csv_save_total_averaged_results = f"{args.outdir}/Randomforest_shap_averaged_total_results.csv"

# ------------------- Read data and basic settings -------------------
df = pd.read_csv(data_path)

# Select feature columns for the given variable_type
eid_col = 'eid'
if args.cls_type == 'gf':
    label_col = "fluid_2_p10"
    category_col, continue_col, Categories = select_data_gf_cls(variable_type)
elif args.cls_type == 'edu':
    label_col = "ed_b_2"
    category_col, continue_col, Categories = select_data_edu_cls(variable_type)

with open(json_path, "r") as f:
    ids = json.load(f)


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
val_all_iter_acc, val_all_iter_auc, val_all_iter_sen, val_all_iter_spc = [], [], [], []

# ------------------- Cross-validation loop -------------------
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

        # Extract training subset to compute normalization
        train_df_full = (
            df.loc[df[eid_col].isin(train_eids)]
            .copy()
            .sort_values(eid_col)
            .reset_index(drop=True)
        )
        train_cont = train_df_full[continue_col].to_numpy(dtype=np.float32)

        # Compute mean and std from the training set for z-normalization
        mean_cont = train_cont.mean(axis=0, keepdims=True)
        std_cont = train_cont.std(axis=0, ddof=0, keepdims=True)
        std_cont[std_cont == 0.0] = 1.0  # avoid division by zero

        # ColumnTransformer for preprocessing:
        #   One-hot encode categorical features
        #   Pass through z-normalized continuous features
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

        # Random Forest classifier
        model = RandomForestClassifier(random_state=42,
                                       n_estimators=args.n_estimators,
                                       max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=args.min_samples_leaf,
                                       min_samples_split=args.min_samples_split,
                                       class_weight=class_weight,
                                       max_samples=max_samples,)

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

        # Build train / test sets with z-normalized continuous features
        train_X, train_y, _ = get_datalist(
            category_col, continue_col, df, train_eids, mean_cont, std_cont
        )
        valid_X, valid_y, _ = get_datalist(
            category_col, continue_col, df, valid_eids, mean_cont, std_cont
        )
        test_X, test_y, test_eids_sorted = get_datalist(
            category_col, continue_col, df, test_eids, mean_cont, std_cont
        )

        # Fit Random Forest
        pipeline.fit(train_X, train_y)

        # Predict on validation and test set
        y_pred_proba_val = pipeline.predict_proba(valid_X)[:, 1]
        y_pred_val = pipeline.predict(valid_X)
        y_pred_proba = pipeline.predict_proba(test_X)[:, 1]
        y_pred = pipeline.predict(test_X)

        # Evaluation metrics
        acc_val = accuracy_score(valid_y, y_pred_val)
        auc_val = roc_auc_score(valid_y, y_pred_proba_val)
        acc = accuracy_score(test_y, y_pred)
        auc = roc_auc_score(test_y, y_pred_proba)

        tn_val, fp_val, fn_val, tp_val = confusion_matrix(valid_y, y_pred_val, labels=[0, 1]).ravel()
        sensitivity_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0  # recall / TPR
        specificity_val = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0.0  # TNR
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
        with open(save_result_csv, mode="a", newline="") as f:
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

        # SHAP interpretation per subject
        # Extract trained Random Forest and preprocessor from the pipeline
        rf_model = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Apply the same preprocessing to the test set
        X_test_trans = preprocessor.transform(test_X)

        if hasattr(X_test_trans, "toarray"):
            X_test_dense = X_test_trans.toarray()
        else:
            X_test_dense = X_test_trans

        # Compute SHAP values using tree-based SHAP via shap.Explainer
        # for each test subject and feature on the trained Random Forest
        explainer = shap.Explainer(rf_model)
        sv = explainer(X_test_dense)
        shap_values = sv.values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        shap_values = np.squeeze(shap_values)

        # Retrieve feature names after preprocessing
        #    e.g., 'cat__alcohol_2_3.0', 'num__age_2', ...
        feature_names = preprocessor.get_feature_names_out()

        # Store per-subject SHAP values and predictions
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
            # Add SHAP value for each feature
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

tot_avg_cols = ['variable', 'clstype',
                'val mean auc', 'val mean acc', 'val mean sen', 'val mean spc', 'val std auc', 'val std acc', 'val std sen', 'val std spc',
                'mean auc', 'mean acc', 'mean sen', 'mean spc', 'std auc', 'std acc', 'std sen', 'std spc',
                'n_estimators', 'max_depth', 'max_features', 'min_samples_leaf',
                'min_samples_split', 'class_weight', 'max_samples']
tot_avg_vals = [variable_type, args.cls_type,
                np.mean(val_all_iter_auc), np.mean(val_all_iter_acc), np.mean(val_all_iter_sen), np.mean(val_all_iter_spc),
                np.std(val_all_iter_auc, ddof=1), np.std(val_all_iter_acc, ddof=1), np.std(val_all_iter_sen, ddof=1),
                np.std(val_all_iter_spc, ddof=1),
                np.mean(all_iter_auc), np.mean(all_iter_acc), np.mean(all_iter_sen), np.mean(all_iter_spc),
                np.std(all_iter_auc, ddof=1), np.std(all_iter_acc, ddof=1), np.std(all_iter_sen, ddof=1),
                np.std(all_iter_spc, ddof=1),
                args.n_estimators, args.max_depth, args.max_features, args.min_samples_leaf,
                args.min_samples_split, args.class_weight, args.max_samples]

df_row = pd.DataFrame([tot_avg_vals], columns=tot_avg_cols)

file_exists = os.path.isfile(csv_save_total_averaged_results)

df_row.to_csv(
    csv_save_total_averaged_results,
    mode='a' if file_exists else 'w',
    header=not file_exists,
    index=False)
