"""
XGBoost classification with TreeSHAP interpretation on UKB fluid intelligence (top/bottom 10%).

This script:
- Loads the UKB dataset and cross-validation splits (5 repeats × 5 folds).
- Trains an XGBoost classifier for each (iteration, fold) using different variable subsets.
- Evaluates classification performance (ACC, AUC, sensitivity, specificity).
- Computes TreeSHAP feature contributions via booster.predict(pred_contribs=True).

Inputs:
- --variable_type: which feature subset to use (all / brain / health / socio / brain_health / brain_socio / health_socio)
- --json_path: path to the cross-validation split JSON file.
- --data_path: path to the input CSV file.
- --outdir: directory to save all results.
- --gpu: CUDA device index (-1 for CPU). If GPU is used, XGBoost runs with device='gpu'.

Outputs:
- Outputs are saved under: {outdir}/{cls_type}/{variable_type}/{parameter_name}/
- XGBoost_shap_<variable_type>_final_result_value.csv
    Per-fold metrics (ACC, AUC, sensitivity, specificity) across all iterations/folds.
- XGBoost_shap_<variable_type>_all_iters_folds.csv
    Row-wise SHAP values and predictions for all test subjects in all iterations/folds.
- shap_iter<it>_fold<fold>.csv
    SHAP values and predictions for a specific (iteration, fold).
"""

import os, json, csv, argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from data_utils import select_data_gf_cls, select_data_edu_cls
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


# ------------------- Config (argparse) -------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost + SHAP on UKB fluid intelligence (top/bottom 10%)"
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
        choices=["all", "brain", "health", "socio", "brain_health", "brain_socio", "health_socio"],
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
        default="./results/Tree_based/XGBoost_shap_plz",
        help="Directory to save all results and SHAP outputs.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="CUDA device index (used for CUDA_VISIBLE_DEVICES). Use -1 for CPU.",
    )
    parser.add_argument(
        "--tree_method",
        type=str,
        default="hist",
        choices=["hist", "approx", "exact"],  # (GPU면 gpu_hist도 고려)
        help="Tree construction algorithm. Mainly affects speed/memory; usually keep fixed (e.g., hist).",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=3,
        help=(
            "Maximum depth of individual trees. Controls model complexity; "
            "deeper trees can capture complex patterns but may overfit. "
            "Typical range: 2–10 (e.g., 3, 6)."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help=(
            "Step size shrinkage (eta) used in each boosting step. "
            "Lower values improve generalization but require more trees. "
            "Typical range: 0.005–0.3 (e.g., 0.01, 0.05, 0.1)."
        ),
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=800,
        help=(
            "Number of boosting rounds (trees). Works in trade-off with learning_rate. "
            "Typical range: 100–2000 (e.g., 100, 300, 800)."
        ),
    )

    # Regularization & overfitting control
    parser.add_argument(
        "--min_child_weight",
        type=float,
        default=5.0,
        help=(
            "Minimum sum of Hessian (approximate sample size) required in a child node. "
            "Larger values make the model more conservative. "
            "Typical range: 1–20 (e.g., 1, 3, 5)."
        ),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help=(
            "Minimum loss reduction required to make a split. "
            "Higher values lead to fewer splits and stronger regularization. "
            "Typical range: 0–5 (e.g., 0.0, 0.1)."
        ),
    )

    # Stochasticity (variance reduction)
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help=(
            "Fraction of training samples used for each tree. "
            "Introduces randomness to reduce overfitting. "
            "Typical range: 0.5–1.0 (e.g., 0.8, 0.9, 1.0)."
        ),
    )

    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=0.8,
        help=(
            "Fraction of features used when constructing each tree. "
            "Helps prevent overfitting and improves generalization. "
            "Typical range: 0.5–1.0 (0.8, 0.9, 1.0)."
        ),
    )

    # Weight regularization
    parser.add_argument(
        "--reg_alpha",
        type=float,
        default=0.2,
        help=(
            "L1 regularization term on leaf weights. Encourages sparsity. "
            "Useful for high-dimensional or noisy features. "
            "Typical range: 0–10 (e.g., 0.0, 0.2)."
        ),
    )

    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=1.0,
        help=(
            "L2 regularization term on leaf weights. Controls weight magnitude "
            "and stabilizes training. "
            "Typical range: 0–20 (e.g., 1.0, 2.0, 5.0)."
        ),
    )
    return parser.parse_args()

args = parse_args()
if args.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.tree_method == 'hist':
        device = 'gpu'
tree_method = args.tree_method

variable_type = args.variable_type
json_path = args.json_path
data_path = args.data_path
parameter_name = (f'{args.tree_method}_md{args.max_depth}_'
                  f'lr{args.learning_rate}_ne{args.n_estimators}_mcw{args.min_child_weight}'
                  f'_gam{args.gamma}_ss{args.subsample}_cb{args.colsample_bytree}_ra{args.reg_alpha}_rl{args.reg_lambda}')
outdir = os.path.join(args.outdir, args.cls_type, variable_type, parameter_name)

os.makedirs(outdir, exist_ok=True)

save_result_csv                 = f"{outdir}/XGBoost_shap_{variable_type}_final_result_value.csv"
csv_save_all_iter_fold          = f"{outdir}/XGBoost_shap_{variable_type}_all_iters_folds.csv"
csv_save_total_averaged_results = f"{args.outdir}/XGBoost_shap_averaged_total_validation_results.csv"

# ------------------- Read Data -------------------
df = pd.read_csv(data_path)

with open(json_path, "r") as f:
    ids = json.load(f)

eid_col = 'eid'
if args.cls_type == 'gf':
    label_col = "fluid_2_p10"
    category_col, continue_col, Categories = select_data_gf_cls(variable_type)
elif args.cls_type == 'edu':
    label_col = "ed_b_2"
    category_col, continue_col, Categories = select_data_edu_cls(variable_type)


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

        train_cat, train_con, train_y, _ = get_datalist(variable_type,
                                                        category_col, continue_col, df, train_eids,
                                                        mean_cont.astype(np.float32), std_cont.astype(np.float32)
                                                        )
        valid_cat, valid_con, valid_y, _ = get_datalist(variable_type,
                                                        category_col, continue_col, df, valid_eids,
                                                        mean_cont.astype(np.float32), std_cont.astype(np.float32)
                                                        )
        test_cat, test_con, test_y, test_eids_sorted = get_datalist(variable_type,
                                                                    category_col, continue_col, df, test_eids,
                                                                    mean_cont.astype(np.float32),
                                                                    std_cont.astype(np.float32)
                                                                    )

        if variable_type != 'brain':
            model = XGBClassifier(objective='binary:logistic',
                                  eval_metric='auc',
                                  random_state=42,
                                  tree_method=tree_method, device=device,
                                  enable_categorical=True,
                                  max_depth=args.max_depth,  # 3,  # 3, 6, 10
                                  learning_rate=args.learning_rate,  # 0.05,  # 0.01, 0.3
                                  n_estimators=args.n_estimators,  # 100,  # 100, 200, 800
                                  min_child_weight=args.min_child_weight,  # 3,  # 1, 10
                                  gamma=args.gamma,  # 0.1,  # 0.0, 1.0
                                  subsample=args.subsample,  # 0.9,  # 0.6, 1.0
                                  colsample_bytree=args.colsample_bytree,  # 0.9,  # 0.6, 1.0, 0.8
                                  reg_alpha=args.reg_alpha,  # 0.2,  # 0.0, 2.0
                                  reg_lambda=args.reg_lambda,  # 2.0,  # 0.5, 5.0, 1
                                  )
        else:
            # The model for brain only case:
            model = XGBClassifier(objective='binary:logistic',
                                  eval_metric='auc',
                                  random_state=42,
                                  tree_method=tree_method, device=device,
                                  max_depth=args.max_depth,  # 3,  # 3, 6, 10
                                  learning_rate=args.learning_rate,  # 0.05,  # 0.01, 0.3
                                  n_estimators=args.n_estimators,  # 100,  # 100, 200, 800
                                  min_child_weight=args.min_child_weight,  # 3,  # 1, 10
                                  gamma=args.gamma,  # 0.1,  # 0.0, 1.0
                                  subsample=args.subsample,  # 0.9,  # 0.6, 1.0
                                  colsample_bytree=args.colsample_bytree,  # 0.9,  # 0.6, 1.0, 0.8
                                  reg_alpha=args.reg_alpha,  # 0.2,  # 0.0, 2.0
                                  reg_lambda=args.reg_lambda,  # 2.0,  # 0.5, 5.0, 1
                                  )

        # Concatenate features: continuous first, then categorical
        if variable_type != 'brain':
            cont_cols = [f"cont_{c}" for c in continue_col]
            cat_cols = [f"cat_{c}" for c in category_col]

            # Create separate DataFrames for continuous and categorical variables
            X_train_cont = pd.DataFrame(train_con.astype("float64"), columns=cont_cols)
            X_valid_cont = pd.DataFrame(valid_con.astype("float64"), columns=cont_cols)
            X_test_cont = pd.DataFrame(test_con.astype("float64"), columns=cont_cols)

            X_train_cat = pd.DataFrame(train_cat.astype("int64"), columns=cat_cols)
            X_valid_cat = pd.DataFrame(valid_cat.astype("int64"), columns=cat_cols)
            X_test_cat = pd.DataFrame(test_cat.astype("int64"), columns=cat_cols)

            # Column-wise concatenation while preserving dtypes:
            #   float64 for continuous variables and int64 for categorical variables
            X_train = pd.concat([X_train_cont, X_train_cat], axis=1)
            X_valid = pd.concat([X_valid_cont, X_valid_cat], axis=1)
            X_test = pd.concat([X_test_cont, X_test_cat], axis=1)

            # Unify categorical dtypes and category sets between train and test
            for col in cat_cols:
                train_int = X_train[col].astype("int64")
                valid_int = X_valid[col].astype("int64")
                test_int = X_test[col].astype("int64")

                # Collect unique values across train and test
                train_unique = np.unique(train_int.to_numpy())
                valid_unique = np.unique(valid_int.to_numpy())
                all_unique = np.unique(np.concatenate([train_unique, valid_unique]))

                # Sort the combined unique values
                final_categories = np.sort(all_unique)
                # Ensure there is at least one category
                if len(final_categories) == 0:
                    final_categories = np.array([0])

                # Convert the column into a pandas Categorical
                # All observed integer codes are included in 'categories'
                X_train[col] = pd.Categorical(train_int, categories=final_categories, ordered=False)
                X_valid[col] = pd.Categorical(valid_int, categories=final_categories, ordered=False)
                X_test[col] = pd.Categorical(test_int, categories=final_categories, ordered=False)

        else:
            # use only continuous value for brain only case
            X_train = pd.DataFrame(train_con.astype("float64"), columns=[f"cont_{c}" for c in continue_col])
            X_valid = pd.DataFrame(valid_con.astype("float64"), columns=[f"cont_{c}" for c in continue_col])
            X_test = pd.DataFrame(test_con.astype("float64"), columns=[f"cont_{c}" for c in continue_col])

        model.fit(X_train, train_y,
                  eval_set=[(X_valid, valid_y)],
                  verbose=False)

        # Predict on validation and test set
        y_pred_proba_val = model.predict_proba(X_valid)[:, 1]
        y_pred_val = model.predict(X_valid)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Evaluation metrics
        acc_val = accuracy_score(valid_y, y_pred_val)
        auc_val = roc_auc_score(valid_y, y_pred_proba_val)
        acc = accuracy_score(test_y, y_pred)
        auc = roc_auc_score(test_y, y_pred_proba)

        tn_val, fp_val, fn_val, tp_val = confusion_matrix(valid_y, y_pred_val, labels=[0, 1]).ravel()
        tn, fp, fn, tp = confusion_matrix(test_y, y_pred, labels=[0, 1]).ravel()
        sensitivity_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0  # recall, TPR
        specificity_val = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0.0  # TNR
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall, TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR

        print(
            f"[iter {it} fold {fold}] Validation acc={acc_val:.4f}, auc={auc_val:.4f}, "
            f"sen={sensitivity_val:.4f}, spc={specificity_val:.4f}\n"
            f"Test acc={acc:.4f}, auc={auc:.4f}, "
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

        # ===== interpretation for each subject (XGBoost has their own TreeSHAP for interpretation) =====
        booster = model.get_booster()
        X_test_for_shap = X_test.copy()
        if variable_type != "brain":
            for col in cat_cols:
                if not isinstance(X_test_for_shap[col].dtype, pd.CategoricalDtype):
                    X_test_for_shap[col] = X_test_for_shap[col].astype("int64").astype("category")

        dtest = xgb.DMatrix(X_test_for_shap, enable_categorical=True)
        shap_values = booster.predict(dtest, pred_contribs=True)

        pred_contrib = booster.predict(dtest, pred_contribs=True)
        shap_values = pred_contrib[:, :-1]
        base_values = pred_contrib[:, -1]

        if variable_type != 'brain':
            feature_names = (
                    [f"cont_{c}" for c in continue_col]
                    + [f"cat_{c}" for c in category_col]
            )
        else:
            feature_names = [f"cont_{c}" for c in continue_col]

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

        pd.DataFrame(rows).to_csv(f"{outdir}/shap_iter{it}_fold{fold}.csv", index=False)

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

print(f"Average acc={np.mean(all_iter_acc):.4f}, auc={np.mean(all_iter_auc):.4f},"
      f"sen={np.mean(all_iter_sen):.4f}, spc={np.mean(all_iter_spc):.4f}")
print(f"STD acc={np.std(all_iter_acc):.4f}, auc={np.std(all_iter_auc):.4f},"
      f"sen={np.std(all_iter_sen):.4f}, spc={np.std(all_iter_spc):.4f}")

df_all = pd.DataFrame(all_rows)
df_all.to_csv(csv_save_all_iter_fold, index=False)

tot_avg_cols = ['variable', 'clstype',
                'val mean auc', 'val mean acc', 'val mean sen', 'val mean spc', 'val std auc', 'val std acc', 'val std sen', 'val std spc',
                'mean auc', 'mean acc', 'mean sen', 'mean spc', 'std auc', 'std acc', 'std sen', 'std spc',
                'tree_method', 'max_depth', 'learning_rate', 'n_estimators', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree',
                'reg_alpha', 'reg_lambda']
tot_avg_vals = [variable_type, args.cls_type,
                np.mean(val_all_iter_auc), np.mean(val_all_iter_acc), np.mean(val_all_iter_sen), np.mean(val_all_iter_spc),
                np.std(val_all_iter_auc, ddof=1), np.std(val_all_iter_acc, ddof=1), np.std(val_all_iter_sen, ddof=1),
                np.std(val_all_iter_spc, ddof=1),
                np.mean(all_iter_auc), np.mean(all_iter_acc), np.mean(all_iter_sen), np.mean(all_iter_spc),
                np.std(all_iter_auc, ddof=1), np.std(all_iter_acc, ddof=1), np.std(all_iter_sen, ddof=1),
                np.std(all_iter_spc, ddof=1),
                tree_method, args.max_depth, args.learning_rate, args.n_estimators, args.min_child_weight, args.gamma, args.subsample,
                args.colsample_bytree, args.reg_alpha, args.reg_lambda]

df_row = pd.DataFrame([tot_avg_vals], columns=tot_avg_cols)

file_exists = os.path.isfile(csv_save_total_averaged_results)

df_row.to_csv(
    csv_save_total_averaged_results,
    mode='a' if file_exists else 'w',   # append or write
    header=not file_exists,              # write header only if new file
    index=False)
