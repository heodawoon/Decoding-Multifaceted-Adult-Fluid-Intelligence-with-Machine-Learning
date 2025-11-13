"""
XGBoost classification with TreeSHAP interpretation on UKB fluid intelligence (top/bottom 10%).

This script:
- Loads the UKB dataset and cross-validation splits (5 repeats × 5 folds).
- Trains an XGBoost classifier for each (iteration, fold) using different variable subsets.
- Evaluates classification performance (ACC, AUC, sensitivity, specificity).
- Computes TreeSHAP values for each test subject and feature.

Inputs:
- --variable_type: which feature subset to use (all / brain / health / socio / brain_health / brain_socio / health_socio)
- --json_path: path to the cross-validation split JSON file.
- --data_path: path to the input CSV file.
- --outdir: directory to save all results.
- --gpu: CUDA device index (set -1 for CPU).

Outputs:
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
from dataloader import select_data
from sklearn.metrics import confusion_matrix


# ------------------- Config (argparse) -------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost + SHAP on UKB fluid intelligence (top/bottom 10%)"
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
        default="/LocalData1/daheo/BRL/UKB_new/Model1_Tree_based/XGBoost_shap",
        help="Directory to save all results and SHAP outputs.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="CUDA device index (used for CUDA_VISIBLE_DEVICES). Use -1 for CPU.",
    )
    return parser.parse_args()


args = parse_args()
if args.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

variable_type = args.variable_type
json_path = args.json_path
data_path = args.data_path
outdir = os.path.join(args.outdir, variable_type)

os.makedirs(outdir, exist_ok=True)

save_result_csv = f"{outdir}/XGBoost_shap_{variable_type}_final_result_value.csv"
csv_save_all_iter_fold = f"{outdir}/XGBoost_shap_{variable_type}_all_iters_folds.csv"

df = pd.read_csv(data_path)
df = df.fillna(0.0)

category_col, continue_col, Categories = select_data(variable_type)

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

all_iter_acc = []
all_iter_auc = []
all_iter_sen = []
all_iter_spc = []

for it in range(5):
    all_fold_acc = []
    all_fold_auc = []
    all_fold_sen = []
    all_fold_spc = []

    for fold in range(5):
        itfold = ids["iterations"][it]["folds"][fold]
        train_eids = itfold["train_eid"]
        test_eids = itfold["valid_eid"]

        train_df = df.loc[df[eid_col].isin(train_eids)].copy().sort_values(eid_col)
        train_cont = train_df[continue_col].to_numpy(dtype=np.float32)

        # ---- Compute mean and std from the training set for z-normalization ----
        mean_cont = train_cont.mean(axis=0, keepdims=True)
        std_cont = train_cont.std(axis=0, ddof=0, keepdims=True)
        std_cont[std_cont == 0.0] = 1.0  # zero-variance 보호

        train_cat, train_con, train_y, _ = get_datalist(variable_type,
                                                        category_col, continue_col, df, train_eids,
                                                        mean_cont.astype(np.float32), std_cont.astype(np.float32)
                                                        )
        test_cat, test_con, test_y, test_eids_sorted = get_datalist(variable_type,
                                                                    category_col, continue_col, df, test_eids,
                                                                    mean_cont.astype(np.float32),
                                                                    std_cont.astype(np.float32)
                                                                    )

        # Setting: GPU flag (currently both use 'hist'; switch to 'gpu_hist' if GPU training is desired)
        if args.gpu >= 0:
            device = 'gpu'
            tree_method = 'hist'
        else:
            device = 'cpu'
            tree_method = 'hist'

        if variable_type != 'brain':
            model = XGBClassifier(objective='binary:logistic',
                                  eval_metric='logloss',
                                  use_label_encoder=False,
                                  random_state=42,
                                  tree_method=tree_method, device=device,
                                  enable_categorical=True,
                                  max_depth=3,  # 3, 6, 10
                                  learning_rate=0.05,  # 0.01, 0.3
                                  n_estimators=100,  # 100, 200, 800
                                  min_child_weight=3,  # 1, 10
                                  gamma=0.1,  # 0.0, 1.0
                                  subsample=0.9,  # 0.6, 1.0
                                  colsample_bytree=0.9,  # 0.6, 1.0, 0.8
                                  reg_alpha=0.2,  # 0.0, 2.0
                                  reg_lambda=2.0,  # 0.5, 5.0, 1
                                  )
        else:
            # The model for only use brain case:
            model = XGBClassifier(objective='binary:logistic',
                                  eval_metric='logloss',
                                  use_label_encoder=False,
                                  random_state=42,
                                  tree_method=tree_method, device=device,
                                  max_depth=3,  # 3, 6, 10
                                  learning_rate=0.05,  # 0.01, 0.3
                                  n_estimators=100,  # 100, 200, 800
                                  min_child_weight=3,  # 1, 10
                                  gamma=0.1,  # 0.0, 1.0
                                  subsample=0.9,  # 0.6, 1.0
                                  colsample_bytree=0.9,  # 0.6, 1.0, 0.8
                                  reg_alpha=0.2,  # 0.0, 2.0
                                  reg_lambda=2.0,  # 0.5, 5.0, 1
                                  )

        # Concatenate features: continuous first, then categorical
        if variable_type != 'brain':
            cont_cols = [f"cont_{c}" for c in continue_col]
            cat_cols = [f"cat_{c}" for c in category_col]

            # 1) Create separate DataFrames for continuous and categorical variables
            X_train_cont = pd.DataFrame(train_con.astype("float64"), columns=cont_cols)
            X_test_cont = pd.DataFrame(test_con.astype("float64"), columns=cont_cols)

            X_train_cat = pd.DataFrame(train_cat.astype("int64"), columns=cat_cols)
            X_test_cat = pd.DataFrame(test_cat.astype("int64"), columns=cat_cols)

            # 2) Column-wise concatenation while preserving dtypes:
            #    float64 for continuous variables and int64 for categorical variables
            X_train = pd.concat([X_train_cont, X_train_cat], axis=1)
            X_test = pd.concat([X_test_cont, X_test_cat], axis=1)

            # 3) Unify categorical dtypes and category sets between train and test
            for col in cat_cols:
                train_int = X_train[col].astype("int64")
                test_int = X_test[col].astype("int64")

                # Collect unique values across train and test
                train_unique = np.unique(train_int.to_numpy())
                test_unique = np.unique(test_int.to_numpy())
                all_unique = np.unique(np.concatenate([train_unique, test_unique]))

                # Sort the combined unique values
                final_categories = np.sort(all_unique)
                # Ensure there is at least one category
                if len(final_categories) == 0:
                    final_categories = np.array([0])

                # Convert the column into a pandas Categorical
                # All observed integer codes are included in 'categories' (NaNs already handled by fillna)
                X_train[col] = pd.Categorical(train_int, categories=final_categories, ordered=False)
                X_test[col] = pd.Categorical(test_int, categories=final_categories, ordered=False)

        else:
            # use only continuous value for brain type
            X_train = pd.DataFrame(train_con.astype("float64"), columns=[f"cont_{c}" for c in continue_col])
            X_test = pd.DataFrame(test_con.astype("float64"), columns=[f"cont_{c}" for c in continue_col])

        model.fit(X_train, train_y)

        # Make predictions on the test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
        y_pred = model.predict(X_test)  # Get hard class predictions (0 or 1)

        # Evaluate the model
        acc = accuracy_score(test_y, y_pred)
        auc = roc_auc_score(test_y, y_pred_proba)

        tn, fp, fn, tp = confusion_matrix(test_y, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall, TPR
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

        # If file does not exist → write header; otherwise append without header
        write_header = not os.path.exists(save_result_csv)
        with open(save_result_csv, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(result_row)

        all_fold_acc.append(acc)
        all_fold_auc.append(auc)
        all_fold_sen.append(sensitivity)
        all_fold_spc.append(specificity)

        # ===== interpretation for each subject (XGBoost has their own TreeSHAP for interpretation) =====
        booster = model.get_booster()

        # Convert X_test to purely numeric dtypes for SHAP computation.
        # XGBoost internally stores categorical features as integer codes,
        # so we use the same integer codes here when computing TreeSHAP values.
        X_test_for_shap = X_test.copy()
        if variable_type != 'brain':
            for col in cat_cols:
                # Use integer codes (.cat.codes) for categorical variables
                # so that SHAP uses the same representation as the trained model.
                if isinstance(X_test_for_shap[col].dtype, pd.CategoricalDtype):
                    X_test_for_shap[col] = X_test_for_shap[col].cat.codes.astype("int64")
                else:
                    X_test_for_shap[col] = X_test_for_shap[col].astype("int64")

        dtest = xgb.DMatrix(X_test_for_shap)

        # pred_contribs=True → returns TreeSHAP values
        # Shape: (n_samples, n_features + 1) where the last column is the bias term.
        shap_values = booster.predict(dtest, pred_contribs=True)

        # Drop the bias term; we only keep feature-wise SHAP values.
        shap_values = shap_values[:, :-1]

        # ===== SHAP =====
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
            for j, fname in enumerate(feature_names):
                row[f"shap::{fname}"] = float(shap_values[i, j])
            rows.append(row)

        # to save the results for iter-fold-wise
        df_fold = pd.DataFrame(rows)
        df_fold.to_csv(f"{outdir}/shap_iter{it}_fold{fold}.csv", index=False)
        # concat all data into one variable
        all_rows.extend(rows)

    all_fold_acc = np.mean(np.array(all_fold_acc))
    all_fold_auc = np.mean(np.array(all_fold_auc))
    all_fold_sen = np.mean(np.array(all_fold_sen))
    all_fold_spc = np.mean(np.array(all_fold_spc))

    all_iter_acc.append(all_fold_acc)
    all_iter_auc.append(all_fold_auc)
    all_iter_sen.append(all_fold_sen)
    all_iter_spc.append(all_fold_spc)

all_iter_acc = np.array(all_iter_acc)
all_iter_auc = np.array(all_iter_auc)
all_iter_sen = np.array(all_iter_sen)
all_iter_spc = np.array(all_iter_spc)

print(f"Average acc={np.mean(all_iter_acc):.4f}, auc={np.mean(all_iter_auc):.4f},"
      f"sen={np.mean(all_iter_sen):.4f}, spc={np.mean(all_iter_spc):.4f}")
print(f"STD acc={np.std(all_iter_acc):.4f}, auc={np.std(all_iter_auc):.4f},"
      f"sen={np.std(all_iter_sen):.4f}, spc={np.std(all_iter_spc):.4f}")

# Save subject-level SHAP and prediction results across all iterations/folds
df_all = pd.DataFrame(all_rows)
df_all.to_csv(csv_save_all_iter_fold, index=False)
