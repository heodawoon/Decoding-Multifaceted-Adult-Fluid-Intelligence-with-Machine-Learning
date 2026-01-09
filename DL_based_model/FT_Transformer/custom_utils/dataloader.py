import json
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from omegaconf import DictConfig, open_dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def select_data(variable_type):
    # Disase - categorical
    disease_cols_all = ['A02', 'A04', 'A07', 'A08', 'A09', 'A15', 'A37', 'A38', 'A41', 'A49', 'A60', 'A63', 'A80',
                        'B00', 'B01', 'B02', 'B05', 'B06', 'B07', 'B08', 'B15', 'B16', 'B19', 'B26', 'B27', 'B34',
                        'B35', 'B36', 'B37', 'B54', 'B80', 'B86', 'B95', 'B96', 'B97', 'B98', 'B99', 'D50', 'D51',
                        'D61', 'D64', 'D68', 'D69', 'D70', 'D72', 'D75', 'D86', 'E03', 'E04', 'E05', 'E06', 'E07',
                        'E10', 'E11', 'E14', 'E16', 'E21', 'E22', 'E23', 'E27', 'E28', 'E53', 'E55', 'E66', 'E78',
                        'E80', 'E83', 'E86', 'E87', 'E89', 'F10', 'F17', 'F31', 'F32', 'F33', 'F34', 'F39', 'F40',
                        'F41', 'F43', 'F45', 'F48', 'F52', 'F53', 'H00', 'H01', 'H02', 'H04', 'H05', 'H10', 'H11',
                        'H15', 'H16', 'H18', 'H20', 'H21', 'H25', 'H26', 'H31', 'H33', 'H34', 'H35', 'H36', 'H40',
                        'H43', 'H44', 'H47', 'H50', 'H52', 'H53', 'H54', 'H57', 'H60', 'H61', 'H65', 'H66', 'H68',
                        'H69', 'H72', 'H80', 'H81', 'H83', 'H90', 'H91', 'H92', 'H93', 'I00', 'I08', 'I10', 'I20',
                        'I21', 'I24', 'I25', 'I26', 'I30', 'I31', 'I34', 'I35', 'I42', 'I44', 'I45', 'I47', 'I48',
                        'I49', 'I50', 'I51', 'I71', 'I73', 'I74', 'I77', 'I78', 'I80', 'I82', 'I83', 'I84', 'I86',
                        'I87', 'I89', 'I95', 'J00', 'J01', 'J02', 'J03', 'J04', 'J06', 'J11', 'J13', 'J18', 'J20',
                        'J22', 'J30', 'J31', 'J32', 'J33', 'J34', 'J35', 'J36', 'J38', 'J39', 'J40', 'J43', 'J44',
                        'J45', 'J47', 'J81', 'J84', 'J90', 'J92', 'J93', 'J98', 'K01', 'K02', 'K04', 'K05', 'K07',
                        'K08', 'K09', 'K10', 'K11', 'K12', 'K13', 'K14', 'K20', 'K21', 'K22', 'K25', 'K26', 'K27',
                        'K29', 'K30', 'K31', 'K35', 'K37', 'K40', 'K41', 'K42', 'K43', 'K44', 'K46', 'K50', 'K51',
                        'K52', 'K55', 'K56', 'K57', 'K58', 'K59', 'K60', 'K61', 'K62', 'K63', 'K64', 'K65', 'K66',
                        'K70', 'K75', 'K76', 'K80', 'K81', 'K82', 'K83', 'K85', 'K86', 'K90', 'K92', 'L02', 'L03',
                        'L05', 'L08', 'L13', 'L20', 'L21', 'L23', 'L25', 'L28', 'L29', 'L30', 'L40', 'L42', 'L43',
                        'L50', 'L53', 'L57', 'L60', 'L63', 'L65', 'L70', 'L71', 'L72', 'L73', 'L74', 'L80', 'L81',
                        'L82', 'L84', 'L85', 'L90', 'L91', 'L92', 'L94', 'L98', 'M06', 'M10', 'M11', 'M13', 'M15',
                        'M16', 'M17', 'M18', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25', 'M31', 'M32', 'M35',
                        'M41', 'M43', 'M45', 'M46', 'M47', 'M48', 'M50', 'M51', 'M53', 'M54', 'M62', 'M65', 'M66',
                        'M67', 'M70', 'M71', 'M72', 'M75', 'M76', 'M77', 'M79', 'M80', 'M81', 'M84', 'M85', 'M86',
                        'M89', 'M93', 'M94', 'M95', 'N02', 'N05', 'N10', 'N12', 'N13', 'N17', 'N18', 'N19', 'N20',
                        'N23', 'N28', 'N30', 'N31', 'N32', 'N34', 'N35', 'N36', 'N39', 'N40', 'N41', 'N42', 'N43',
                        'N45', 'N46', 'N47', 'N48', 'N50', 'N60', 'N61', 'N62', 'N63', 'N64', 'N70', 'N72', 'N73',
                        'N75', 'N76', 'N80', 'N81', 'N83', 'N84', 'N85', 'N86', 'N87', 'N88', 'N89', 'N90', 'N91',
                        'N92', 'N93', 'N94', 'N95', 'N97', 'O00', 'O02', 'O03', 'O04', 'O06', 'O13', 'O14', 'O16',
                        'O20', 'O21', 'O24', 'O26', 'O30', 'O32', 'O34', 'O36', 'O42', 'O46', 'O47', 'O48', 'O60',
                        'O62', 'O63', 'O64', 'O66', 'O68', 'O69', 'O70', 'O72', 'O73', 'O75', 'O80', 'O81', 'O82',
                        'O86', 'O99', 'Q18', 'Q43', 'Q53', 'Q61', 'Q63', 'Q66', 'Q76', 'Q82']

    # Brain imaging
    brain_index = np.arange(25056, 25103 + 1)  # FA
    brain_cols = [f"{idx}-2.0" for idx in brain_index]

    if variable_type == 'all':
        # ------ Categorical variables
        category_col = ["gender", "ethnicity_0", "marital_2",                                      # Demographic
                        "emp_2", "income_fam_2",                                                   # Socioeconomic
                        "lone_2", "social_act_2_sport", "social_act_2_pub",                        # Network
                        "social_act_2_religious", "social_act_2_education", "social_act_2_other",  # Network
                        "smoke_status_2", "glass_lenses_2", "eye_issue_2",                         # Health
                        "hearing_issue_2", "hearing_issue_bg_2", "hearing_aid_2"]                  # Health

        # ------ Disase - categorical
        category_col = category_col + disease_cols_all

        # ------ Categories info - each of a number of categories
        category_without_disease = [2, 5, 2, 7, 5, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2]
        category_with_disease = [2] * len(disease_cols_all)
        Categories = category_without_disease + category_with_disease

        # ------ Continuous variables
        continue_col = ["age_2",                                                  # Demographic
                        "ed_yr_2", "fncl_sat_2", "hthcare_2",                     # Socioeconomic
                        "social_act_n_2", "freq_visit_2", "confide_2",            # Network
                        "fam_sat_2", "frnd_sat_2", "N_fam_2",                     # Network
                        "alcohol_2", "phq2_2", "hlth_sat_2", "sleep_2", "bmi_2",  # Health
                        "met_2"]                                                  # Physical

        continue_col = continue_col + brain_cols

        print('!!!!!!!!!!!!!!!!!!!! total variables: ', len(continue_col))
        return category_col, continue_col, Categories

    elif variable_type == 'brain':
        # continue_col = brain_cols
        return None, brain_cols, None

    elif variable_type == 'health':
        # ------ Categorical variables
        category_col = ["smoke_status_2", "glass_lenses_2", "eye_issue_2",         # Health
                        "hearing_issue_2", "hearing_issue_bg_2", "hearing_aid_2"]  # Health

        category_col = category_col + disease_cols_all

        # ------ Categories info - each of a number of categories
        category_without_disease = [3, 2, 2, 2, 2, 2]
        category_with_disease = [2] * len(disease_cols_all)
        Categories = category_without_disease + category_with_disease

        # ------ Continuous variables
        continue_col = ["alcohol_2", "phq2_2", "hlth_sat_2", "sleep_2", "bmi_2",  # Health
                        "met_2"]                                                  # Physical

        return category_col, continue_col, Categories

    elif variable_type == 'socio':
        # ------ Categorical variables
        category_col = ["gender", "ethnicity_0", "marital_2",                                        # Demographic
                        "emp_2", "income_fam_2",                                                     # Socioeconomic
                        "lone_2", "social_act_2_sport", "social_act_2_pub",                          # Network
                        "social_act_2_religious", "social_act_2_education", "social_act_2_other", ]  # Network

        # ------ Disase - categorical
        category_col = category_col + disease_cols_all

        # ------ Categories info - each of a number of categories
        category_without_disease = [2, 5, 2, 7, 5, 2, 2, 2, 2, 2, 2]
        category_with_disease = [2] * len(disease_cols_all)
        Categories = category_without_disease + category_with_disease

        # ------ Continuous variables
        continue_col = ["age_2",                                        # Demographic
                        "ed_yr_2", "fncl_sat_2", "hthcare_2",           # Socioeconomic
                        "social_act_n_2", "freq_visit_2", "confide_2",  # Network
                        "fam_sat_2", "frnd_sat_2", "N_fam_2"]           # Network

        return category_col, continue_col, Categories


    elif variable_type == 'brain_health':

        health_cont_cols = ["alcohol_2", "phq2_2", "hlth_sat_2", "sleep_2", "bmi_2",  # Health
                            "met_2"]                                                  # Physical
        continue_col = health_cont_cols + brain_cols

        # ------ Categorical variables
        category_col = ["smoke_status_2", "glass_lenses_2", "eye_issue_2",         # Health
                        "hearing_issue_2", "hearing_issue_bg_2", "hearing_aid_2"]  # Health

        # ------ Disase - categorical
        category_col = category_col + disease_cols_all

        # ------ Categories info - each of a number of categories
        category_without_disease = [3, 2, 2, 2, 2, 2]
        category_with_disease = [2] * len(disease_cols_all)
        Categories = category_without_disease + category_with_disease

        return category_col, continue_col, Categories


    elif variable_type == 'brain_socio':
        # ------ Continuous variables
        socio_cont_cols = ["age_2",                                        # Demographic
                           "ed_yr_2", "fncl_sat_2", "hthcare_2",           # Socioeconomic
                           "social_act_n_2", "freq_visit_2", "confide_2",  # Network
                           "fam_sat_2", "frnd_sat_2", "N_fam_2"]           # Network

        continue_col = socio_cont_cols + brain_cols

        # ------ Categorical variables
        category_col = ["gender", "ethnicity_0", "marital_2",                                        # Demographic
                        "emp_2", "income_fam_2",                                                     # Socioeconomic
                        "lone_2", "social_act_2_sport", "social_act_2_pub",                          # Network
                        "social_act_2_religious", "social_act_2_education", "social_act_2_other", ]  # Network

        # ------ Disase - categorical
        category_col = category_col + disease_cols_all

        # ------ Categories info - each of a number of categories
        category_without_disease = [2, 5, 2, 7, 5, 2, 2, 2, 2, 2, 2]
        category_with_disease = [2] * len(disease_cols_all)
        Categories = category_without_disease + category_with_disease

        return category_col, continue_col, Categories


    elif variable_type == 'health_socio':
        # ------ Continuous variables
        socio_cont_cols = ["age_2",                                        # Demographic
                           "ed_yr_2", "fncl_sat_2", "hthcare_2",           # Socioeconomic
                           "social_act_n_2", "freq_visit_2", "confide_2",  # Network
                           "fam_sat_2", "frnd_sat_2", "N_fam_2"]           # Network

        # ------ Continuous variables
        health_cont_cols = ["alcohol_2", "phq2_2", "hlth_sat_2", "sleep_2", "bmi_2",  # Health
                            "met_2"]                                                  # Physical

        continue_col = socio_cont_cols + health_cont_cols

        # ------ Categorical variables
        socio_cat_cols = ["gender", "ethnicity_0", "marital_2",                                        # Demographic
                          "emp_2", "income_fam_2",                                                     # Socioeconomic
                          "lone_2", "social_act_2_sport", "social_act_2_pub",                          # Network
                          "social_act_2_religious", "social_act_2_education", "social_act_2_other", ]  # Network

        health_cat_cols = ["smoke_status_2", "glass_lenses_2", "eye_issue_2",         # Health
                           "hearing_issue_2", "hearing_issue_bg_2", "hearing_aid_2"]  # Health

        # ------ Disase - categorical
        category_col = socio_cat_cols + health_cat_cols + disease_cols_all

        # ------ Categories info - each of a number of categories
        category_without_disease = [2, 5, 2, 7, 5, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2]
        category_with_disease = [2] * len(disease_cols_all)
        Categories = category_without_disease + category_with_disease

        return category_col, continue_col, Categories


class UKBDataset(Dataset):
    def __init__(self, *, variable_type: str, category_col: Optional[List[str]], continue_col: List[str],
                 data: pd.DataFrame, eids: List[int],
                 mean_cont: np.ndarray, std_cont: np.ndarray):
        self.variable_type = variable_type
        eid_col = "eid"
        label_col = "fluid_2_p10"

        # Select only rows corresponding to the given eids
        sub = data.loc[data[eid_col].isin(eids)].copy()

        # Fix ordering for reproducibility
        sub = sub.sort_values(eid_col).reset_index(drop=True)

        # Check that all required columns exist (otherwise raise an error)
        if variable_type == 'brain':
            missing_con = [c for c in continue_col if c not in sub.columns]
            if missing_con or (label_col not in sub.columns):
                raise KeyError(f"Missing columns: cont={missing_con}, label={label_col not in sub.columns}")
        else:
            if category_col is None:
                raise ValueError("category_col must not be None when variable_type != 'brain'")
            missing_cat = [c for c in category_col if c not in sub.columns]
            missing_con = [c for c in continue_col if c not in sub.columns]
            if missing_cat or missing_con or (label_col not in sub.columns):
                raise KeyError(f"Missing columns: cat={missing_cat}, cont={missing_con}, label={label_col not in sub.columns}")

        self.eids = sub[eid_col].to_numpy()
        self.y = sub[label_col].astype(np.int64).to_numpy()  # binary label

        if variable_type != 'brain':
            X_cat = sub[category_col].to_numpy(dtype=np.int64)
            self.X_cat = torch.from_numpy(X_cat).long()

        X_con = sub[continue_col].to_numpy(dtype=np.float32)

        # Standardization (z-score normalization): (x - mean) / std
        self.mean_cont = mean_cont
        self.std_cont = std_cont
        X_con = (X_con - self.mean_cont) / self.std_cont

        # Store as torch tensors
        self.X_con = torch.from_numpy(X_con).float()
        self.y = F.one_hot(torch.from_numpy(self.y), num_classes=2).long()

    def __len__(self):
        return len(self.eids)

    def __getitem__(self, idx: int):
        if self.variable_type == 'brain':
            return {
                "eid": int(self.eids[idx]),
                "x_cat": None,
                "x_con": self.X_con[idx],  # FloatTensor
                "y": self.y[idx]           # LongTensor one-hot label
            }
        else:
            return {
                "eid": int(self.eids[idx]),
                "x_cat": self.X_cat[idx],  # LongTensor
                "x_con": self.X_con[idx],  # FloatTensor
                "y": self.y[idx]           # LongTensor one-hot label
            }

def collate_ukb(batch):
    eids = [b["eid"] for b in batch]

    x_con = torch.stack([b["x_con"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)

    if batch[0].get("x_cat") is None:
        x_cat = None
    else:
        x_cat = torch.stack([b["x_cat"] for b in batch], dim=0)

    return {
        "eid": eids,
        "x_cat": x_cat,
        "x_con": x_con,
        "y": y,
    }

def build_dataloaders(cfg: DictConfig, fold: int) -> Dict[str, Any]:
    """
    Read train_eid and test_eid from the JSON file for the specified iteration and fold.
    A stratified 10% split of train_eid is used to construct the validation set,
    and train/validation/test DataLoaders are returned with normalization based on training-set statistics.
    """
    eid_col = "eid"

    with open(cfg.dataset.json_path, "r") as f:
        ids = json.load(f)
    it = cfg.dataset.iteration
    itfold = ids["iterations"][it]["folds"][fold]
    train_val_eids = itfold["train_eid"]
    test_eids = itfold["valid_eid"]

    # Load dataset
    df = pd.read_csv(cfg.dataset.data_path)
    category_col, continue_col, Categories = select_data(cfg.dataset.variable_type)
    assert not df[continue_col].isna().any().any()
    if category_col is not None:
        assert not df[category_col].isna().any().any()

    # ---- train_val dataframe ----
    label_col = "fluid_2_p10"
    train_valid_df = df.loc[df[eid_col].isin(train_val_eids)].copy().sort_values(eid_col)
    train_valid_eid = train_valid_df[eid_col].to_numpy()
    train_valid_labels = train_valid_df[label_col].to_numpy()

    # ---- Split validation (Stratified 10%) ----
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, valid_idx = next(sss.split(train_valid_eid, train_valid_labels))

    train_eids = train_valid_eid[train_idx].tolist()
    valid_eids = train_valid_eid[valid_idx].tolist()

    # ---- Compute train statistics (train ONLY) ----
    train_df = df.loc[df[eid_col].isin(train_eids)].copy().sort_values(eid_col)

    train_cont = train_df[continue_col].to_numpy(dtype=np.float32)

    mean_cont = train_cont.mean(axis=0, keepdims=True)
    std_cont = train_cont.std(axis=0, ddof=0, keepdims=True)
    std_cont[std_cont == 0.0] = 1.0  # zero-variance 보호

    # ---- Dataset & DataLoader ----
    train_set = UKBDataset(variable_type=cfg.dataset.variable_type,
                           category_col=category_col, continue_col=continue_col,
                           data=df, eids=train_eids,
                           mean_cont=mean_cont.astype(np.float32),
                           std_cont=std_cont.astype(np.float32))
    valid_set = UKBDataset(variable_type=cfg.dataset.variable_type,
                           category_col=category_col, continue_col=continue_col,
                           data=df, eids=valid_eids,
                           mean_cont=mean_cont.astype(np.float32),
                           std_cont=std_cont.astype(np.float32))
    test_set = UKBDataset(variable_type=cfg.dataset.variable_type,
                          category_col=category_col, continue_col=continue_col,
                          data=df, eids=test_eids,
                          mean_cont=mean_cont.astype(np.float32),
                          std_cont=std_cont.astype(np.float32))

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_ukb,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_ukb,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_ukb,
        pin_memory=True
    )

    # Update steps_per_epoch, total_steps, warmup, etc.
    with open_dict(cfg):
        cfg.model.d_numerical = len(continue_col)
        cfg.steps_per_epoch = len(train_loader)
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.train_epochs

        # Warmup
        min_warmup_steps = cfg.optimizer.lr_scheduler.min_warmup_steps
        max_warmup_steps = cfg.optimizer.lr_scheduler.max_warmup_steps
        warmup_steps = int(cfg.steps_per_epoch * cfg.optimizer.lr_scheduler.warmup_percentage)

        # Clamp to reasonable bounds
        warmup_steps = max(min_warmup_steps, min(warmup_steps, max_warmup_steps))
        cfg.optimizer.lr_scheduler.warm_up_steps = warmup_steps


    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
        "categories": Categories,
    }

