"""
Re-filter categorical variables to ensure that all categories start from 0.

Inputs:
- Step4_4_binarize_disease_column.csv

Outputs:
- Step5_refilter_categorical_for_deeplearning.csv

Description:
This step adjusts categorical variable values so that their minimum category index
starts at 0. (e.g., if 'gender' values were [1, 2], they become [0, 1])
It ensures compatibility with deep learning models that expect 0-based category indices.
"""


import os
import pandas as pd
import numpy as np

# ---------- Config ----------
root_path = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB_Final/Final_Version_for_Git'

csv_path = os.path.join(root_path, 'Step4', 'Step4_4_binarize_disease_column.csv')
save_root = os.path.join(root_path, 'Step5')
os.makedirs(save_root, exist_ok=True)
save_path = os.path.join(save_root, 'Step5_refilter_categorical_for_deeplearning.csv')


# ---------- Columns ----------
category_col = ["gender", "ethnicity_0", "marital_2",                     # Demographic
              "emp_2", "income_fam_2",                                    # Socioeconomic
              "lone_2", "social_act_2_sport", "social_act_2_pub",         # Network
              "social_act_2_religious", "social_act_2_education",         # Network
              "social_act_2_other",                                       # Network
              "smoke_status_2",  "glass_lenses_2", "eye_issue_2",         # Health
              "hearing_issue_2", "hearing_issue_bg_2", "hearing_aid_2",]  # Health


# ---------- Load ----------
df = pd.read_csv(csv_path)
print("Loaded:", df.shape)

# ---------- Adjust categorical variables to start from 0 ----------
present_cat_cols = [c for c in category_col if c in df.columns]
if present_cat_cols:
    mins = df[present_cat_cols].min(axis=0)
    shift = mins.where(mins > 0, other=0)
    df[present_cat_cols] = df[present_cat_cols] - shift
    print(f"Shifted {len(present_cat_cols)} categorical columns to start from 0.")

# ---------- Validation ----------
neg_cols = [c for c in category_col if (df[c] < 0).any()]
if neg_cols:
    print("Negtive value exists:", neg_cols)
else:
    print("Every categoricl variables are >= 0")

# ---------- 4) Save ----------
df.to_csv(save_path, index=False)
print(f"Saved to: {save_path}")
print("Final shape:", df.shape)

