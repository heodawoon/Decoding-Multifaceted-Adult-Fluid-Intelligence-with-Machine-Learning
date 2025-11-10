"""
Brain-related disease filtering.

Reference:
Klinedinst, B. S., et al. “Aging-related changes in fluid intelligence, muscle and adipose mass,
and sex-specific immunologic mediation: A longitudinal UK Biobank study.”
Brain, Behavior, and Immunity 82 (2019): 396–405.

Disease sets to remove (ICD-10):
- Cerebrovascular diseases: I60–I69
- Diseases of the nervous system: G00–G99

Inputs:
- Step2_1_ukb669045_disease_timing_redefined.csv
  (disease timing coded as: 0.0, 1.0, 1.5, 2.0)

Outputs:
- Step3_ukb669045_remove_brain_related_disease_subjects.csv
  (subjects with any brain-related disease before or at imaging removed)
- Step3_ukb669045_removed_brain_related_subjects.csv
  (list of subjects removed with their brain-related disease timings)
"""

import os
import numpy as np
import pandas as pd

# ---------- Config ----------
root_path = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB/FinalFinal_Version_1108'

csv_path = os.path.join(root_path, 'Step2', 'Step2_1_ukb669045_disease_timing_redefined.csv')
save_root = os.path.join(root_path, 'Step3')
os.makedirs(save_root, exist_ok=True)

csv_path_out = os.path.join(save_root, 'Step3_ukb669045_remove_brain_related_disease_subjects.csv')
removed_path = os.path.join(save_root, 'Step3_ukb669045_removed_brain_related_subjects.csv')

# Brain-related disease list
# Diseases of the nervous system (ICD-10: G00–G99)
glist = ['G00', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'G14', 'G20', 'G21', 'G22', 'G23', 'G24', 'G25', 'G30', 'G31', 'G32', 'G35', 'G36', 'G37', 'G40', 'G41', 'G43', 'G44', 'G45', 'G46', 'G47', 'G50', 'G51', 'G52', 'G53', 'G54', 'G55', 'G56', 'G57', 'G58', 'G59', 'G60', 'G61', 'G62', 'G63', 'G64', 'G70', 'G71', 'G72', 'G73', 'G80', 'G81', 'G82', 'G83', 'G90', 'G91', 'G92', 'G93', 'G94', 'G95', 'G96', 'G97', 'G98', 'G99']
# Cerebrovascular diseases (ICD-10: I60–I69)
ilist = ['I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69']

# Full list for column selection
brain_icd_list = glist + ilist

# ---------- Load ----------
df = pd.read_csv(csv_path)

# Keep only those brain-related ICD codes that actually exist in the DataFrame
brain_cols = [c for c in brain_icd_list if c in df.columns]

if len(brain_cols) == 0:
    print("[WARN] No brain-related disease columns found in the input file.")
    # In this case, simply save the original file and exit
    df.to_csv(csv_path_out, index=False)
else:
    # Pull the timing matrix (N × D) for brain-related diseases only
    # Values are expected to be:
    #   0.0 : no disease
    #   1.0 : before imaging
    #   1.5 : on imaging date
    #   2.0 : after imaging
    brain_timing = df[brain_cols].astype(float).to_numpy(copy=True)

    # Quick logs: which brain codes are present before or at imaging (1.0 or 1.5)
    def present_cols_before_or_at(arr: np.ndarray) -> list:
        if arr.shape[1] == 0:
            return []
        has_before_or_at = ((arr > 0) & (arr <= 1.5)).any(axis=0)
        return [c for c, keep in zip(brain_cols, has_before_or_at) if keep]

    print("Brain-related disease codes (before or at imaging):")
    print(present_cols_before_or_at(brain_timing))

    # ---------- No-disease subject mask ----------
    # Subjects with any brain-related disease before or at imaging:
    # timing > 0 and timing <= 1.5
    if brain_timing.size:
        has_any_brain_before_or_at = ((brain_timing > 0) & (brain_timing <= 1.5)).any(axis=1)
        mask_safe = ~has_any_brain_before_or_at
    else:
        # If for some reason there are no brain-related columns, keep everyone
        mask_safe = np.ones(len(df), dtype=bool)

    print(f"No brain disease before/at imaging = {mask_safe.sum()}  "
          f"Any brain disease before/at imaging = {(~mask_safe).sum()}")

    # ---------- Drop brain-related columns and save ----------
    drop_set = set(brain_cols)
    keep_cols = ["eid"] + [c for c in df.columns if c not in drop_set and c != "eid"]

    safe_total = df.loc[mask_safe, keep_cols].copy()
    print("safe_total len:", len(safe_total))

    safe_total.to_csv(csv_path_out, index=False)

    # ---------- Sanity check ----------
    vals = df[brain_cols].astype(float).to_numpy(copy=False)

    # Presence matrix (1 if any brain disease before or at imaging, 0 otherwise)
    has_brain_before_or_at = ((vals > 0) & (vals <= 1.5))

    # For "safe" subjects, there should be no brain disease (should print False)
    print(
        "Exist mask_safe with any brain disease before/at imaging:",
        has_brain_before_or_at[mask_safe].any(),
    )

    # Check removed subject according to the brain-related disease
    removed = df.loc[~mask_safe, ["eid"] + brain_cols].copy()
    removed.to_csv(removed_path, index=False)
