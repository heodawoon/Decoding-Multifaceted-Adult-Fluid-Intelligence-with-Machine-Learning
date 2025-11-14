"""
Code for recoding variables and applying reverse coding where needed.

If a variable is defined such that low numeric values correspond to
conceptually higher / better states (e.g., 1 = "very good"), we apply
reverse coding so that the direction of the scale is consistent across
variables.

Input
------
- Step0_2_ukb669045_total_data_with_complete_cortical_thickness.csv

Outputs
-------
1) Main dataset after reverse coding, recoding, and renaming:
   Step1_1_ukb669045_reverse_coding_recoding_renaming.csv

2) Value counts and proportions for each variable:
   Step1_2_ukb669045_variable_recoding_and_renaming_value_counts.xlsx

3) Dataset restricted to rows with no missing values (complete cases only):
   Step1_3_ukb669045_variable_recoding_and_renaming_value_without_nan_rows.csv
"""


import os, re
import pandas as pd
import numpy as np

total_csv_path = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB/FinalFinal_Version_1108/Step0/Step0_2_ukb669045_total_data_with_complete_cortical_thickness.csv'
save_root = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB/FinalFinal_Version_1108/Step1'
os.makedirs(save_root, exist_ok=True)
save_reverse_code = os.path.join(save_root, 'Step1_1_ukb669045_reverse_coding_recoding_renaming.csv')
save_reverse_code_count = os.path.join(save_root, "Step1_2_ukb669045_variable_recoding_and_renaming_value_counts.xlsx")
save_path_without_nan = os.path.join(save_root, "Step1_3_ukb669045_variable_recoding_and_renaming_value_without_nan_rows.csv")

total_df = pd.read_csv(total_csv_path, low_memory=False)

##### Eid
eid = total_df['eid']

##### Gender
print('======================= Gender')
# Extract gender column (field 31-0.0: 0=female, 1=male)
gender_tmp = total_df['31-0.0']
print(f"Unique values in the original gender column: {np.unique(gender_tmp)}")

# Reverse the coding: convert (0 → 1 for female, 1 → 0 for male)
mapping = {0:1, 1:0}
gender = gender_tmp.replace(mapping)
print("Value counts after mapping (1=female, 0=male):")
print(gender.value_counts(dropna=False))
del mapping


##### Age
print('======================= Age')
# Extract birth date (field 34-0.0) and visit date (field 53-2.0)
birth = total_df['34-0.0']   # YYYY
visit = pd.to_datetime(total_df['53-2.0'], errors="coerce")   # Visit date YYYYMMDD

# Check NaN values
# If there are NaN only in the birth information, but not in the visit date information
birth_only_nan = birth.isna() & visit.notna()
print(f"The number of NaN values only in the birth column: {birth_only_nan.sum()}")

# If there are NaN only in the visit date information, but not in the birth information
visit_only_nan = visit.isna() & birth.notna()
print(f"The number of NaN values only in the visit date column: {visit_only_nan.sum()}")

# Compute rough age (year difference between imaging visit year and birth year)
age_2 = visit.dt.year - birth
# TODO: optionally plot the distribution of age_2 here

# Convert visit date (YYYYMMDD) into integer format (NaT → NaN handled via fillna)
visit_yr_2 = visit.dt.strftime("%Y%m%d").fillna("0").astype(int)


##### Frequency of friend/family visits
print('======================= Frequency of friend/family visits')

# Extract frequency of friend/family visits (field 1031-2.0)
freq_visit = total_df['1031-2.0']
print(f"Unique values in the original frequency of friend/family visits column: {np.unique(freq_visit)}")

# Reverse-coding: 1~7 (almost daily ~ no friends) → inverse scale
# Note: -1 (do not know) and -3 (prefer not to answer) will remain as NaN
mapping = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
freq_visit_2 = freq_visit.replace(mapping)
freq_visit_2 = freq_visit_2.replace([-1, -3], np.nan)

print("Value counts after reverse-coding (1~7: almost daily ~ no friends):")
print(freq_visit_2.value_counts(dropna=False))  # Check value counts
del mapping


##### Sleep duration
print('======================= Sleep duration')

# Extract sleep duration (field 1160-2.0)
sleep_2 = total_df['1160-2.0']
print(f"Unique values in the original sleep duration column: {np.unique(sleep_2)}")

# Note: -1 (do not know) and -3 (prefer not to answer) will remain as NaN
sleep_2 = sleep_2.replace([-1, -3], np.nan)
print(f"Unique values after recoding sleep duration: {np.unique(sleep_2)}")
print(sleep_2.value_counts(dropna=False))  # Check value counts
# TODO: optionally plot the distribution of sleep_2 here


##### Alcohol intake
print('======================= Alcohol intake')

# Extract alcohol intake (field 1558-2.0)
alcohol = total_df['1558-2.0']
print(f"Unique values in the original alcohol intake frequency column: {np.unique(alcohol)}")

# Reverse-coding: 1~6 (daily or almost daily → never)
mapping = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
alcohol_2 = alcohol.replace(mapping)

# Replace -3 (prefer not to answer) with NaN
alcohol_2 = alcohol_2.replace([-3], np.nan)

print("Value counts after reverse-coding (1~6: daily or almost daily → never):")
print(alcohol_2.value_counts(dropna=False))  # Check value counts
del mapping


##### Fluid intelligence
print('======================= Fluid intelligence')
fluid_2 = total_df['20016-2.0']
print("Unique values in the original fluid intelligence column:")
print(fluid_2.value_counts(dropna=False))  # Check value counts


##### Smoking status
print('======================= Smoking status')
smoke_status_2 = total_df['20116-2.0']  # 0: never, 1: previous, 2: current
print(f"Unique values in the smoking status column: {np.unique(smoke_status_2)}")

# Note: -3 (prefer not to answer) will remain as NaN
smoke_status_2 = smoke_status_2.replace([-3], np.nan)

print("Value counts after recoding smoking status:")
print(smoke_status_2.value_counts(dropna=False))  # Check value counts


##### Loneliness
print('======================= Loneliness')
lone_2 = total_df['2020-2.0']
print(f"Unique values in the original loneliness column: {np.unique(lone_2)}")

# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
lone_2 = lone_2.replace([-1, -3], np.nan)

print("Value counts after recoding loneliness:")
print(lone_2.value_counts(dropna=False))  # Check value counts


##### PHQ-2
print('======================= PHQ-2')

# Extract fields: 2050 (frequency of depressed mood) and 2060 (unenthusiasm/disinterest)
phq2_2050 = total_df['2050-2.0']
print(f"Unique values in the original 'frequency of depressed mood in last 2 weeks' column: {np.unique(phq2_2050)}")

# Replace -1 (do not know), -3 (prefer not to answer), and -7 (none of the above) with NaN
phq2_2050 = phq2_2050.replace([-1, -3, -7], np.nan)

phq2_2060 = total_df['2060-2.0']
print(f"Unique values in the original 'frequency of unenthusiasm/disinterest in last 2 weeks' column: {np.unique(phq2_2060)}")

# Replace -1 (do not know), -3 (prefer not to answer), and -7 (none of the above) with NaN
phq2_2060 = phq2_2060.replace([-1, -3, -7], np.nan)

# Compute PHQ-2 score (mean of the two items; NaN if either is missing)
phq2_2 = pd.concat([phq2_2050, phq2_2060], axis=1).mean(axis=1, skipna=False)
print("Value counts for the PHQ-2 score:")
print(phq2_2.value_counts(dropna=False))

# Check missing patterns
both_nan = pd.concat([phq2_2050, phq2_2060], axis=1).isna().all(axis=1)  # Both NaN
nan_2050_only = phq2_2050.isna() & phq2_2060.notna() & ~both_nan  # Only 2050 NaN
nan_2060_only = phq2_2060.isna() & phq2_2050.notna() & ~both_nan  # Only 2060 NaN

print('--- Missingness summary ---')
print(f"Number of subjects with NaN in both 2050 and 2060: {both_nan.sum()}")
print(f"Number of subjects with NaN only in 2050: {nan_2050_only.sum()}")
print(f"Number of subjects with NaN only in 2060: {nan_2060_only.sum()}")


##### Ethnicity
print('======================= Ethnicity')

# Extract ethnicity information (field 21000-0.0)
ethnicity = total_df['21000-0.0']
print(f"Unique values in the original ethnicity column: {np.unique(ethnicity)}")

# Re-code detailed categories into broader groups
mapping = {
    1001: 1, 2001: 1, 3001: 1, 4001: 1,  # White
    1002: 2, 2002: 2, 3002: 2, 4002: 2,  # Mixed
    1003: 3, 2003: 3, 3003: 3, 4003: 3,  # Asian or Asian British
    2004: 4, 3004: 4,                    # Black and Black British
    5: 3, 6: 5                           # 5 → Chinese (Asian); 6 → Others
}
# Group codes:
# 1 = White
# 2 = Mixed
# 3 = Asian or Asian British or Chinese
# 4 = Black and Black British
# 5 = Others

ethnicity_0 = ethnicity.replace(mapping)

# Note: -1 (do not know) and -3 (prefer not to answer) will remain as NaN
ethnicity_0 = ethnicity_0.replace([-1, -3], np.nan)

print("Value counts after re-coding:")
print(ethnicity_0.value_counts(dropna=False))  # Check value counts
del mapping


##### BMI
print('======================= BMI')
# Body Mass Index information (field 21001-2.0)
bmi_2 = total_df['21001-2.0']


##### Confiding to someone
print('======================= Confide')
# Confiding to someone (field 2110-2.0)
confide_2 = total_df['2110-2.0']
print(f"Unique values in the original 'confiding to someone' column: {np.unique(confide_2)}")

# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
confide_2 = confide_2.replace([-1, -3], np.nan)

print("Value counts after re-coding:")
print(confide_2.value_counts(dropna=False))  # Check value counts


##### Wears glasses or contact lenses
print('======================= Wears glasses or contact lenses')
# Wears glasses or contact lenses (field 2207-2.0)
glass_lenses_2 = total_df['2207-2.0']  # 1: Yes, 0: No, -3: Prefer not to answer
print(f"Unique values in the original 'wears glasses or contact lenses' column: {np.unique(glass_lenses_2)}")

glass_lenses_2 = glass_lenses_2.replace([-3], np.nan)
print("Value counts after replacing -3 (prefer not to answer) with NaN:")
print(glass_lenses_2.value_counts(dropna=False))  # Check value counts


##### Other eye problems
print('======================= Other eye problems')
# Other eye problems (field 2227-2.0)
eye_issue_2 = total_df['2227-2.0']  # 1: Yes, 0: No, -3: Prefer not to answer
print(f"Unique values in the original 'other eye problems' column: {np.unique(eye_issue_2)}")

eye_issue_2 = eye_issue_2.replace([-3], np.nan)
print("Value counts after replacing -3 (prefer not to answer) with NaN:")
print(eye_issue_2.value_counts(dropna=False))  # Check value counts


##### Hearing difficulty/problems
print('======================= Hearing difficulty/problems')
# Hearing difficulty/problems (field 2247-2.0)
hearing_issue = total_df['2247-2.0']  # 1: Yes, 0: No, 99: Deaf, -1: Do not know, -3: Prefer not to answer
print(f"Unique values in the original hearing difficulty/problems column: {np.unique(hearing_issue)}")

# Re-code: 0 → 0 (No), 1 → 1 (Yes), 99 → 2 (Deaf)
mapping = {0: 0, 1: 1, 99: 2}
hearing_issue_2 = hearing_issue.replace(mapping)

# Note: -1 (Do not know) and -3 (Prefer not to answer) are treated as NaN
hearing_issue_2 = hearing_issue_2.replace([-1, -3], np.nan)

print("Value counts after re-coding (0=No, 1=Yes, 2=Deaf):")
print(hearing_issue_2.value_counts(dropna=False))  # Check value counts
del mapping


##### Hearing difficulty/problems with background noise
print('======================= Hearing difficulty/problems with background noise')
# Hearing difficulty/problems with background noise (field 2257-2.0)
hearing_issue_bg_2 = total_df['2257-2.0']  # 1: Yes, 0: No, -1: Do not know, -3: Prefer not to answer
print(f"Unique values in the original background noise hearing difficulty column: {np.unique(hearing_issue_bg_2)}")

hearing_issue_bg_2 = hearing_issue_bg_2.replace([-1, -3], np.nan)
print("Value counts after replacing -1 and -3 with NaN:")
print(hearing_issue_bg_2.value_counts(dropna=False))  # Check value counts


##### Hearing aid user
print('======================= Hearing aid user')
# Hearing aid use (field 3393-2.0)
hearing_aid_2 = total_df['3393-2.0']  # 1: Yes, 0: No, -3: Prefer not to answer
print(f"Unique values in the original hearing aid user column: {np.unique(hearing_aid_2)}")

hearing_aid_2 = hearing_aid_2.replace([-3], np.nan)
print("Value counts after replacing -3 (prefer not to answer) with NaN:")
print(hearing_aid_2.value_counts(dropna=False))  # Check value counts


##### Health satisfaction
print('======================= Health satisfaction')
# Health satisfaction (field 4548-2.0)
hlth_sat = total_df['4548-2.0']
print(f"Unique values in the original health satisfaction column: {np.unique(hlth_sat)}")

# Original coding: 1 = extremely happy, 6 = extremely unhappy
# After reverse-coding: 1 = extremely unhappy, 6 = extremely happy
# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
mapping = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
hlth_sat_2 = hlth_sat.replace(mapping)

# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
hlth_sat_2 = hlth_sat_2.replace([-1, -3], np.nan)

print("Value counts after reverse-coding:")
print(hlth_sat_2.value_counts(dropna=False))  # Check value counts
del mapping


##### Family relationship satisfaction
print('======================= Family relationship satisfaction')
# Family relationship satisfaction (field 4559-2.0)
fam_sat = total_df['4559-2.0']  # 1–6: extremely happy → extremely unhappy, -1: Do not know, -3: Prefer not to answer
print(f"Unique values in the original family relationship satisfaction column: {np.unique(fam_sat)}")

# Original coding: 1 = extremely happy, 6 = extremely unhappy
# After reverse-coding: 1 = extremely unhappy, 6 = extremely happy
# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
mapping = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
fam_sat_2 = fam_sat.replace(mapping)

# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
fam_sat_2 = fam_sat_2.replace([-1, -3], np.nan)

print("Value counts after reverse-coding:")
print(fam_sat_2.value_counts(dropna=False))  # Check value counts
del mapping


##### Friendships satisfaction
print('======================= Friendships satisfaction')
# Friendships satisfaction (field 4570-2.0)
frnd_sat = total_df['4570-2.0']
print(f"Unique values in the original friendships satisfaction column: {np.unique(frnd_sat)}")

# Original coding: 1 = extremely happy, 6 = extremely unhappy
# After reverse-coding: 1 = extremely unhappy, 6 = extremely happy
# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
mapping = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
frnd_sat_2 = frnd_sat.replace(mapping)

# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
frnd_sat_2 = frnd_sat_2.replace([-1, -3], np.nan)

print("Value counts after reverse-coding:")
print(frnd_sat_2.value_counts(dropna=False))  # Check value counts
del mapping


##### Financial situation satisfaction
print('======================= Financial situation satisfaction')
# Financial situation satisfaction (field 4581-2.0)
fncl_sat = total_df['4581-2.0']
print(f"Unique values in the original financial situation satisfaction column: {np.unique(fncl_sat)}")

# Original coding: 1 = extremely happy, 6 = extremely unhappy
# After reverse-coding: 1 = extremely unhappy, 6 = extremely happy
# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
mapping = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
fncl_sat_2 = fncl_sat.replace(mapping)

# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
fncl_sat_2 = fncl_sat_2.replace([-1, -3], np.nan)

print("Value counts after reverse-coding:")
print(fncl_sat_2.value_counts(dropna=False))  # Check value counts
del mapping


##### Private healthcare
print('======================= Private healthcare')
# Private healthcare (field 4674-2.0)
# 1–4: Yes, all of the time → No, never; -1: Do not know; -3: Prefer not to answer
hthcare = total_df['4674-2.0']
print(f"Unique values in the original private healthcare column: {np.unique(hthcare)}")

# Original coding: 1 = yes, all of the time; 4 = no, never
# After reverse-coding: 1 = no, never; 4 = yes, all of the time
mapping = {1: 4, 2: 3, 3: 2, 4: 1}
hthcare_2 = hthcare.replace(mapping)

# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
hthcare_2 = hthcare_2.replace([-1, -3], np.nan)

print("Value counts after reverse-coding:")
print(hthcare_2.value_counts(dropna=False))  # Check value counts
del mapping


##### Qualifications (Education)
print('======================= Qualifications (Education)')
# Highest qualification (field 6138-2.0)
# Mapping based on years of education equivalence (ISCED-based)
ed = total_df['6138-2.0']
print(f"Unique values in the original education column: {np.unique(ed)}")

mapping = {1: 20, 2: 13, 3: 10, 4: 10, 5: 19, 6: 15, -7: 7}
ed_yr_2 = ed.replace(mapping)

# Note: -3 (prefer not to answer) are treated as NaN
ed_yr_2 = ed_yr_2.replace([-3], np.nan)

print("Value counts after mapping to education years:")
print(ed_yr_2.value_counts(dropna=False))  # Check value counts
del mapping


##### How are people in household related to participant
print('======================= How are people in household related to participant')
# Relationship to household members (field 6141-2.0)
# 1: Spouse/Partner → 1; all others → 0
marital = total_df['6141-2.0']
print(f"Unique values in the original household relationship column: {np.unique(marital)}")

mapping = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
marital_2 = marital.replace(mapping)

# Note: -3 (prefer not to answer) are treated as NaN
marital_2 = marital_2.replace([-3], np.nan)

print("Value counts after binary mapping (1=spouse/partner [married], 0=others [single]):")
print(marital_2.value_counts(dropna=False))  # Check value counts
del mapping


##### Current employment status
print('======================= Current employment status')
# Current employment status (field 6142-2.0)
# 1–7: Employment categories; -7: None of the above; -3: Prefer not to answer
emp_2 = total_df['6142-2.0']
print(f"Unique values in the original employment status column: {np.unique(emp_2)}")

# Note: -1 (do not know), -3 (prefer not to answer), and -7 (None of the above) are treated as NaN
emp_2 = emp_2.replace([-1, -3, -7], np.nan)

print("Value counts for employment status:")
print(emp_2.value_counts(dropna=False))  # Check value counts


##### Leisure/social activities - Categorical
print('======================= Leisure/social activities - Categorical')

soc_cols = ['6160-2.0', '6160-2.1', '6160-2.2', '6160-2.3', '6160-2.4']
soc_raw = total_df[soc_cols]

print(f"Unique values in the original leisure/social activities columns: {np.unique(soc_raw.to_numpy())}")

# --- Step 1. Check counts before any replacement ---
n_nan_init   = soc_raw.isna().sum().sum()
n_minus3_init = (soc_raw == -3).sum().sum()
n_minus7_init = (soc_raw == -7).sum().sum()
n_zero_init   = (soc_raw == 0).sum().sum()
print(f"[INFO] Initial NaN count: {n_nan_init}")
print(f"[INFO] Initial -3 count:  {n_minus3_init}")
print(f"[INFO] Initial -7 count:  {n_minus7_init}")
print(f"[INFO] Initial 0 count:   {n_zero_init}")

# --- Step 2. Replace -3 → NaN ---
S = soc_raw.replace([-3], np.nan)

# Check counts after -3 → NaN
n_nan_after3 = S.isna().sum().sum()
n_minus3_after = (S == -3).sum().sum()
n_minus7_after = (S == -7).sum().sum()
n_zero_after3  = (S == 0).sum().sum()
print(f"[INFO] After replacing -3 → NaN:")
print(f"       NaN count: {n_nan_after3}, -3 count (should be NaN): {n_minus3_after}, -7 count: {n_minus7_after}, 0 count: {n_zero_after3}")

# --- Step 3. Replace -7 → 0 ---
S = S.replace([-7], 0)

# Check counts after -7→0
n_nan_after7 = S.isna().sum().sum()
n_minus7_final = (S == -7).sum().sum()
n_zero_after7  = (S == 0).sum().sum()
print(f"[INFO] After replacing -7→0:")
print(f"       NaN count: {n_nan_after7}, -7 count (should be 0): {n_minus7_final}, 0 count: {n_zero_after7}")

# --- Step 4. Generate categorical variables ---
valid_vals = [1, 2, 3, 4, 5]
valid_mask = S.isin(valid_vals)            # (n,5): each entry is 0~5 or not
all_nan    = S.isna().all(axis=1)          # rows where all entries are NaN

soc_out = pd.DataFrame(index=total_df.index)
for k in valid_vals:
    colname = f"social_act_2_cat{k}"
    soc_out[colname] = S.eq(k).any(axis=1).astype(int)
    soc_out.loc[all_nan, colname] = np.nan

# Optionally expose as separate variables
for k in valid_vals:
    varname = f"social_act_2_cat{k}"
    globals()[varname] = soc_out[varname]

print('Social act category 1:', social_act_2_cat1.value_counts(dropna=False))
print('Social act category 2:', social_act_2_cat2.value_counts(dropna=False))
print('Social act category 3:', social_act_2_cat3.value_counts(dropna=False))
print('Social act category 4:', social_act_2_cat4.value_counts(dropna=False))
print('Social act category 5:', social_act_2_cat5.value_counts(dropna=False))


##### Leisure/social activities - Continuous
print('======================= Leisure/social activities - Continuous')
count_valid = valid_mask.sum(axis=1).astype(float)  # number of valid (0~5) responses
count_valid[all_nan] = np.nan

social_act_n_2 = count_valid
print("Value counts after counting the number of social activities:")
print(social_act_n_2.value_counts(dropna=False))


##### Number in household
print('======================= Number in household')
# Number in household (field 709-2.0)
N_fam_2 = total_df['709-2.0']
print(f"Unique values in the number in household column: {np.unique(N_fam_2)}")

# Note: -1 (do not know) and -3 (prefer not to answer) are treated as NaN
N_fam_2 = N_fam_2.replace([-1, -3], np.nan)

print("Value counts for the number in household:")
print(N_fam_2.value_counts(dropna=False))  # Check value counts
# TODO: optionally plot the distribution of N_fam_2 here


##### Average total household income before tax
print('======================= Average total household income before tax')
# Average total household income before tax (field 738-2.0)
income_fam_2 = total_df['738-2.0']  # 1–5: income categories; -1: Do not know; -3: Prefer not to answer
print(f"Unique values in the original income column: {np.unique(income_fam_2)}")

income_fam_2 = income_fam_2.replace([-1, -3], np.nan)
print("Value counts after replacing -1 (Do not know) and -3 (Prefer not to answer) with NaN:")
print(income_fam_2.value_counts(dropna=False))  # Check value counts


### MET
print('======================= MET')
def compute_met_no_row_drop(df, exclude_extreme_outliers=True):
    """
    Compute MET-min/week without dropping rows.

    - Keeps all rows (N fixed)
    - For duration columns: values <10 minutes set to 0 (IPAQ rule),
      but missing values (including -1, -2, -3) are kept as NaN.
    - Codes -1, -2, -3 (do not know / unable to walk / prefer not to answer)
      are treated as missing and converted to NaN.
    - Days are truncated to [0, 7], minutes/day to [0, 180].
    - MET_walk, MET_mod, MET_vig are computed from (days × minutes × MET factor).
    - MET_total is the sum of the three METs where:
        * if all three METs are NaN → MET_total = NaN
        * otherwise, NaNs in MET_walk/MET_mod/MET_vig are treated as 0 when summing.
    - Extreme cases (total daily minutes > 960) have all MET_* (including MET_total) set to NaN.
    """

    day_cols = ['864-2.0', '884-2.0', '904-2.0']  # walk/mod/vig days per week
    dur_cols = ['874-2.0', '894-2.0', '914-2.0']  # walk/mod/vig minutes per day
    cols = day_cols + dur_cols

    out = df[cols].copy()

    # 1) Convert codes -1, -2, -3 to NaN (missing values)
    out[cols] = out[cols].replace([-1, -2, -3], np.nan)

    # 2) Apply IPAQ rule: values <10 minutes are set to 0
    #    Missing values remain as NaN
    for dur_col in dur_cols:
        out[dur_col] = np.where(out[dur_col].ge(10), out[dur_col],
                                np.where(out[dur_col].isna(), np.nan, 0.0))

    # 3) Apply truncation limits according to IPAQ:
    #    - Days are limited to the range [0, 7]
    #    - Minutes per day are limited to the range [0, 180]
    for dcol in day_cols:
        out[dcol] = out[dcol].clip(0, 7)
    for tcol in dur_cols:
        out[tcol] = out[tcol].clip(0, 180)

    # 4) Compute MET values for each activity type
    #    MET = MET_factor × minutes/day × days/week
    WALK_MET, MOD_MET, VIG_MET = 3.3, 4.0, 8.0
    out["MET_walk"] = WALK_MET * out['874-2.0'] * out['864-2.0']
    out["MET_mod"]  = MOD_MET  * out['894-2.0'] * out['884-2.0']
    out["MET_vig"]  = VIG_MET  * out['914-2.0'] * out['904-2.0']

    # 5) Compute total MET:
    #    - If all three MET values are NaN → MET_total = NaN
    #    - Otherwise, replace NaNs with 0 and sum the three
    met_cols = ["MET_walk", "MET_mod", "MET_vig"]
    all_nan_mask = out[met_cols].isna().all(axis=1)

    out["MET_total"] = out[met_cols].fillna(0).sum(axis=1)
    out.loc[all_nan_mask, "MET_total"] = np.nan

    # 6) Mark extreme cases as NaN:
    #    If total minutes/day > 960, all MET fields are masked
    if exclude_extreme_outliers:
        total_daily_mins = out['874-2.0'] + out['894-2.0'] + out['914-2.0']
        mask_extreme = total_daily_mins > 960
        out.loc[mask_extreme, ["MET_walk","MET_mod","MET_vig","MET_total"]] = np.nan
        print(f"[INFO] Number of extreme-outlier subjects masked (>960 min/day): {mask_extreme.sum()}")

    # 7) Summary checks
    n_valid_met = out["MET_total"].notna().sum()
    print(f"[INFO] Number of subjects with valid MET_total: {n_valid_met}")

    print("=== [DEBUG CHECK] ===")
    print(f"Total NaN count across all columns: {out.isna().sum().sum()}")
    print(f"Rows with all input cols NaN: {out[cols].isna().all(axis=1).sum()}")
    print(f"Rows with all MET outputs NaN: {out[['MET_walk','MET_mod','MET_vig','MET_total']].isna().all(axis=1).sum()}")
    print("======================")

    return out


act_df = pd.DataFrame({
    "864-2.0": total_df['864-2.0'],
    "874-2.0":  total_df['874-2.0'],
    "884-2.0":  total_df['884-2.0'],
    "894-2.0":   total_df['894-2.0'],
    "904-2.0":  total_df['904-2.0'],
    "914-2.0":   total_df['914-2.0']})

met_df = compute_met_no_row_drop(act_df)
met_2 = met_df["MET_total"]

# 864-2.0: walk_week - In a typical WEEK, on how many days did you walk for at least 10 minutes at a time? (Include walking that you do at work, travelling to and from work, and for sport or leisure)
# 874-2.0: walk_dur  - How many minutes did you usually spend walking on a typical DAY?
# 884-2.0: mod_week  - In a typical WEEK, on how many days did you do 10 minutes or more of moderate physical activities like carrying light loads, cycling at normal pace? (Do not include walking)
# 894-2.0: mod_dur   - How many minutes did you usually spend doing moderate activities on a typical DAY?
# 904-2.0: vig_week  - In a typical WEEK, how many days did you do 10 minutes or more of vigorous physical activity? (These are activities that make you sweat or breathe hard such as fast cycling, aerobics, heavy lifting)
# 914-2.0: vig_dur   - How many minutes did you usually spend doing vigorous activities on a typical DAY?

print("Summary of MET_total:")
print(met_2.describe())
print(met_2.value_counts(bins=10, dropna=False))


### Cortical thickness
print('======================= Brain (Cortical thickness)')
# Define index ranges for left and right hemisphere cortical thickness fields
left_index = np.arange(26756, 26788+1)
right_index = np.arange(26857, 26889+1)

# Convert indices to strings and append '-2.0' to indicate the MRI visit instance
left_cols = [f"{str(idx)}-2.0" for idx in left_index]
right_cols = [f"{str(idx)}-2.0" for idx in right_index]

# Check if all expected columns exist in the dataset
missing_left = [c for c in left_cols if c not in total_df.columns]
missing_right = [c for c in right_cols if c not in total_df.columns]
if missing_left:
    print(f"[WARN] Missing left hemisphere columns: {len(missing_left)} (e.g., {missing_left[:3]}...)")
if missing_right:
    print(f"[WARN] Missing right hemisphere columns: {len(missing_right)} (e.g., {missing_right[:3]}...)")

# Extract hemisphere-specific columns
left_h = total_df[["eid"] + [c for c in left_cols if c in total_df.columns]].copy()
right_h = total_df[["eid"] + [c for c in right_cols if c in total_df.columns]].copy()

# Merge left and right hemispheres into a single DataFrame
brain_hemisphere = pd.merge(left_h, right_h, on="eid", how="left")

# Check average missing rate across cortical thickness columns
hemi_cols = [c for c in brain_hemisphere.columns if c != "eid"]
nan_ratio = brain_hemisphere[hemi_cols].isna().mean().mean()
print(f"Average missing rate across cortical thickness columns: {nan_ratio:.2%}")


##### Concatenate all variables into a single DataFrame
# social_act_2_cat1..5 correspond to:
# 1: sport, 2: pub, 3: religious, 4: education, 5: other
all_vars = [eid, visit_yr_2, gender, age_2, ethnicity_0, marital_2,        # Demographic
            ed_yr_2, emp_2, income_fam_2, fncl_sat_2, hthcare_2,           # Socioeconomic
            lone_2, social_act_n_2, social_act_2_cat1,  # Network
            social_act_2_cat2, social_act_2_cat3, social_act_2_cat4,       # Network
            social_act_2_cat5,                                             # Network
            freq_visit_2, confide_2, fam_sat_2, frnd_sat_2, N_fam_2,       # Network
            smoke_status_2, alcohol_2, glass_lenses_2, eye_issue_2,        # Health
            hearing_issue_2, hearing_issue_bg_2, hearing_aid_2,            # Health
            phq2_2, hlth_sat_2, sleep_2, bmi_2,                            # Health
            met_2,                                                         # Physical
            fluid_2]                                                       # Cognition

all_var_names = ["eid", "visit_yr_2", "gender", "age_2", "ethnicity_0", "marital_2",        # Demographic
                 "ed_yr_2", "emp_2", "income_fam_2", "fncl_sat_2", "hthcare_2",             # Socioeconomic
                 "lone_2", "social_act_n_2", "social_act_2_sport",     # Network
                 "social_act_2_pub", "social_act_2_religious", "social_act_2_education",    # Network
                 "social_act_2_other",                                                      # Network
                 "freq_visit_2", "confide_2", "fam_sat_2", "frnd_sat_2", "N_fam_2",         # Network
                 "smoke_status_2",  "alcohol_2", "glass_lenses_2", "eye_issue_2",           # Health
                 "hearing_issue_2", "hearing_issue_bg_2", "hearing_aid_2",                  # Health
                 "phq2_2", "hlth_sat_2", "sleep_2", "bmi_2",                                # Health
                 "met_2",                                                                   # Physical
                 "fluid_2"]                                                                 # Cognition


# Concatenate all selected variables into a single DataFrame
final_df = pd.concat(all_vars, axis=1)
final_df.columns = all_var_names

# Align the final DataFrame to the original eid order
final_df = final_df.set_index("eid").reindex(total_df["eid"]).reset_index()
print("Final DataFrame shape:", final_df.shape)
print("Head of the final DataFrame:")
print(final_df.head())

final_df = final_df.merge(brain_hemisphere, on="eid", how="left")
print("DataFrame shape after adding cortical thickness:", final_df.shape)
print("Head of the DataFrame after adding cortical thickness:")
print(final_df.head())


### Disease (first occurrence ICD-10 codes)
# Helper function to detect date-like columns and convert them safely
date_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def is_date_col(s: pd.Series, thresh: float = 0.8) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object:
        nonnull = s.dropna().astype(str)
        if len(nonnull) == 0:
            return False
        return (nonnull.str.match(date_pat)).mean() >= thresh
    return False

df = total_df.copy()
disease_idx = ['130000', '130002', '130004', '130006', '130008', '130010', '130012', '130014', '130016', '130018', '130020', '130022', '130024', '130026', '130028', '130030', '130032', '130034', '130036', '130038', '130040', '130042', '130044', '130046', '130048', '130050', '130052', '130054', '130056', '130058', '130060', '130062', '130064', '130066', '130068', '130070', '130072', '130074', '130076', '130078', '130080', '130082', '130084', '130086', '130088', '130090', '130092', '130094', '130096', '130098', '130100', '130102', '130104', '130106', '130108', '130110', '130112', '130114', '130116', '130118', '130120', '130122', '130124', '130126', '130128', '130130', '130132', '130134', '130136', '130138', '130140', '130142', '130144', '130146', '130148', '130150', '130152', '130154', '130156', '130158', '130160', '130162', '130164', '130166', '130168', '130170', '130172', '130174', '130176', '130178', '130180', '130182', '130184', '130186', '130188', '130190', '130192', '130194', '130196', '130198', '130200', '130202', '130204', '130206', '130208', '130210', '130212', '130214', '130216', '130218', '130220', '130222', '130224', '130226', '130228', '130230', '130232', '130234', '130236', '130238', '130240', '130242', '130244', '130246', '130248', '130250', '130252', '130254', '130256', '130258', '130260', '130262', '130264', '130266', '130268', '130270', '130272', '130274', '130276', '130278', '130280', '130282', '130284', '130286', '130288', '130290', '130292', '130294', '130296', '130298', '130300', '130302', '130304', '130306', '130308', '130310', '130312', '130314', '130316', '130318', '130320', '130322', '130324', '130326', '130328', '130330', '130332', '130334', '130336', '130338', '130340', '130342', '130344', '130622', '130624', '130626', '130628', '130630', '130632', '130634', '130636', '130638', '130640', '130642', '130644', '130646', '130648', '130650', '130652', '130654', '130656', '130658', '130660', '130662', '130664', '130666', '130668', '130670', '130672', '130674', '130676', '130678', '130680', '130682', '130684', '130686', '130688', '130690', '130692', '130694', '130696', '130698', '130700', '130702', '130704', '130706', '130708', '130710', '130712', '130714', '130716', '130718', '130720', '130722', '130724', '130726', '130728', '130730', '130732', '130734', '130736', '130738', '130740', '130742', '130744', '130746', '130748', '130750', '130752', '130754', '130756', '130758', '130760', '130762', '130764', '130766', '130768', '130770', '130772', '130774', '130776', '130778', '130780', '130782', '130784', '130786', '130788', '130790', '130792', '130794', '130796', '130798', '130800', '130802', '130804', '130806', '130808', '130810', '130812', '130814', '130816', '130818', '130820', '130822', '130824', '130826', '130828', '130830', '130832', '130834', '130836', '130838', '130840', '130842', '130844', '130846', '130848', '130850', '130852', '130854', '130856', '130858', '130860', '130862', '130864', '130866', '130868', '130870', '130872', '130874', '130876', '130878', '130880', '130882', '130884', '130886', '130888', '130890', '130892', '130894', '130896', '130898', '130900', '130902', '130904', '130906', '130908', '130910', '130912', '130914', '130916', '130918', '130920', '130922', '130924', '130926', '130928', '130930', '130932', '130934', '130936', '130938', '130940', '130942', '130944', '130946', '130948', '130950', '130952', '130954', '130956', '130958', '130960', '130962', '130964', '130966', '130968', '130970', '130972', '130974', '130976', '130978', '130980', '130982', '130984', '130986', '130988', '130990', '130992', '130994', '130996', '130998', '131000', '131002', '131004', '131006', '131008', '131010', '131012', '131014', '131016', '131018', '131020', '131022', '131024', '131026', '131028', '131030', '131032', '131034', '131036', '131038', '131040', '131042', '131044', '131046', '131048', '131050', '131052', '131054', '131056', '131058', '131060', '131062', '131064', '131066', '131068', '131070', '131072', '131074', '131076', '131078', '131080', '131082', '131084', '131086', '131088', '131090', '131092', '131094', '131096', '131098', '131100', '131102', '131104', '131106', '131108', '131110', '131112', '131114', '131116', '131118', '131120', '131122', '131124', '131126', '131128', '131130', '131132', '131134', '131136', '131138', '131140', '131142', '131144', '131146', '131148', '131150', '131152', '131154', '131156', '131158', '131160', '131162', '131164', '131166', '131168', '131170', '131172', '131174', '131176', '131178', '131180', '131182', '131184', '131186', '131188', '131190', '131192', '131194', '131196', '131198', '131200', '131202', '131204', '131206', '131208', '131210', '131212', '131214', '131216', '131218', '131220', '131222', '131224', '131226', '131228', '131230', '131232', '131234', '131236', '131238', '131240', '131242', '131244', '131246', '131248', '131250', '131252', '131254', '131256', '131258', '131260', '131262', '131264', '131266', '131268', '131270', '131272', '131274', '131276', '131278', '131280', '131282', '131284', '131286', '131288', '131290', '131292', '131294', '131296', '131298', '131300', '131302', '131304', '131306', '131308', '131310', '131312', '131314', '131316', '131318', '131320', '131322', '131324', '131326', '131328', '131330', '131332', '131334', '131336', '131338', '131340', '131342', '131344', '131346', '131348', '131350', '131352', '131354', '131356', '131358', '131360', '131362', '131364', '131366', '131368', '131370', '131372', '131374', '131376', '131378', '131380', '131382', '131384', '131386', '131388', '131390', '131392', '131394', '131396', '131398', '131400', '131402', '131404', '131406', '131408', '131410', '131412', '131414', '131416', '131418', '131420', '131422', '131424', '131426', '131428', '131430', '131432', '131434', '131436', '131438', '131440', '131442', '131444', '131446', '131448', '131450', '131452', '131454', '131456', '131458', '131460', '131462', '131464', '131466', '131468', '131470', '131472', '131474', '131476', '131478', '131480', '131482', '131484', '131486', '131488', '131490', '131492', '131494', '131496', '131498', '131500', '131502', '131504', '131506', '131508', '131510', '131512', '131514', '131516', '131518', '131520', '131522', '131524', '131526', '131528', '131530', '131532', '131534', '131536', '131538', '131540', '131542', '131544', '131546', '131548', '131550', '131552', '131554', '131556', '131558', '131560', '131562', '131564', '131566', '131568', '131570', '131572', '131574', '131576', '131578', '131580', '131582', '131584', '131586', '131588', '131590', '131592', '131594', '131596', '131598', '131600', '131602', '131604', '131606', '131608', '131610', '131612', '131614', '131616', '131618', '131620', '131622', '131624', '131626', '131628', '131630', '131632', '131634', '131636', '131638', '131640', '131642', '131644', '131646', '131648', '131650', '131652', '131654', '131656', '131658', '131660', '131662', '131664', '131666', '131668', '131670', '131672', '131674', '131676', '131678', '131680', '131682', '131684', '131686', '131688', '131690', '131692', '131694', '131696', '131698', '131700', '131702', '131704', '131706', '131708', '131710', '131712', '131714', '131716', '131718', '131720', '131722', '131724', '131726', '131728', '131730', '131732', '131734', '131736', '131738', '131740', '131742', '131744', '131746', '131748', '131750', '131752', '131754', '131756', '131758', '131760', '131762', '131764', '131766', '131768', '131770', '131772', '131774', '131776', '131778', '131780', '131782', '131784', '131786', '131788', '131790', '131792', '131794', '131796', '131798', '131800', '131802', '131804', '131806', '131808', '131810', '131812', '131814', '131816', '131818', '131820', '131822', '131824', '131826', '131828', '131830', '131832', '131834', '131836', '131838', '131840', '131842', '131844', '131846', '131848', '131850', '131852', '131854', '131856', '131858', '131860', '131862', '131864', '131866', '131868', '131870', '131872', '131874', '131876', '131878', '131880', '131882', '131884', '131886', '131888', '131890', '131892', '131894', '131896', '131898', '131900', '131902', '131904', '131906', '131908', '131910', '131912', '131914', '131916', '131918', '131920', '131922', '131924', '131926', '131928', '131930', '131932', '131934', '131936', '131938', '131940', '131942', '131944', '131946', '131948', '131950', '131952', '131954', '131956', '131958', '131960', '131962', '131964', '131966', '131968', '131970', '131972', '131974', '131976', '131978', '131980', '131982', '131984', '131986', '131988', '131990', '131992', '131994', '131996', '131998', '132000', '132002', '132004', '132006', '132008', '132010', '132012', '132014', '132016', '132018', '132020', '132022', '132024', '132026', '132028', '132030', '132032', '132034', '132036', '132038', '132040', '132042', '132044', '132046', '132048', '132050', '132052', '132054', '132056', '132058', '132060', '132062', '132064', '132066', '132068', '132070', '132072', '132074', '132076', '132078', '132080', '132082', '132084', '132086', '132088', '132090', '132092', '132094', '132096', '132098', '132100', '132102', '132104', '132106', '132108', '132110', '132112', '132114', '132116', '132118', '132120', '132122', '132124', '132126', '132128', '132130', '132132', '132134', '132136', '132138', '132140', '132142', '132144', '132146', '132148', '132150', '132152', '132154', '132156', '132158', '132160', '132162', '132164', '132166', '132168', '132170', '132172', '132174', '132176', '132178', '132180', '132182', '132184', '132186', '132188', '132190', '132192', '132194', '132196', '132198', '132200', '132202', '132204', '132206', '132208', '132210', '132212', '132214', '132216', '132218', '132220', '132222', '132224', '132226', '132228', '132230', '132232', '132234', '132236', '132238', '132240', '132242', '132244', '132246', '132248', '132250', '132252', '132254', '132256', '132258', '132260', '132262', '132264', '132266', '132268', '132270', '132272', '132274', '132276', '132278', '132280', '132282', '132284', '132286', '132288', '132290', '132292', '132294', '132296', '132298', '132300', '132302', '132304', '132306', '132308', '132310', '132312', '132314', '132316', '132318', '132320', '132322', '132324', '132326', '132328', '132330', '132332', '132334', '132336', '132338', '132340', '132342', '132344', '132346', '132348', '132350', '132352', '132354', '132356', '132358', '132360', '132362', '132364', '132366', '132368', '132370', '132372', '132374', '132376', '132378', '132380', '132382', '132384', '132386', '132388', '132390', '132392', '132394', '132396', '132398', '132400', '132402', '132404', '132406', '132408', '132410', '132412', '132414', '132416', '132418', '132420', '132422', '132424', '132426', '132428', '132430', '132432', '132434', '132436', '132438', '132440', '132442', '132444', '132446', '132448', '132450', '132452', '132454', '132456', '132458', '132460', '132462', '132464', '132466', '132468', '132470', '132472', '132474', '132476', '132478', '132480', '132482', '132484', '132486', '132488', '132490', '132492', '132494', '132496', '132498', '132500', '132502', '132504', '132506', '132508', '132510', '132512', '132514', '132516', '132518', '132520', '132522', '132524', '132526', '132528', '132530', '132532', '132534', '132536', '132538', '132540', '132542', '132544', '132546', '132548', '132550', '132552', '132554', '132556', '132558', '132560', '132562', '132564', '132566', '132568', '132570', '132572', '132574', '132576', '132578', '132580', '132582', '132584', '132586', '132588', '132590', '132592', '132594', '132596', '132598', '132600', '132602', '132604']
disease_name = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A46', 'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 'A60', 'A63', 'A64', 'A65', 'A66', 'A67', 'A68', 'A69', 'A70', 'A71', 'A74', 'A75', 'A77', 'A78', 'A79', 'A80', 'A81', 'A82', 'A83', 'A84', 'A85', 'A86', 'A87', 'A88', 'A89', 'A90', 'A91', 'A92', 'A93', 'A94', 'A95', 'A96', 'A97', 'A98', 'A99', 'B00', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B30', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40', 'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47', 'B48', 'B49', 'B50', 'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57', 'B58', 'B59', 'B60', 'B64', 'B65', 'B66', 'B67', 'B68', 'B69', 'B70', 'B71', 'B72', 'B73', 'B74', 'B75', 'B76', 'B77', 'B78', 'B79', 'B80', 'B81', 'B82', 'B83', 'B85', 'B86', 'B87', 'B88', 'B89', 'B90', 'B91', 'B92', 'B94', 'B95', 'B96', 'B97', 'B98', 'B99', 'D50', 'D51', 'D52', 'D53', 'D55', 'D56', 'D57', 'D58', 'D59', 'D60', 'D61', 'D62', 'D63', 'D64', 'D65', 'D66', 'D67', 'D68', 'D69', 'D70', 'D71', 'D72', 'D73', 'D74', 'D75', 'D76', 'D77', 'D80', 'D81', 'D82', 'D83', 'D84', 'D86', 'D89', 'E00', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E32', 'E34', 'E35', 'E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E56', 'E58', 'E59', 'E60', 'E61', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E70', 'E71', 'E72', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E83', 'E84', 'E85', 'E86', 'E87', 'E88', 'E89', 'E90', 'F00', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F09', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F28', 'F29', 'F30', 'F31', 'F32', 'F33', 'F34', 'F38', 'F39', 'F40', 'F41', 'F42', 'F43', 'F44', 'F45', 'F48', 'F50', 'F51', 'F52', 'F53', 'F54', 'F55', 'F59', 'F60', 'F61', 'F62', 'F63', 'F64', 'F65', 'F66', 'F68', 'F69', 'F70', 'F71', 'F72', 'F73', 'F78', 'F79', 'F80', 'F81', 'F82', 'F83', 'F84', 'F88', 'F89', 'F90', 'F91', 'F92', 'F93', 'F94', 'F95', 'F98', 'F99', 'G00', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'G14', 'G20', 'G21', 'G22', 'G23', 'G24', 'G25', 'G26', 'G30', 'G31', 'G32', 'G35', 'G36', 'G37', 'G40', 'G41', 'G43', 'G44', 'G45', 'G46', 'G47', 'G50', 'G51', 'G52', 'G53', 'G54', 'G55', 'G56', 'G57', 'G58', 'G59', 'G60', 'G61', 'G62', 'G63', 'G64', 'G70', 'G71', 'G72', 'G73', 'G80', 'G81', 'G82', 'G83', 'G90', 'G91', 'G92', 'G93', 'G94', 'G95', 'G96', 'G97', 'G98', 'G99', 'H00', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H10', 'H11', 'H13', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H25', 'H26', 'H27', 'H28', 'H30', 'H31', 'H32', 'H33', 'H34', 'H35', 'H36', 'H40', 'H42', 'H43', 'H44', 'H45', 'H46', 'H47', 'H48', 'H49', 'H50', 'H51', 'H52', 'H53', 'H54', 'H55', 'H57', 'H58', 'H59', 'H60', 'H61', 'H62', 'H65', 'H66', 'H67', 'H68', 'H69', 'H70', 'H71', 'H72', 'H73', 'H74', 'H75', 'H80', 'H81', 'H82', 'H83', 'H90', 'H91', 'H92', 'H93', 'H94', 'H95', 'I00', 'I01', 'I02', 'I05', 'I06', 'I07', 'I08', 'I09', 'I10', 'I11', 'I12', 'I13', 'I15', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I30', 'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40', 'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'I50', 'I51', 'I52', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'I70', 'I71', 'I72', 'I73', 'I74', 'I77', 'I78', 'I79', 'I80', 'I81', 'I82', 'I83', 'I84', 'I85', 'I86', 'I87', 'I88', 'I89', 'I95', 'I97', 'I98', 'I99', 'J00', 'J01', 'J02', 'J03', 'J04', 'J05', 'J06', 'J09', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J20', 'J21', 'J22', 'J30', 'J31', 'J32', 'J33', 'J34', 'J35', 'J36', 'J37', 'J38', 'J39', 'J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67', 'J68', 'J69', 'J70', 'J80', 'J81', 'J82', 'J84', 'J85', 'J86', 'J90', 'J91', 'J92', 'J93', 'J94', 'J95', 'J96', 'J98', 'J99', 'K00', 'K01', 'K02', 'K03', 'K04', 'K05', 'K06', 'K07', 'K08', 'K09', 'K10', 'K11', 'K12', 'K13', 'K14', 'K20', 'K21', 'K22', 'K23', 'K25', 'K26', 'K27', 'K28', 'K29', 'K30', 'K31', 'K35', 'K36', 'K37', 'K38', 'K40', 'K41', 'K42', 'K43', 'K44', 'K45', 'K46', 'K50', 'K51', 'K52', 'K55', 'K56', 'K57', 'K58', 'K59', 'K60', 'K61', 'K62', 'K63', 'K64', 'K65', 'K66', 'K67', 'K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77', 'K80', 'K81', 'K82', 'K83', 'K85', 'K86', 'K87', 'K90', 'K91', 'K92', 'K93', 'L00', 'L01', 'L02', 'L03', 'L04', 'L05', 'L08', 'L10', 'L11', 'L12', 'L13', 'L14', 'L20', 'L21', 'L22', 'L23', 'L24', 'L25', 'L26', 'L27', 'L28', 'L29', 'L30', 'L40', 'L41', 'L42', 'L43', 'L44', 'L45', 'L50', 'L51', 'L52', 'L53', 'L54', 'L55', 'L56', 'L57', 'L58', 'L59', 'L60', 'L62', 'L63', 'L64', 'L65', 'L66', 'L67', 'L68', 'L70', 'L71', 'L72', 'L73', 'L74', 'L75', 'L80', 'L81', 'L82', 'L83', 'L84', 'L85', 'L86', 'L87', 'L88', 'L89', 'L90', 'L91', 'L92', 'L93', 'L94', 'L95', 'L97', 'L98', 'L99', 'M00', 'M01', 'M02', 'M03', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25', 'M30', 'M31', 'M32', 'M33', 'M34', 'M35', 'M36', 'M40', 'M41', 'M42', 'M43', 'M45', 'M46', 'M47', 'M48', 'M49', 'M50', 'M51', 'M53', 'M54', 'M60', 'M61', 'M62', 'M63', 'M65', 'M66', 'M67', 'M68', 'M70', 'M71', 'M72', 'M73', 'M75', 'M76', 'M77', 'M79', 'M80', 'M81', 'M82', 'M83', 'M84', 'M85', 'M86', 'M87', 'M88', 'M89', 'M90', 'M91', 'M92', 'M93', 'M94', 'M95', 'M96', 'M99', 'N00', 'N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'N21', 'N22', 'N23', 'N25', 'N26', 'N27', 'N28', 'N29', 'N30', 'N31', 'N32', 'N33', 'N34', 'N35', 'N36', 'N37', 'N39', 'N40', 'N41', 'N42', 'N43', 'N44', 'N45', 'N46', 'N47', 'N48', 'N49', 'N50', 'N51', 'N60', 'N61', 'N62', 'N63', 'N64', 'N70', 'N71', 'N72', 'N73', 'N74', 'N75', 'N76', 'N77', 'N80', 'N81', 'N82', 'N83', 'N84', 'N85', 'N86', 'N87', 'N88', 'N89', 'N90', 'N91', 'N92', 'N93', 'N94', 'N95', 'N96', 'N97', 'N98', 'N99', 'O00', 'O01', 'O02', 'O03', 'O04', 'O05', 'O06', 'O07', 'O08', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15', 'O16', 'O20', 'O21', 'O22', 'O23', 'O24', 'O25', 'O26', 'O28', 'O29', 'O30', 'O31', 'O32', 'O33', 'O34', 'O35', 'O36', 'O40', 'O41', 'O42', 'O43', 'O44', 'O45', 'O46', 'O47', 'O48', 'O60', 'O61', 'O62', 'O63', 'O64', 'O65', 'O66', 'O67', 'O68', 'O69', 'O70', 'O71', 'O72', 'O73', 'O74', 'O75', 'O80', 'O81', 'O82', 'O83', 'O84', 'O85', 'O86', 'O87', 'O88', 'O89', 'O90', 'O91', 'O92', 'O94', 'O95', 'O96', 'O97', 'O98', 'O99', 'P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P07', 'P08', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P35', 'P36', 'P37', 'P38', 'P39', 'P50', 'P51', 'P52', 'P53', 'P54', 'P55', 'P56', 'P57', 'P58', 'P59', 'P60', 'P61', 'P70', 'P71', 'P72', 'P74', 'P75', 'P76', 'P77', 'P78', 'P80', 'P81', 'P83', 'P90', 'P91', 'P92', 'P93', 'P94', 'P95', 'P96', 'Q00', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q30', 'Q31', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43', 'Q44', 'Q45', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54', 'Q55', 'Q56', 'Q60', 'Q61', 'Q62', 'Q63', 'Q64', 'Q65', 'Q66', 'Q67', 'Q68', 'Q69', 'Q70', 'Q71', 'Q72', 'Q73', 'Q74', 'Q75', 'Q76', 'Q77', 'Q78', 'Q79', 'Q80', 'Q81', 'Q82', 'Q83', 'Q84', 'Q85', 'Q86', 'Q87', 'Q89', 'Q90', 'Q91', 'Q92', 'Q93', 'Q95', 'Q96', 'Q97', 'Q98', 'Q99']
disease_idx_full = [f"{idx}-0.0" for idx in disease_idx] + ['eid']
disease_name_full = [f"{idx}Date" for idx in disease_name] + ['eid']
df_cols = set(df.columns)
present_cols  = [c for c in disease_idx_full if c in df_cols]
present_names = [n for n, keep in zip(disease_name_full, [c in df_cols for c in disease_idx_full]) if keep]
rename_map = dict(zip(present_cols, present_names))
disease_df = df.filter(items=present_cols).rename(columns=rename_map).copy()
target_cols = [c for c in present_names if c != "eid"]

# Convert date-like columns to YYYYMMDD integers; use 0 for failed parsing or missing values
for c in target_cols:
    if is_date_col(disease_df[c]):
        dt = pd.to_datetime(disease_df[c], errors="coerce")
        disease_df[c] = (pd.to_numeric(dt.dt.strftime("%Y%m%d"), errors="coerce").fillna(0).astype("int64"))
    else:
        disease_df[c] = disease_df[c].replace({pd.NA: 0, None: 0}).fillna(0)

disease_name_present = present_names
na_total = disease_df[target_cols].isna().sum().sum()
print(f"Number of remaining NaN values in disease columns: {na_total}")  # Should be 0

# Save the final dataset (approximately 500k participants)
print('Total sample size:', len(final_df))

final_df = pd.merge(final_df, disease_df, on='eid', how='left')

# Check how many negative numeric entries remain before replacing with NaN
neg_mask = final_df.select_dtypes(include=[np.number]) < 0
print("Total number of negative entries:", neg_mask.sum().sum())

if neg_mask.sum().sum() > 0:
    # Replace remaining negative codes (-1, -3, -7) with NaN
    final_df = final_df.replace([-1, -3, -7], np.nan)

final_df.to_csv(save_reverse_code, index=False)
print(f"Saved final dataset to: {save_reverse_code}")


# Save value counts for selected variables to an Excel file
def value_counts_to_excel(df, columns, excel_path):
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for col in columns:
            if col not in df.columns:
                print(f"⚠️ {col} is not in DataFrame. Skipped.")
                continue

            # Compute value counts (including NaN)
            vc = df[col].value_counts(dropna=False)
            vc_df = vc.reset_index()
            vc_df.columns = ["value", "count"]
            vc_df["proportion"] = vc_df["count"] / len(df)
            vc_df["proportion"] = vc_df["proportion"].round(4)

            # Replace NaN or missing values with 'Missing'
            vc_df["value"] = vc_df["value"].astype(str)
            vc_df.loc[vc_df["value"].isin(["nan", "NaT", "<NA>", "None", "none"]), "value"] = "Missing"

            # Sort by count in descending order
            vc_df = vc_df.sort_values("count", ascending=False)

            # Write each variable’s value counts to a separate sheet
            # (Excel sheet names are limited to 31 characters)
            vc_df.to_excel(writer, sheet_name=col[:31], index=False)

            # Optional: adjust column width for better readability
            worksheet = writer.sheets[col[:31]]
            for i, width in enumerate([20, 10, 15]):  # value, count, proportion
                worksheet.set_column(i, i, width)

    print(f"Finished saving value counts to: {excel_path}")

value_counts_to_excel(final_df, all_var_names, save_reverse_code_count)


##### Save no-NaN rows only
previous_df = final_df.copy()

# Drop all rows with any NaN value
previous_df_no_nan = previous_df.dropna(how="any")

print("=== [Final Clean Dataset: No NaN Rows] ===")
print(f"Original rows: {len(final_df)}")
print(f"Remaining rows after dropna: {len(previous_df_no_nan)}")
print(f"Number of columns: {len(previous_df_no_nan.columns)}")

# Optional: check if truly no NaN remains
n_nans = previous_df_no_nan.isna().sum().sum()
print(f"Total remaining NaN values (should be 0): {n_nans}")

# Save
previous_df_no_nan.to_csv(save_path_without_nan, index=False)
print(f"Saved clean dataset to: {save_path_without_nan}")
