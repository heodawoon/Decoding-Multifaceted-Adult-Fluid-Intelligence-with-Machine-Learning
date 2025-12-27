"""
Recode disease timing relative to the imaging visit date.

Mapping (reference: imaging visit date):
- Occurred before the imaging visit date  -> 1.0
- Occurred on the same imaging date       -> 1.5
- Occurred after the imaging visit date   -> 2.0
- No disease occurrence                   -> 0.0

Disease first-occurrence dates are taken from UK Biobank field 1712.
Each disease column is expected to be named '{ICDCode}Date' and to contain dates in the format YYYYMMDD.
Note that this sub-variable naming ('{ICDCode}Date') does not follow the original UK Biobank convention;
it was redefined here for convenience.
Input
------
- Step1_3_ukb669045_variable_recoding_and_renaming_value_without_nan_rows.csv

Outputs
-------
1) Redefined disease timing (date → 1.0, 1.5, 2.0, 0.0):
   - Step2_1_ukb669045_disease_timing_redefined.csv

2) `Disease count per disease` (before or at imaging):
   - Step2_2_ukb669045_disease_subject_count_before_or_at_imaging.xlsx

"""

import os
import numpy as np
import pandas as pd

# ---------- Config ----------
root_path = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB_Final/Final_Version_for_Git/Step1'
save_root = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB_Final/Final_Version_for_Git/Step2'
os.makedirs(save_root, exist_ok=True)

disease_list = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A30', 'A31', 'A32', 'A33', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A46', 'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A58', 'A59', 'A60', 'A63', 'A64', 'A66', 'A67', 'A68', 'A69', 'A70', 'A71', 'A74', 'A75', 'A77', 'A78', 'A79', 'A80', 'A81', 'A82', 'A83', 'A84', 'A85', 'A86', 'A87', 'A88', 'A89', 'A90', 'A91', 'A92', 'A93', 'A94', 'A95', 'A97', 'A98', 'B00', 'B01', 'B02', 'B03', 'B05', 'B06', 'B07', 'B08', 'B09', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B30', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47', 'B48', 'B49', 'B50', 'B51', 'B52', 'B53', 'B54', 'B55', 'B57', 'B58', 'B59', 'B60', 'B65', 'B66', 'B67', 'B68', 'B69', 'B71', 'B73', 'B74', 'B75', 'B76', 'B77', 'B78', 'B79', 'B80', 'B81', 'B82', 'B83', 'B85', 'B86', 'B87', 'B88', 'B89', 'B90', 'B91', 'B94', 'B95', 'B96', 'B97', 'B98', 'B99', 'D50', 'D51', 'D52', 'D53', 'D55', 'D56', 'D57', 'D58', 'D59', 'D60', 'D61', 'D62', 'D63', 'D64', 'D65', 'D66', 'D67', 'D68', 'D69', 'D70', 'D71', 'D72', 'D73', 'D74', 'D75', 'D76', 'D77', 'D80', 'D81', 'D82', 'D83', 'D84', 'D86', 'D89', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E32', 'E34', 'E35', 'E41', 'E43', 'E44', 'E45', 'E46', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E56', 'E58', 'E59', 'E60', 'E61', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E70', 'E71', 'E72', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E83', 'E84', 'E85', 'E86', 'E87', 'E88', 'E89', 'F00', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F09', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F28', 'F29', 'F30', 'F31', 'F32', 'F33', 'F34', 'F38', 'F39', 'F40', 'F41', 'F42', 'F43', 'F44', 'F45', 'F48', 'F50', 'F51', 'F52', 'F53', 'F54', 'F55', 'F59', 'F60', 'F61', 'F62', 'F63', 'F64', 'F65', 'F66', 'F68', 'F69', 'F70', 'F71', 'F72', 'F78', 'F79', 'F80', 'F81', 'F82', 'F83', 'F84', 'F88', 'F89', 'F90', 'F91', 'F92', 'F93', 'F94', 'F95', 'F98', 'F99', 'G00', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'G14', 'G20', 'G21', 'G22', 'G23', 'G24', 'G25', 'G30', 'G31', 'G32', 'G35', 'G36', 'G37', 'G40', 'G41', 'G43', 'G44', 'G45', 'G46', 'G47', 'G50', 'G51', 'G52', 'G53', 'G54', 'G55', 'G56', 'G57', 'G58', 'G59', 'G60', 'G61', 'G62', 'G63', 'G64', 'G70', 'G71', 'G72', 'G73', 'G80', 'G81', 'G82', 'G83', 'G90', 'G91', 'G92', 'G93', 'G94', 'G95', 'G96', 'G97', 'G98', 'G99', 'H00', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H10', 'H11', 'H13', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H25', 'H26', 'H27', 'H28', 'H30', 'H31', 'H32', 'H33', 'H34', 'H35', 'H36', 'H40', 'H42', 'H43', 'H44', 'H45', 'H46', 'H47', 'H48', 'H49', 'H50', 'H51', 'H52', 'H53', 'H54', 'H55', 'H57', 'H58', 'H59', 'H60', 'H61', 'H62', 'H65', 'H66', 'H67', 'H68', 'H69', 'H70', 'H71', 'H72', 'H73', 'H74', 'H75', 'H80', 'H81', 'H82', 'H83', 'H90', 'H91', 'H92', 'H93', 'H94', 'H95', 'I00', 'I01', 'I02', 'I05', 'I06', 'I07', 'I08', 'I09', 'I10', 'I11', 'I12', 'I13', 'I15', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I30', 'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40', 'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'I50', 'I51', 'I52', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'I70', 'I71', 'I72', 'I73', 'I74', 'I77', 'I78', 'I79', 'I80', 'I81', 'I82', 'I83', 'I84', 'I85', 'I86', 'I87', 'I88', 'I89', 'I95', 'I97', 'I98', 'I99', 'J00', 'J01', 'J02', 'J03', 'J04', 'J05', 'J06', 'J09', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J20', 'J21', 'J22', 'J30', 'J31', 'J32', 'J33', 'J34', 'J35', 'J36', 'J37', 'J38', 'J39', 'J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J66', 'J67', 'J68', 'J69', 'J70', 'J80', 'J81', 'J82', 'J84', 'J85', 'J86', 'J90', 'J91', 'J92', 'J93', 'J94', 'J95', 'J96', 'J98', 'J99', 'K00', 'K01', 'K02', 'K03', 'K04', 'K05', 'K06', 'K07', 'K08', 'K09', 'K10', 'K11', 'K12', 'K13', 'K14', 'K20', 'K21', 'K22', 'K23', 'K25', 'K26', 'K27', 'K28', 'K29', 'K30', 'K31', 'K35', 'K36', 'K37', 'K38', 'K40', 'K41', 'K42', 'K43', 'K44', 'K45', 'K46', 'K50', 'K51', 'K52', 'K55', 'K56', 'K57', 'K58', 'K59', 'K60', 'K61', 'K62', 'K63', 'K64', 'K65', 'K66', 'K67', 'K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77', 'K80', 'K81', 'K82', 'K83', 'K85', 'K86', 'K87', 'K90', 'K91', 'K92', 'K93', 'L00', 'L01', 'L02', 'L03', 'L04', 'L05', 'L08', 'L10', 'L11', 'L12', 'L13', 'L14', 'L20', 'L21', 'L22', 'L23', 'L24', 'L25', 'L26', 'L27', 'L28', 'L29', 'L30', 'L40', 'L41', 'L42', 'L43', 'L44', 'L50', 'L51', 'L52', 'L53', 'L54', 'L55', 'L56', 'L57', 'L58', 'L59', 'L60', 'L62', 'L63', 'L64', 'L65', 'L66', 'L67', 'L68', 'L70', 'L71', 'L72', 'L73', 'L74', 'L75', 'L80', 'L81', 'L82', 'L83', 'L84', 'L85', 'L86', 'L87', 'L88', 'L89', 'L90', 'L91', 'L92', 'L93', 'L94', 'L95', 'L97', 'L98', 'L99', 'M00', 'M01', 'M02', 'M03', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25', 'M30', 'M31', 'M32', 'M33', 'M34', 'M35', 'M36', 'M40', 'M41', 'M42', 'M43', 'M45', 'M46', 'M47', 'M48', 'M49', 'M50', 'M51', 'M53', 'M54', 'M60', 'M61', 'M62', 'M63', 'M65', 'M66', 'M67', 'M68', 'M70', 'M71', 'M72', 'M73', 'M75', 'M76', 'M77', 'M79', 'M80', 'M81', 'M82', 'M83', 'M84', 'M85', 'M86', 'M87', 'M88', 'M89', 'M90', 'M91', 'M92', 'M93', 'M94', 'M95', 'M96', 'M99', 'N00', 'N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'N21', 'N22', 'N23', 'N25', 'N26', 'N27', 'N28', 'N29', 'N30', 'N31', 'N32', 'N33', 'N34', 'N35', 'N36', 'N37', 'N39', 'N40', 'N41', 'N42', 'N43', 'N44', 'N45', 'N46', 'N47', 'N48', 'N49', 'N50', 'N51', 'N60', 'N61', 'N62', 'N63', 'N64', 'N70', 'N71', 'N72', 'N73', 'N74', 'N75', 'N76', 'N77', 'N80', 'N81', 'N82', 'N83', 'N84', 'N85', 'N86', 'N87', 'N88', 'N89', 'N90', 'N91', 'N92', 'N93', 'N94', 'N95', 'N96', 'N97', 'N98', 'N99', 'O00', 'O01', 'O02', 'O03', 'O04', 'O05', 'O06', 'O07', 'O08', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15', 'O16', 'O20', 'O21', 'O22', 'O23', 'O24', 'O25', 'O26', 'O28', 'O29', 'O30', 'O31', 'O32', 'O33', 'O34', 'O35', 'O36', 'O40', 'O41', 'O42', 'O43', 'O44', 'O45', 'O46', 'O47', 'O48', 'O60', 'O61', 'O62', 'O63', 'O64', 'O65', 'O66', 'O67', 'O68', 'O69', 'O70', 'O71', 'O72', 'O73', 'O74', 'O75', 'O80', 'O81', 'O82', 'O83', 'O84', 'O85', 'O86', 'O87', 'O88', 'O89', 'O90', 'O91', 'O92', 'O94', 'O96', 'O98', 'O99', 'P00', 'P02', 'P03', 'P04', 'P05', 'P07', 'P08', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P35', 'P36', 'P37', 'P38', 'P39', 'P50', 'P51', 'P52', 'P53', 'P54', 'P55', 'P58', 'P59', 'P61', 'P70', 'P71', 'P78', 'P83', 'P91', 'P92', 'P94', 'P95', 'P96', 'Q00', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q30', 'Q31', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43', 'Q44', 'Q45', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54', 'Q55', 'Q56', 'Q60', 'Q61', 'Q62', 'Q63', 'Q64', 'Q65', 'Q66', 'Q67', 'Q68', 'Q69', 'Q70', 'Q71', 'Q72', 'Q73', 'Q74', 'Q75', 'Q76', 'Q77', 'Q78', 'Q79', 'Q80', 'Q81', 'Q82', 'Q83', 'Q84', 'Q85', 'Q86', 'Q87', 'Q89', 'Q90', 'Q91', 'Q92', 'Q93', 'Q95', 'Q96', 'Q97', 'Q98', 'Q99']

total_columns = ["eid", "visit_yr_2", "gender", "age_2", "ethnicity_0", "marital_2",        # Demographic
                 "ed_yr_2", "ed_b_2", "emp_2", "income_fam_2", "fncl_sat_2", "hthcare_2",    # Socioeconomic
                 "lone_2", "social_act_n_2", "social_act_2_sport",                          # Network
                 "social_act_2_pub", "social_act_2_religious", "social_act_2_education",    # Network
                 "social_act_2_other",                                                      # Network
                 "freq_visit_2", "confide_2", "fam_sat_2", "frnd_sat_2", "N_fam_2",         # Network
                 "smoke_status_2",  "alcohol_2", "glass_lenses_2", "eye_issue_2",           # Health
                 "hearing_issue_2", "hearing_issue_bg_2", "hearing_aid_2",                  # Health
                 "phq2_2", "hlth_sat_2", "sleep_2", "bmi_2",                                # Health
                 "met_2",                                                                   # Physical
                 "fluid_2"]                                                                 # Cognition

brain_index  = np.arange(25056, 25103+1)   # 26755 ~ 26788
brain_cols  = [f"{idx}-2.0" for idx in brain_index]

total_columns = total_columns + brain_cols

csv_path = os.path.join(root_path, 'Step1_3_ukb669045_variable_recoding_and_renaming_value_without_nan_rows.csv')
csv_path_out = os.path.join(save_root, 'Step2_1_ukb669045_disease_timing_redefined.csv')
csv_path_disease_number_out = os.path.join(save_root, 'Step2_2_ukb669045_disease_subject_count_before_or_at_imaging.csv')

# ---------- Load ----------
df = pd.read_csv(csv_path)
df_final = pd.read_csv(csv_path)

# Original total N (before removing placeholder subjects)
n_original = len(df)
print(f"[INFO] Original number of subjects: {n_original}")

# Collect all disease date columns that actually exist
disease_date_cols = [f"{code}Date" for code in disease_list if f"{code}Date" in df.columns]
print(f"[INFO] Number of disease date columns found: {len(disease_date_cols)}")

if len(disease_date_cols) > 0:
    date_array = df[disease_date_cols].to_numpy()

    total_cells = date_array.size
    nonzero_mask = (date_array > 0)
    n_nonzero = nonzero_mask.sum()

    print(f"[INFO] Total cells in disease date matrix: {total_cells}")
    print(f"[INFO] Non-zero cells (any recorded date including placeholders): {n_nonzero}")

# ---------- Recompute visit_date and eid after filtering ----------
eid = df['eid']
visit_date = pd.to_datetime(
    df['visit_yr_2'].replace(0, np.nan),
    format="%Y%m%d",
    errors="coerce"
)

# ---------- Recode disease timing ----------
out = np.zeros((len(df), len(disease_list)), dtype=float)

for j, code in enumerate(disease_list):
    col = f'{code}Date'
    if col not in df.columns:
        continue

    # disease_date is stored as YYYYMMDD integer; 0 means "no event"
    disease_date_int = df[col]

    # Valid events are non-zero dates
    valid = disease_date_int > 0

    disease_date = pd.to_datetime(
        disease_date_int.where(valid, np.nan),  # 0 → NaN
        format="%Y%m%d",
        errors="coerce"
    )

    before = valid & (disease_date < visit_date)
    same   = valid & (disease_date == visit_date)
    after  = valid & (disease_date > visit_date)

    out[before.to_numpy(), j] = 1.0
    out[same.to_numpy(),   j] = 1.5
    out[after.to_numpy(),  j] = 2.0

# ---------- Save ----------
df_out = pd.concat(
    [eid.reset_index(drop=True),
     pd.DataFrame(out, columns=disease_list)],
    axis=1)
# 'eid' column is preserved as-is

df_merged_out = pd.merge(df_final[total_columns], df_out, how='left', on='eid')
df_merged_out.to_csv(csv_path_out, index=False)
print(df_merged_out.columns.tolist())

# ---------- Sanity checks ----------
uniq = np.unique(out)
print("unique values in timing matrix:", uniq)

# Check the presence of disease, excluding those that occurred only after the imaging visit.
# Here we only need a simple present/absent indicator (0/1) for "before or at" the visit.
out_sum_check = np.zeros(out.shape, dtype=int)
out_sum_check[(out > 0) & (out <= 1.5)] = 1
out_sum = out_sum_check.sum(axis=0).astype(np.int32)
print("Number of subjects with disease before or at imaging (per code):")
print(out_sum)

# Disease presence matrix (subject × disease), BEFORE or AT imaging only
out_has = ((out > 0) & (out <= 1.5)).astype(int)

# For each disease, count how many subjects have the disease
n_subj_per_disease = out_has.sum(axis=0).astype(int)

df_out_numb = pd.DataFrame([n_subj_per_disease], columns=disease_list)
df_out_numb.to_csv(csv_path_disease_number_out, index=False)

print('Print disease code list: ')
print(disease_list)
print('Number of subjects who have each disease (in order): ')
print(n_subj_per_disease)

# Check how many subjects have at least one disease
n_subj_with_any_disease = (out_has.sum(axis=1) > 0).sum()
print(f'Number of subjects who have at least one disease: {n_subj_with_any_disease}')
