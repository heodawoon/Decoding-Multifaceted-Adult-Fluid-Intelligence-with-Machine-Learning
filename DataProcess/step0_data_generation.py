import os
import pandas as pd
import numpy as np

other_csv_path = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB/Real_Final_Version_0928/ukb669045_250926_sohyun_1.csv'
brain_csv_path = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB/Real_Final_Version_0928/ukb669045_250926_sohyun_2.csv'
save_root = '/media/dwh/b9bd8b27-0895-494e-8a2a-2d019ae4bf2c/UKB/Real_Final_Version_0928'

other_df = pd.read_csv(other_csv_path, low_memory=False)
brain_df = pd.read_csv(brain_csv_path, low_memory=False)

##### Eid
eid = other_df['eid']

##### Gender
print('======================= Gender')
gender_tmp = other_df['31-0.0']
mapping = {0:1, 1:0}
gender = gender_tmp.replace(mapping)
del mapping
print(gender.value_counts(dropna=False))

##### Age
print('======================= Age')
birth = pd.to_datetime(other_df['34-0.0'], errors="coerce")   # 생년월일
visit = pd.to_datetime(other_df['53-2.0'], errors="coerce")   # 방문일
# YYYYMMDD 정수형으로 변환 (NaT는 NaN 처리 → astype("Int64")로)
visit_yr_2 = visit.dt.strftime("%Y%m%d").fillna("0").astype(int)
# 단순 연도 차이 (rough age)
age_2 = visit.dt.year - birth.dt.year

# For check
# birth만 NaN, visit은 값 있음
birth_only_nan = birth.isna() & visit.notna()
# visit만 NaN, birth는 값 있음
visit_only_nan = visit.isna() & birth.notna()
print("birth만 NaN:", birth_only_nan.sum())
print("visit만 NaN:", visit_only_nan.sum())

##### Frequency of friend/family visits
print('======================= Frequency of friend/family visits')
freq_visit = other_df['1031-2.0']  # 역코딩하기. -1: do not know, -3: prefer not to answer, 1~7: almost daily ~ no friends
# 1~7은 7~1로 역코딩, -3, -7은 그대로 두기
mapping = {1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1, -1:np.nan, -3:np.nan}
freq_visit_2 = freq_visit.map(mapping)
del mapping
print(freq_visit_2.value_counts(dropna=False))  # Check value

##### Sleep
print('======================= Sleep')
sleep_d = other_df['1160-2.0']
sleep_2 = sleep_d.replace([-1, -3], np.nan)
print(sleep_2.value_counts(dropna=False))  # Check value

##### Alcohol intake
print('======================= Alcohol intake')
alcohol = other_df['1558-2.0'] # 역코딩하기. -3: prefer not to answer, 1 ~ 6: daily or almost daily ~ never
mapping = {1:6, 2:5, 3:4, 4:3, 5:2, 6:1}
alcohol_2 = alcohol.map(mapping)
del mapping
print(alcohol_2.value_counts(dropna=False))  # Check value

##### Fluid
print('======================= Fluid')
fluid_2 = other_df['20016-2.0']
print(fluid_2.value_counts(dropna=False))  # Check value

##### Smoke
print('======================= Smoke')
smoke_status = other_df['20116-2.0']  # 0: never, 1: pervious, 2: current
smoke_status_2 = smoke_status.replace([-3], np.nan)
print(smoke_status_2.value_counts(dropna=False))  # Check value

##### Lone
print('======================= Lone')
lone_ = other_df['2020-2.0']
lone_2 = lone_.replace([-1, -3], np.nan)
print(lone_2.value_counts(dropna=False))  # Check value

##### PHQ2
print('======================= PHQ2')
phq2_2050 = other_df['2050-2.0']
phq2_2050 = phq2_2050.replace([-1, -3], np.nan)
phq2_2060 = other_df['2060-2.0']
phq2_2060 = phq2_2060.replace([-1, -3], np.nan)
phq2_2 = pd.concat([phq2_2050, phq2_2060], axis=1).mean(axis=1, skipna=True)
print(phq2_2.value_counts(dropna=False))  # Check value

# For check
# 둘 다 NaN은 False
both_nan = pd.concat([phq2_2050, phq2_2060], axis=1).isna().all(axis=1)
# 2050만 NaN
nan_2050_only = phq2_2050.isna() & phq2_2060.notna() & ~both_nan
# 2060만 NaN
nan_2060_only = phq2_2060.isna() & phq2_2050.notna() & ~both_nan
print("2050만 NaN 개수:", nan_2050_only.sum())
print("2060만 NaN 개수:", nan_2060_only.sum())

##### Ethnicity
print('======================= Ethnicity')
ethnicity = other_df['21000-0.0']
mapping = {1001:1, 2001:1, 3001:1, 4001:1,
           1002:2, 2002:2, 3002:2, 4002:2,
           1003:3, 2003:3, 3003:3, 4003:3,
           2004:4, 3004:4, 5:3, 6:5}
ethnicity_0 = ethnicity.map(mapping)
del mapping
print(ethnicity_0.value_counts(dropna=False))  # Check value

##### BMI
print('======================= BMI')
bmi_2 = other_df['21001-2.0']

##### Confide
print('======================= Confide')
confide = other_df['2110-2.0']
confide_2 = confide.replace([-1, -3], np.nan)
print(confide_2.value_counts(dropna=False))  # Check value

##### Wears glasses or contact lenses
print('======================= Wears glasses or contact lenses')
glass_lenses = other_df['2207-2.0']  # 1: Yes, 0: No, -3: Prefer not to answer
glass_lenses_2 = glass_lenses.replace([-3], np.nan)
print(glass_lenses_2.value_counts(dropna=False))  # Check value

##### Other eye problems
print('======================= Other eye problems')
eye_issue = other_df['2227-2.0']  # 1: Yes, 0: No, -3: Prefer not to answer
eye_issue_2 = eye_issue.replace([-3], np.nan)
print(eye_issue_2.value_counts(dropna=False))  # Check value

##### Hearing difficulty/problems
print('======================= Hearing difficulty/problems')
hearing_issue = other_df['2247-2.0']  # 1: Yes, 0: No, 99: deaf, -1: do not know, -3: Prefer not to answer
mapping = {0:0, 1:1, 99:2}
hearing_issue_2 = hearing_issue.map(mapping)
del mapping
print(hearing_issue_2.value_counts(dropna=False))  # Check value

##### Hearing difficulty/problems with background noise
print('======================= Hearing difficulty/problems with background noise')
hearing_issue_bg = other_df['2257-2.0']  # 1: Yes, 0: No, -1: do not know, -3: Prefer not to answer
hearing_issue_bg_2 = hearing_issue_bg.replace([-1, -3], np.nan)
print(hearing_issue_bg_2.value_counts(dropna=False))  # Check value

##### Hearing aid user
print('======================= Hearing aid user')
hearing_aid = other_df['3393-2.0']  # 1: Yes, 0: No, -3: Prefer not to answer
hearing_aid_2 = hearing_aid.replace([-3], np.nan)
print(hearing_aid_2.value_counts(dropna=False))  # Check value

##### Health satisfaction
print('======================= Hearing satisfaction')
hlth_sat = other_df['4548-2.0']  # 1 ~ 6: extremly happy ~ extremly unhappy, -1: do not know, -3: prefer not to answer
# 1~6은 6~1로 역코딩, -3, -7은 그대로 두기
mapping = {1:6, 2:5, 3:4, 4:3, 5:2, 6:1}
hlth_sat_2 = hlth_sat.map(mapping)
print(hlth_sat_2.value_counts(dropna=False))  # Check value

##### Family relationship satisfaction
print('======================= Family relationship satisfaction')
fam_sat = other_df['4559-2.0']  # 1 ~ 6: extremly happy ~ extremly unhappy, -1: do not know, -3: prefer not to answer
fam_sat_2 = fam_sat.map(mapping)
print(fam_sat_2.value_counts(dropna=False))  # Check value

##### Friendships satisfaction
print('======================= Friendships satisfaction')
frnd_sat = other_df['4570-2.0']
frnd_sat_2 = frnd_sat.map(mapping)
print(frnd_sat_2.value_counts(dropna=False))  # Check value

##### Financial situation satisfaction
print('======================= Financial situation satisfaction')
fncl_sat = other_df['4581-2.0']
fncl_sat_2 = fncl_sat.map(mapping)
del mapping
print(fncl_sat_2.value_counts(dropna=False))  # Check value

##### Private healthcare
print('======================= Private healthcare')
hthcare = other_df['4674-2.0']  # 1 ~ 4: yes all of the time ~ no never, -1: do not know, -3: prefer not to answer
mapping = {1:4, 2:3, 3:2, 4:1}
hthcare_2 = hthcare.map(mapping)
del mapping
print(hthcare_2.value_counts(dropna=False))  # Check value

##### Qualifications (Education)
print('======================= Qualifications (Education)')
ed = other_df['6138-2.0']
mapping = {1:20, 2:13, 3:10, 4:10, 5:19, 6:15, -7:7}
ed_yr_2 = ed.map(mapping)
del mapping
print(ed_yr_2.value_counts(dropna=False))  # Check value

##### How are people in household related to participant
print('======================= How are people in household related to participant')
marital = other_df['6141-2.0']
mapping = {1:1, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
marital_2 = marital.map(mapping)
del mapping
print(marital_2.value_counts(dropna=False))  # Check value

##### Current employment status
print('======================= Current employment status')
emp = other_df['6142-2.0']  # 1, 2, 3, ..., 7; -7: none of the above, -3: prefer not to answer
emp_2 = emp.replace([-3, -7], np.nan)
print(emp_2.value_counts(dropna=False))  # Check value

##### Leisure/social activities - Categorical
print('======================= Leisure/social activities - Categorical')
soc_cols = ['6160-2.0','6160-2.1','6160-2.2','6160-2.3','6160-2.4']
S = other_df[soc_cols]
S = S.replace([-3, -7], np.nan)

valid_vals = [1,2,3,4,5]
valid_mask = S.isin(valid_vals)            # (n,5) 각각이 1~5인지
has_valid  = valid_mask.any(axis=1)        # 행 단위로 1~5 중 하나라도 존재
all_nan    = S.isna().all(axis=1)          # 행 단위로 전부 NaN
# 결과 프레임 생성 (길이 유지)
soc_out = pd.DataFrame(index=other_df.index)
# social_act_cat1..5 생성
for k in valid_vals:
    colname = f"social_act_2_cat{k}"
    soc_out[colname] = S.eq(k).any(axis=1).astype(int)
    soc_out.loc[all_nan, colname] = np.nan
# 각 값에 대한 변수로 저장
for k in valid_vals:
    varname = f"social_act_2_cat{k}"
    globals()[varname] = soc_out[varname]
# # (선택) 확인용: 어떤 행이 어떤 규칙에 걸렸는지 마커
# out["_rule_all_nan"] = all_nan.astype(int)
# out["_rule_neg_only"] = has_neg_only.astype(int)
# out["_rule_has_valid"] = has_valid.astype(int)
# out["_neg_code_row"] = neg_code_row
print(social_act_2_cat1.value_counts(dropna=False))  # Check value
print(social_act_2_cat2.value_counts(dropna=False))  # Check value
print(social_act_2_cat3.value_counts(dropna=False))  # Check value
print(social_act_2_cat4.value_counts(dropna=False))  # Check value
print(social_act_2_cat5.value_counts(dropna=False))  # Check value

##### Leisure/social activities - Continuous
print('======================= Leisure/social activities - Continuous')
# 1) 기본: 1~5의 개수(각 행에서 유효값의 개수)
count_valid = valid_mask.sum(axis=1).astype('float')  # float로 두어 NaN 대입 용이
# 2) 전부 NaN인 행은 NaN
count_valid[all_nan] = np.nan
# 최종 연속형 결과
social_act_n_2 = count_valid
print(social_act_n_2.value_counts(dropna=False))  # Check value

##### Number in household
print('======================= Number in household')
N_fam = other_df['709-2.0']
N_fam_2 = N_fam.replace([-1, -3], np.nan)
print(N_fam_2.value_counts(dropna=False))  # Check value

##### Average total household income before tax
print('======================= Average total household income before tax')
income_fam = other_df['738-2.0']  # 1, 2, 3, 4, 5; -1: Do not know, -3: prefer not to answer
income_fam_2 = income_fam.replace([-1, -3], np.nan)
print(income_fam_2.value_counts(dropna=False))  # Check value

### MET
print('======================= MET')
def compute_met_no_row_drop(df, weight_col=None, exclude_extreme_outliers=True):
    """
    - 행을 절대 삭제하지 않음 (N 유지)
    - <10분은 0 처리(IPAQ), 결측은 NaN 유지
    - 극단치(하루 walk+mod+vig 총 분 >960)는 결과(MET들)만 NaN으로 마스킹
    - MET_total: 세 파트가 모두 NaN이면 NaN
    """
    day_cols = ['864-2.0', '884-2.0', '904-2.0']  # walk/mod/vig days per week
    dur_cols = ['874-2.0', '894-2.0', '914-2.0']  # walk/mod/vig minutes per day
    cols = day_cols + dur_cols
    out = df[cols].copy()

    # 1) 10분 미만은 0 (분/일 컬럼에만 적용)
    for dur_col in dur_cols:
        out[dur_col] = np.where(out[dur_col].ge(10), out[dur_col],
                                np.where(out[dur_col].isna(), np.nan, 0.0))

    # 2) IPAQ truncation: 요일/분수 클리핑
    # 요일: 0~7
    for dcol in day_cols:
        out[dcol] = out[dcol].clip(lower=0, upper=7)

    # 분/일: 0~180
    for tcol in dur_cols:
        out[tcol] = out[tcol].clip(lower=0, upper=180)

    # 3) 활동별 MET 계산
    WALK_MET, MOD_MET, VIG_MET = 3.3, 4.0, 8.0
    out["MET_walk"] = WALK_MET * out['874-2.0'] * out['864-2.0']  # minutes/day * days/week
    out["MET_mod"]  = MOD_MET  * out['894-2.0'] * out['884-2.0']
    out["MET_vig"]  = VIG_MET  * out['914-2.0'] * out['904-2.0']

    # 4) MET_total (세 파트 중 하나라도 있으면 합)
    out["MET_total"] = out[["MET_walk","MET_mod","MET_vig"]].sum(axis=1, skipna=True, min_count=1)

    # 5) 극단치 마스킹: 하루 총 활동 분 > 960 인 행은 결과만 NaN으로
    if exclude_extreme_outliers:
        total_daily_mins = out['874-2.0'].fillna(0) + out['894-2.0'].fillna(0) + out['914-2.0'].fillna(0)
        mask_extreme = total_daily_mins > 960
        out.loc[mask_extreme, ["MET_walk","MET_mod","MET_vig","MET_total"]] = np.nan

    # # 6) (옵션) kcal 변환
    # if weight_col and weight_col in df.columns:
    #     factor = df[weight_col] / 60.0  # kcal = MET*hours*weight(kg); MET-min × (kg/60)
    #     out["kcal_walk"]  = out["MET_walk"] * factor
    #     out["kcal_mod"]   = out["MET_mod"]  * factor
    #     out["kcal_vig"]   = out["MET_vig"]  * factor
    #     out["kcal_total"] = out["MET_total"] * factor

    return out

act_df = pd.DataFrame({
    "864-2.0": other_df['864-2.0'],
    "874-2.0":  other_df['874-2.0'],
    "884-2.0":  other_df['884-2.0'],
    "894-2.0":   other_df['894-2.0'],
    "904-2.0":  other_df['904-2.0'],
    "914-2.0":   other_df['914-2.0']
})

met_df = compute_met_no_row_drop(act_df, weight_col=None)
met_2 = met_df["MET_total"]

# walk_week = other_df['864-2.0']  # In a typical WEEK, on how many days did you walk for at least 10 minutes at a time? (Include walking that you do at work, travelling to and from work, and for sport or leisure)
# walk_dur = other_df['874-2.0']  # How many minutes did you usually spend walking on a typical DAY?
# mod_week = other_df['884-2.0']  # In a typical WEEK, on how many days did you do 10 minutes or more of moderate physical activities like carrying light loads, cycling at normal pace? (Do not include walking)
# mod_dur = other_df['894-2.0']  # How many minutes did you usually spend doing moderate activities on a typical DAY?
# vig_week = other_df['904-2.0']  # In a typical WEEK, how many days did you do 10 minutes or more of vigorous physical activity? (These are activities that make you sweat or breathe hard such as fast cycling, aerobics, heavy lifting)
# vig_dur = other_df['914-2.0']  # How many minutes did you usually spend doing vigorous activities on a typical DAY?


### Brain
print('======================= Brain')
left_index = np.arange(26755, 26788+1)
right_index = np.arange(26856, 26889+1)

## Brain
# 문자열로 변환하면서 '-2.0' 붙이기
left_cols = [f"{str(idx)}-2.0" for idx in left_index]
right_cols = [f"{str(idx)}-2.0" for idx in right_index]

# other_df에서 해당 컬럼만 추출해서 각각 DataFrame 생성
left_h = brain_df[["eid"] + left_cols].copy()
right_h = brain_df[["eid"] + right_cols].copy()
brain_hemisphere = pd.merge(left_h, right_h, on="eid", how="left")

##### Concatenate into the single dataframe
all_vars = [eid, visit_yr_2, gender, age_2, ethnicity_0, marital_2,  # Demographic
            sleep_2, alcohol_2,smoke_status_2, bmi_2, phq2_2, hlth_sat_2,  # Health
            freq_visit_2, social_act_n_2, social_act_2_cat1, social_act_2_cat2,
            social_act_2_cat3, social_act_2_cat4, social_act_2_cat5, lone_2, confide_2,
            fam_sat_2, frnd_sat_2, N_fam_2,  # Network
            met_2,  # Physical
            fncl_sat_2, emp_2, hthcare_2, ed_yr_2, income_fam_2,  # Socioeconomic
            glass_lenses_2, eye_issue_2, # Visual
            hearing_issue_2, hearing_issue_bg_2, hearing_aid_2,  # Auditory
            fluid_2]  # Cognitive

all_var_names = ["eid", "visit_yr_2", "gender", "age_2", "ethnicity_0", "marital_2",  # Demographic
            "sleep_2", "alcohol_2","smoke_status_2", "bmi_2", "phq2_2", "hlth_sat_2",  # Health
            "freq_visit_2", "social_act_n_2", "social_act_2_cat1", "social_act_2_cat2",
            "social_act_2_cat3", "social_act_2_cat4", "social_act_2_cat5", "lone_2", "confide_2",
            "fam_sat_2", "frnd_sat_2", "N_fam_2",  # Network
            "met_2",  # Physical
            "fncl_sat_2", "emp_2", "hthcare_2", "ed_yr_2", "income_fam_2",  # Socioeconomic
            "glass_lenses_2", "eye_issue_2", # Visual
            "hearing_issue_2", "hearing_issue_bg_2", "hearing_aid_2",  # Auditory
            "fluid_2"]

# DataFrame 합치기
final_df = pd.concat(all_vars, axis=1)
final_df.columns = all_var_names

final_df = final_df.set_index("eid").reindex(other_df["eid"]).reset_index()
print(final_df.shape)
print(final_df.head())
final_df = final_df.merge(brain_hemisphere, on="eid", how="left")

print('After Brain Cat.: ', final_df.shape)
print('After Brain Cat.: ', final_df.head())

# Save the overall data: 50만명
# Total data size
print('Total Data Size: ', len(final_df))
final_df.to_csv(os.path.join(save_root, 'step0_1_ukb669045_251001_final_sorted_data_all.csv'), index=False)


# Save the Only brain imaging data: 5만명
# 53-2.0 (Imaging visit) 값이 NaN이 아닌 행만 선택
mask = ~other_df['53-2.0'].isna()
# final_df에서 해당 행만 추출 → final_df2: Only brain imaging data
final_brain_df = final_df.loc[mask].copy()
final_brain_df.to_csv(os.path.join(save_root, 'step0_2_ukb669045_251001_only_imaging.csv'), index=False)
# Total data size: Only Brain
print('Total Data Size (Brain-based): ', len(final_brain_df))
print(final_brain_df.shape)
print(final_brain_df.head())

# To save the value count in Brain case
def value_counts_to_excel(df, columns, excel_path):
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for col in columns:
            if col not in df.columns:
                print(f"⚠️ {col} is not in DataFrame. Skipped.")
                continue

            vc = df[col].value_counts(dropna=False)
            vc_df = vc.reset_index()
            vc_df.columns = ["value", "count"]
            vc_df["proportion"] = vc_df["count"] / len(df)

            # NaN 처리 → "Missing"
            vc_df["value"] = vc_df["value"].astype(str)
            vc_df.loc[vc_df["value"].isin(["nan", "NaT", "<NA>"]), "value"] = "Missing"

            # 정렬 (count desc)
            vc_df = vc_df.sort_values("count", ascending=False)

            # 시트 이름은 변수명
            vc_df.to_excel(writer, sheet_name=col[:31], index=False)  # Excel 시트명은 31자 제한 있음

    print(f"✅ Finish saving: {excel_path}")

value_counts_to_excel(final_brain_df, all_var_names, os.path.join(save_root, "step0_2_ukb669045_251001_only_imaging_variable_value_counts.xlsx"))
