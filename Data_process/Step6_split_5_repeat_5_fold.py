import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# ---------- Config ----------
root_path = './data'
csv_path = os.path.join(root_path, 'Step5', 'Step5_refilter_categorical_for_deeplearning.csv')

save_root = os.path.join(root_path, 'Step6')
os.makedirs(save_root, exist_ok=True)

N_PER_CLASS = 2200   # number of subjects sampled per class in each iteration
N_ITER = 5           # number of repeated iterations
N_FOLDS = 5          # number of CV folds
SEED = 2025

json_all_path = os.path.join(save_root, f"Iter_{N_ITER}_Folds_{N_FOLDS}.json")
count_csv_path = os.path.join(save_root, f"Subject_occurrence_count_{N_ITER}iter_{N_FOLDS}fold.csv")

# ---------- Load ----------
df = pd.read_csv(csv_path)
if not {'eid', 'fluid_2_p10'}.issubset(df.columns):
    raise ValueError("Required columns: 'eid', 'fluid_2_p10'.")

df = df[['eid', 'fluid_2_p10']].drop_duplicates('eid').copy()

# ---------- Pool of IDs by class ----------
ids0_all = df.loc[df['fluid_2_p10'] == 0, 'eid'].unique().tolist()
ids1_all = df.loc[df['fluid_2_p10'] == 1, 'eid'].unique().tolist()

rng = np.random.RandomState(SEED)
rng.shuffle(ids0_all)
rng.shuffle(ids1_all)

# Feasibility check: can we cover all subjects at least once?
if len(ids0_all) > N_ITER * N_PER_CLASS or len(ids1_all) > N_ITER * N_PER_CLASS:
    raise ValueError(
        f"Cannot guarantee full coverage: "
        f"class0: {len(ids0_all)} subjects, class1: {len(ids1_all)} subjects, "
        f"but N_ITER * N_PER_CLASS = {N_ITER * N_PER_CLASS}."
    )

# ---------- Helper: coverage-aware sampler ----------
def sample_for_iter_with_coverage(all_ids, covered, n_pick, rnd):
    """
    Sample 'n_pick' unique IDs for this iteration such that:
    - IDs not yet in 'covered' are preferred (to guarantee everyone appears at least once)
    - No duplicates within this iteration
    - 'covered' is updated with the picked IDs
    """
    # IDs that have never been used in any previous iteration
    remaining = [e for e in all_ids if e not in covered]

    if len(remaining) >= n_pick:
        # We can fill this iteration entirely with never-used IDs
        pick = rnd.choice(remaining, size=n_pick, replace=False).tolist()
    else:
        # Use all remaining never-used IDs, then fill the rest from the full pool
        base = remaining                       # all still-uncovered IDs
        need = n_pick - len(base)
        # candidates: everyone except those already in this iteration (avoid duplicates)
        candidates = [e for e in all_ids if e not in set(base)]
        extra = rnd.choice(candidates, size=need, replace=False).tolist()
        pick = base + extra

    # Sanity: no duplicates within this iteration
    assert len(pick) == len(set(pick)), "Duplicates within iteration detected!"
    covered.update(pick)
    return pick

# ---------- Build 5 repeated splits ----------
covered0, covered1 = set(), set()
iterations = []

for it in range(N_ITER):
    # Sample N_PER_CLASS per class for this iteration (coverage-aware)
    pick0 = sample_for_iter_with_coverage(ids0_all, covered0, N_PER_CLASS, rng)
    pick1 = sample_for_iter_with_coverage(ids1_all, covered1, N_PER_CLASS, rng)

    iter_ids = np.array(pick0 + pick1, dtype=object)
    rng.shuffle(iter_ids)

    sub = df[df['eid'].isin(iter_ids)].copy()
    y = sub['fluid_2_p10'].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + it)
    folds = []
    for fold_id, (tr_val_idx, te_idx) in enumerate(skf.split(sub, y), start=1):
        train_valid_df = sub.iloc[tr_val_idx][['eid', 'fluid_2_p10']].copy()
        train_valid_eid = train_valid_df['eid'].to_numpy()
        train_valid_labels = train_valid_df['fluid_2_p10'].to_numpy()

        # ---- Split validation (Stratified 10%) ----
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, valid_idx = next(sss.split(train_valid_eid, train_valid_labels))

        tr_eids = train_valid_eid[train_idx].tolist()
        val_eids = train_valid_eid[valid_idx].tolist()

        te_eids = sub.iloc[te_idx]['eid'].tolist()
        folds.append({
            "fold": fold_id,
            "train_eid": tr_eids,
            "valid_eid": val_eids,
            "test_eid": te_eids  # <- 여기서 valid를 test처럼 쓰는 거 그대로 유지
        })

    iterations.append({
        "iteration": it + 1,
        "class0_count": int((sub['fluid_2_p10'] == 0).sum()),
        "class1_count": int((sub['fluid_2_p10'] == 1).sum()),
        "folds": folds
    })

# 최종적으로 coverage가 잘 됐는지 한번 더 확인 (optional)
assert len(covered0) == len(ids0_all), f"class0: {len(covered0)} covered vs {len(ids0_all)} total"
assert len(covered1) == len(ids1_all), f"class1: {len(covered1)} covered vs {len(ids1_all)} total"

# ---------- Save JSON (기존과 동일) ----------
with open(json_all_path, "w") as f:
    json.dump(
        {
            "meta": {
                "n_iter": N_ITER,
                "n_folds": N_FOLDS,
                "per_class": N_PER_CLASS,
                "seed": SEED
            },
            "iterations": iterations
        },
        f,
        indent=2
    )
print(f"Saved: {json_all_path}")

for item in iterations:
    ipath = os.path.join(save_root, f"iter{item['iteration']:02d}_folds.json")
    with open(ipath, "w") as f:
        json.dump(item, f, indent=2)
    print(f"Saved: {ipath}")

# ---------- Sanity check + subject occurrence counting (validation-based) ----------
with open(json_all_path, 'r') as f:
    ids = json.load(f)

valid_total = []
subject_counter = {}

for iteridx in range(N_ITER):
    iter_wise = []
    print(f"\n===== Iteration {iteridx + 1} =====")
    for foldidx in range(N_FOLDS):
        iteration_fold = ids['iterations'][iteridx]['folds'][foldidx]
        valid_eid = iteration_fold['valid_eid']

        iter_wise.extend(valid_eid)
        valid_total.extend(valid_eid)

        for eid in valid_eid:
            subject_counter[eid] = subject_counter.get(eid, 0) + 1

        uniq_numb = len(np.unique(valid_eid))
        print(
            f"Iter {iteridx + 1} / Fold {foldidx + 1} / "
            f"Subjects: {len(valid_eid)} / Unique in fold: {uniq_numb}"
        )

    uniq_iter = len(np.unique(iter_wise))
    print(f"→ Iter {iteridx + 1} total unique validation subjects: {uniq_iter}")

valid_total_unique = len(np.unique(valid_total))
print(f"\n=== Unique validation subjects across all {N_ITER} iterations: {valid_total_unique}")

count_df = pd.DataFrame(list(subject_counter.items()), columns=["eid", "valid_count"])
count_df.to_csv(count_csv_path, index=False)

print(f"\nSaved validation-based subject occurrence counts to: {count_csv_path}")
print(f"Total subjects tracked: {len(count_df)}")
print(count_df.describe())
