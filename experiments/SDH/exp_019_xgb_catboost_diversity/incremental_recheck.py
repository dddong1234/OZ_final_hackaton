"""Continue an already-running exp19 kernel with the corrected B10 baseline.

Run from the bottom of ``experiment.ipynb`` without restarting the kernel::

    %run -i "$EXP_DIR/incremental_recheck.py"

The expensive seed-42 XGBoost/CatBoost ``screen_results`` are reused.  This
script trains only the existing LightGBM partner and the inner-fold fits needed
for one strict candidate per family.
"""

from __future__ import annotations

import importlib
import json
import time


required = (
    "exp", "train", "genes", "labels", "CASES", "screen_results",
    "prepared_by_seed", "anchors", "RESULT_DIR",
)
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "experiment.ipynb의 seed42 XGB/Cat 스크린 셀까지 먼저 실행하세요. "
        f"없는 변수: {missing}"
    )
if not screen_results:
    raise RuntimeError("screen_results가 비어 있습니다. 모델 스크린을 먼저 끝내세요.")

# Reloading adds the corrected evaluation functions without deleting OOF objects
# that were already computed by the running notebook kernel.
exp = importlib.reload(exp)

print("[exp19 correction] 판정 기준: LR -> fold-local LR+LGBM")
print("기존 XGB/Cat outer OOF 재사용:", len(screen_results), "cases")

lgbm_anchors = globals().get("lgbm_anchors", {})
if 42 not in lgbm_anchors:
    started = time.perf_counter()
    lgbm_anchors[42] = exp.evaluate_lgbm_anchor(
        prepared_by_seed[42], labels, seed=42
    )
    print(f"existing LGBM outer OOF: {(time.perf_counter() - started) / 60:.1f} min")

fixed_incremental = exp.fixed_incremental_table(
    anchors[42], lgbm_anchors[42], screen_results, labels, CASES
)
fixed_incremental.to_csv(
    RESULT_DIR / "seed42_lr_lgbm_incremental_fixed_screen.csv", index=False
)
display_columns = [
    "case", "family", "view", "single_f1",
    "fixed_baseline_f1", "fixed_augmented_f1", "fixed_incremental_delta",
    "candidate_lgbm_correlation", "candidate_lgbm_disagreement",
    "base_recovered_count", "base_damaged_count", "base_net_correct_count",
]
display(fixed_incremental[display_columns])

# The fixed 80/20 baseline + 10% candidate score is only a cheap ranking tool.
# Strict validation is intentionally limited to one case per family.
STRICT_CASES = []
for family in ("xgb", "cat"):
    family_rows = fixed_incremental[fixed_incremental["family"] == family]
    if not family_rows.empty:
        STRICT_CASES.append(str(family_rows.iloc[0]["case"]))
print("strict fold-local confirmation:", STRICT_CASES)

foldlocal_anchor_cache = globals().get("foldlocal_anchor_cache", {})
if 42 not in foldlocal_anchor_cache:
    started = time.perf_counter()
    foldlocal_anchor_cache[42] = exp.build_foldlocal_anchor_cache(
        train,
        genes,
        prepared_by_seed[42],
        labels,
        seed=42,
        inner_splits=3,
        verbose=True,
    )
    print(f"shared inner LR/LGBM cache: {(time.perf_counter() - started) / 60:.1f} min")

strict_seed42 = []
strict_seed42_errors = {}
for case_name in STRICT_CASES:
    started = time.perf_counter()
    try:
        row = exp.strict_foldlocal_incremental(
            foldlocal_anchor_cache[42],
            labels,
            anchors[42],
            lgbm_anchors[42],
            screen_results[case_name],
            CASES[case_name],
            seed=42,
            verbose=True,
        )
        strict_seed42.append(row)
        print(
            case_name,
            f"strict delta={row['incremental_delta']:+.6f}",
            f"positive folds={row['positive_folds']}/5",
            f"candidate zero-weight folds={row['candidate_zero_weight_folds']}/5",
            f"elapsed={(time.perf_counter() - started) / 60:.1f} min",
        )
    except Exception as error:
        strict_seed42_errors[case_name] = repr(error)
        print("[ERROR]", case_name, repr(error))

strict_seed42_table = pd.DataFrame([
    {key: value for key, value in row.items() if key not in ("choices", "fold_deltas")}
    | {
        "fold_deltas": json.dumps(row["fold_deltas"]),
        "choices": json.dumps(row["choices"], ensure_ascii=False),
    }
    for row in strict_seed42
])
strict_seed42_table.to_csv(
    RESULT_DIR / "seed42_strict_foldlocal_incremental.csv", index=False
)
(RESULT_DIR / "seed42_strict_foldlocal_errors.json").write_text(
    json.dumps(strict_seed42_errors, indent=2, ensure_ascii=False), encoding="utf-8"
)
display(strict_seed42_table)

PROMOTED_CASES = [
    row["case"] for row in strict_seed42
    if row["incremental_delta"] > 0
    and row["positive_folds"] >= 3
    and row["candidate_zero_weight_folds"] < 5
]
print("3-seed strict 확인 승격 후보:", PROMOTED_CASES if PROMOTED_CASES else "없음")
print(
    "주의: fixed_incremental_delta는 순위용이며 채택 근거가 아닙니다. "
    "seed42 strict fold-local incremental_delta만 다음 단계 판단에 사용합니다."
)
