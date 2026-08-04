# H0 Faithful Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** exp013/014의 규정 안전 H0를 `gs` 내부 self-contained 코드로 재현하고 seed42 OOF `0.543679 ± 0.001` 여부를 감사한다.

**Architecture:** parsing·희소 feature·cross-fitted enrichment를 한 pipeline으로 구현하고, 그 출력에 exp014의 balanced LGBM 및 자동 pair hard specialist를 적용한다. runner는 train만 읽어 OOF 결과와 기준 차이만 저장하며, 기준 불일치 시 이후 모델 실행을 차단하는 판정을 남긴다.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy sparse, scikit-learn, LightGBM, matplotlib, tqdm.

## Global Constraints

- 모든 생성·수정 파일은 `experiments/gs/notebooks/exp_model_006`에 둔다.
- exp013/014는 읽기 전용 참고이며 생성 코드에서 import·실행 의존하지 않는다.
- seed42 OOF에서는 `train.csv`만 읽고 test를 열거나 결합하지 않는다.
- vocabulary·recurrent·enrichment·표준화·자동 pair는 outer fold train에서만 fit한다.
- NaN·WT·빈 문자열은 event 0개이며 `nan_as_mutation_count == 0`이다.
- 고정 암종/유전자/exact mutation 목록을 사용하지 않는다.

### Task 1: Faithful feature-contract tests

**Files:**
- Create: `common/test_h0_faithful_pipeline.py`
- Create: `common/h0_faithful_pipeline.py`

- [ ] Write failing tests for train-only vocabulary, missense-only recurrent selection, 5-fold cross-fitted standardized enrichment, NaN event exclusion, and the 3+7+380+8 dense block contract.
- [ ] Run the tests and confirm the module is missing.
- [ ] Implement parser, sparse matrices, exact feature selection, and cross-fitted enrichment without external imports.
- [ ] Re-run tests.

### Task 2: Safe exp014 blend and reproduction runner

**Files:**
- Create: `common/run_h0_faithful_reproduction.py`
- Create: `common/test_h0_faithful_runner.py`

- [ ] Write failing tests for static no-test/no-concat rules, result schema, and automatic baseline gate.
- [ ] Implement balanced multiclass LGBM, two train-discovered cosine pairs, predicted-only hard probability-mass preservation, and fixed LR 0.80/LGBM 0.20 blend.
- [ ] Store summary, fold/feature/audit CSV and JSON. Set `baseline_reproduced=False` outside ±0.001 and mark `block_downstream_experiments=True`.
- [ ] Run unit tests plus `--smoke`, never full CV.

### Task 3: User execution notebook and handoff

**Files:**
- Create: `exp/exp-h0-faithful-reproduction-01.ipynb`
- Create: `README_h0_faithful_reproduction.md`

- [ ] Add Korean purpose, fixed conditions, Dacon/team rules, execution order, tqdm subprocess runner, result tables, feature/fold plots, and automatic gate display.
- [ ] Validate notebook JSON and compile Python files.
- [ ] Run final test suite, smoke test, static guard scan, and diff check.
