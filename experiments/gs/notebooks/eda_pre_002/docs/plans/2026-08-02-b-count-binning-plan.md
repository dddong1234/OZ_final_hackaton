# exp-gs-002-10 B count 고정 binning 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 08 최종 LR 후보에 B count 고정 bin 12개만 추가한 3-seed OOF 비교 노트북을 만든다.

**Architecture:** 공통 실행기에 선택적 `b_count_binning` 블록을 추가하되 기존 후보의 동작은 바꾸지 않는다. OOF 경로는 train만 읽게 분리하고, 노트북은 baseline과 binning 후보를 각각 42/2024/777로 실행·집계·시각화한다.

**Tech Stack:** Python, pandas, numpy, scipy sparse, scikit-learn, matplotlib, tqdm, subprocess.

## Global Constraints

- 변경 범위는 `experiments/gs/notebooks/eda_pre_002` 하위로 제한한다.
- 08 최종 후보·LR·CV·seed·fold-train 선택 규칙은 변경하지 않는다.
- B bin 경계는 `1,2,3–4,5–7,8+` / `1,2+`로 고정하며 결과 후 바꾸지 않는다.
- OOF 실행 시 train만 읽는다. 제출 분기에서만 평가 데이터 파일을 읽는다.
- 저장 파일은 `result/exp-gs-002-10_*`만 사용한다.

---

### Task 1: 공통 실행기에 B bin 후보와 OOF 데이터 분리 추가

**Files:**
- Modify: `experiments/gs/notebooks/eda_pre_002/common/exp-gs-002-memory-safe.py`

**Interfaces:**
- Produces: `Candidate(..., b_count_binning: bool=False)`
- Produces: candidate `H-AS-LR-exact-confusion-pairs-Apair-log1p-Bbins`
- Produces: OOF CLI path that reads only `train.csv`

- [ ] Add 12 deterministic B one-hot columns after the existing B count block, only when `b_count_binning=True`.
- [ ] Keep the zero-count category implicit as the intercept reference.
- [ ] Move evaluation data loading and validation to the `--submit` branch only.
- [ ] Run parser self-check and compile the runner.

### Task 2: 3-seed comparison notebook 생성

**Files:**
- Create: `experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-10.ipynb`

**Interfaces:**
- Consumes: runner candidate IDs and seed tuple `(42, 2024, 777)`
- Produces: 6 OOF runs, `exp-gs-002-10_seed_summary.csv`, `exp-gs-002-10_class_f1_summary.csv`, `exp-gs-002-10_oof_macro_f1.png`

- [ ] Add Markdown describing the fixed 08 baseline, bin definitions, 3-seed paired-delta decision rule, and test-free OOF execution.
- [ ] Run baseline and B-binning candidate for all three seeds using tqdm.
- [ ] Load only saved OOF CSV/class-F1 CSV results, calculate paired deltas, leakage/NaN/warning checks, then save summaries and a matplotlib comparison.

### Task 3: Static verification

**Files:**
- Verify: runner and notebook JSON

- [ ] Compile the runner and run `--self-check`.
- [ ] Verify the 10 notebook has valid JSON and calls both candidate IDs for all three seeds.
- [ ] Verify the OOF runner path no longer reads evaluation data before entering the submit branch.
