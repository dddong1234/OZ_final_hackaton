# exp-gs-002-09 LGBM 결측 노이즈 augmentation 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** test를 읽지 않고, fold-train에만 고정 결측 노이즈를 적용한 LightGBM과 원본 LightGBM의 OOF Macro F1을 비교하는 재현 가능한 09 노트북을 만든다.

**Architecture:** 단일 노트북이 경로 탐색·train 검증·G 인코딩·fold별 마스킹·LGBM OOF·결과 저장·시각화를 순서대로 수행한다. 두 후보의 차이는 `apply_mask`뿐이며, test 파일은 코드에 포함하지 않는다.

**Tech Stack:** Python, pandas, numpy, LightGBM, scikit-learn, matplotlib, tqdm, nbformat.

## Global Constraints

- 작업 경로는 `experiments/gs/notebooks/eda_pre_002` 하위만 사용한다.
- `test.csv`를 읽거나 파일 경로를 참조하지 않는다.
- `MASK_RATE=0.001`, `MASK_SEED=42`, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`를 상수로 고정한다.
- WT=0, 변이=1, 마스킹된 fold-train 셀만 `np.nan`으로 둔다.
- validation fold는 원본 데이터 그대로 평가한다.
- 결과는 `result/exp-gs-002-09_*`로 저장하며 제출 파일은 만들지 않는다.

---

### Task 1: 노트북 골격과 입력 안전성 추가

**Files:**
- Create: `experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-09.ipynb`

**Interfaces:**
- Consumes: `data/raw/train.csv`
- Produces: `ROOT`, `RESULT_DIR`, `train`, `genes`, `labels`, `TEST_READ=False`

- [ ] **Step 1: 구조 검증용 실패 테스트 작성**

```python
source = notebook_source(path)
assert "test.csv" not in source
assert "MASK_RATE = 0.001" in source
assert "MASK_SEED = 42" in source
```

- [ ] **Step 2: 실패를 확인**

Run: `python3 -c "..."`

Expected: `exp-gs-002-09.ipynb`가 아직 없어 파일 없음으로 실패.

- [ ] **Step 3: 최소 노트북 작성**

```python
MASK_RATE = 0.001
MASK_SEED = 42
CV_SEED = 42
TEST_READ = False
train = pd.read_csv(DATA_DIR / "train.csv")
assert train[genes].isna().sum().sum() == 0
```

- [ ] **Step 4: 구조 검증 통과 확인**

Run: `python3 -c "..."`

Expected: 금지된 test 경로 없이 상수와 train 결측 검증이 확인됨.

### Task 2: fold-train 전용 마스킹과 OOF 실행 구현

**Files:**
- Modify: `experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-09.ipynb`

**Interfaces:**
- Consumes: `make_gene_matrix(frame, genes) -> np.ndarray`, `mask_fold_train(matrix, train_index, rate, seed, fold) -> (np.ndarray, int)`
- Produces: `run_candidate(candidate_name, apply_mask) -> dict`

- [ ] **Step 1: 마스킹 불변식 실패 테스트 작성**

```python
masked, count = mask_fold_train(matrix, train_index, 0.001, 42, 0)
assert np.array_equal(masked[valid_index], matrix[valid_index], equal_nan=True)
assert count > 0
```

- [ ] **Step 2: 실패를 확인**

Run: `python3 -c "..."`

Expected: `mask_fold_train`이 아직 없어 `NameError` 발생.

- [ ] **Step 3: 마스킹과 후보 실행 구현**

```python
rng = np.random.default_rng(MASK_SEED + fold)
flat = rng.choice(train_index.size * n_genes, size=mask_count, replace=False)
rows = train_index[flat // n_genes]
cols = flat % n_genes
masked[rows, cols] = np.nan
```

각 fold에서 동일 split에 대해 baseline과 augmentation 후보를 실행하고 OOF 확률·예측·Macro F1·accuracy·feature 수·runtime·masked cell 수를 축적한다.

- [ ] **Step 4: 마스킹 불변식 통과 확인**

Run: `python3 -c "..."`

Expected: validation 행에는 `NaN`이 새로 생기지 않고, fold-train 마스킹 셀 수가 양수.

### Task 3: 결과·감사 기록·시각화 추가

**Files:**
- Modify: `experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-09.ipynb`

**Interfaces:**
- Consumes: baseline 및 augmentation 후보 실행 결과
- Produces: `result/exp-gs-002-09_summary.csv`, `result/exp-gs-002-09_class_f1.csv`, `result/exp-gs-002-09_metadata.json`, `result/exp-gs-002-09_oof_macro_f1.png`

- [ ] **Step 1: 결과 산출물 실패 테스트 작성**

```python
assert set(summary["candidate"]) == {"lgbm_native_baseline", "lgbm_mask_noise_0p001"}
assert metadata["test_read"] is False
assert metadata["mask_rate"] == 0.001
assert metadata["validation_masked_cell_count"] == 0
```

- [ ] **Step 2: 실패를 확인**

Run: 노트북의 결과 집계 셀 실행.

Expected: 집계 객체와 저장 파일이 아직 없어 실패.

- [ ] **Step 3: 결과 저장과 그래프 구현**

```python
summary.to_csv(RESULT_DIR / "exp-gs-002-09_summary.csv", index=False)
with open(RESULT_DIR / "exp-gs-002-09_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

두 후보의 OOF Macro F1 막대그래프와 fold별 점수 그래프를 그린다. markdown에 “원본 validation 평가이므로 결측 입력 성능을 직접 추정하지 않음”을 명시한다.

- [ ] **Step 4: 정적 검증 및 노트북 구조 검증**

Run: `python3 -m json.tool experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-09.ipynb >/dev/null`

Expected: JSON 유효. 실제 OOF 실행은 사용자의 로컬 커널에서 수행하며, 실행 전에는 결과 수치를 주장하지 않는다.
