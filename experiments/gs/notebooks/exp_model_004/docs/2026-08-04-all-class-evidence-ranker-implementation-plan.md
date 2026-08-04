# All-class Evidence Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** P1 non-EB/P1+EB의 inner-OOF 후보 증거를 이용해 26-class candidate ranker를 seed 42에서 안전하게 screen하는 실행 노트북을 만든다.

**Architecture:** `candidate_evidence_ranker.py`는 후보 행 생성·확률 정규화·계약 검사를 담당한다. `run_all_class_evidence_ranker.py`는 exp_model_002의 P1/EB 구현을 train-only로 호출하고 outer-fold 안에서 inner OOF 기반 ranker를 학습한다. 노트북은 runner 실행, CSV 확인, Macro F1·Top-k·low-margin 결과 시각화만 담당한다.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy sparse, scikit-learn LogisticRegression, StratifiedKFold, matplotlib, tqdm.

## Global Constraints

- 작업 경로는 `experiments/gs/notebooks/exp_model_004`만 사용한다.
- screen seed는 `42`, outer/inner CV는 Stratified 5-fold다.
- base는 P1 non-EB와 P1+EB LR이며, ranker는 `C=0.07`, `max_iter=2000`, `class_weight='balanced'`, `lbfgs` Logistic Regression 하나다.
- ranker 학습 행은 반드시 outer-train 내부 inner OOF에서만 생성한다.
- test 파일·test 통계·test token/vocabulary·test scaling·제출 생성은 금지한다.
- NaN은 mutation event가 아니며 `nan_as_mutation_count=0`을 기록한다.
- screen 통과 전 LightGBM ranker, threshold 재탐색, 3-seed 확장은 금지한다.

---

### Task 1: 후보 행 변환과 계약 검사

**Files:**
- Create: `common/candidate_evidence_ranker.py`
- Create: `common/test_candidate_evidence_ranker.py`

**Interfaces:**
- Produces: `candidate_matrix(p_non_eb, p_eb, eb_scores, burden, class_count) -> tuple[np.ndarray, np.ndarray]`
- Produces: `candidate_scores_to_probability(positive_score, n_samples, n_classes) -> np.ndarray`
- Produces: `topk_metrics(y, probability, classes) -> dict[str, float]`

- [ ] **Step 1: Write the failing test**

```python
def test_candidate_matrix_has_exactly_one_positive_candidate_per_patient():
    p0=np.array([[.7,.3],[.1,.9]])
    p1=np.array([[.6,.4],[.2,.8]])
    evidence=np.array([[1.,-.5],[.2,.7]])
    x, row_index=candidate_matrix(p0,p1,evidence,np.array([2.,4.]),2)
    assert x.shape[0] == 4
    assert row_index.tolist() == [0,0,1,1]

def test_candidate_scores_are_softmax_normalized_per_patient():
    probability=candidate_scores_to_probability(np.array([2.,0.,1.,1.]),2,2)
    assert np.allclose(probability.sum(axis=1),1.0)
    assert probability.shape == (2,2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_candidate_evidence_ranker.py`

Expected: FAIL because `candidate_evidence_ranker` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def candidate_matrix(p_non_eb,p_eb,eb_scores,burden,class_count):
    row_index=np.repeat(np.arange(len(p_eb)),class_count)
    candidate=np.tile(np.arange(class_count),len(p_eb))
    # candidate probability/logit, evidence, row-wise competition and burden only
    return features,row_index

def candidate_scores_to_probability(positive_score,n_samples,n_classes):
    logits=np.asarray(positive_score).reshape(n_samples,n_classes)
    logits=logits-logits.max(axis=1,keepdims=True)
    probability=np.exp(logits); return probability/probability.sum(axis=1,keepdims=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_candidate_evidence_ranker.py`

Expected: PASS with all candidate matrix and probability tests green.

- [ ] **Step 5: Commit**

```bash
git add experiments/gs/notebooks/exp_model_004/common/candidate_evidence_ranker.py experiments/gs/notebooks/exp_model_004/common/test_candidate_evidence_ranker.py
git commit -m "feat: add candidate evidence ranker utilities"
```

### Task 2: Nested train-only ranker runner

**Files:**
- Create: `common/run_all_class_evidence_ranker.py`
- Modify: `common/candidate_evidence_ranker.py`
- Modify: `common/test_candidate_evidence_ranker.py`

**Interfaces:**
- Consumes: `candidate_matrix`, `candidate_scores_to_probability`, `topk_metrics`.
- Produces: CLI `--seed 42 --run-id exp-all-class-evidence-ranker-01`.
- Produces: `{run_id}_summary.csv`, `{run_id}_fold_metrics.csv`, `{run_id}_class_metrics.csv`, `{run_id}_low_margin_metrics.csv`, `{run_id}_oof_probabilities.csv`, `{run_id}_leakage_audit.json` in `result/`.

- [ ] **Step 1: Write the failing test**

```python
def test_ranker_training_features_are_inner_oof_only():
    audit=build_ranker_audit(outer_train=np.array([0,1,2,3]),
                             inner_prediction_rows=np.array([0,1,2,3]),
                             outer_valid=np.array([4,5]))
    assert audit['ranker_training_rows_are_inner_oof'] is True
    assert audit['outer_validation_used_for_ranker_fit'] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_candidate_evidence_ranker.py`

Expected: FAIL because `build_ranker_audit` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_ranker_audit(outer_train,inner_prediction_rows,outer_valid):
    return {
      'ranker_training_rows_are_inner_oof': set(inner_prediction_rows)==set(outer_train),
      'outer_validation_used_for_ranker_fit': bool(set(inner_prediction_rows)&set(outer_valid)),
    }

# Runner outer fold:
# (a) create P1/EB inner OOF probability/evidence for outer_train
# (b) fit candidate LR on candidate rows of inner OOF only
# (c) fit base P1/EB on whole outer_train and transform outer_validation
# (d) rank outer-validation candidates and softmax to 26 probabilities
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_candidate_evidence_ranker.py`

Expected: PASS and audit flags prove no outer-validation row is used to fit ranker features or labels.

- [ ] **Step 5: Run static runner checks**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m py_compile common/candidate_evidence_ranker.py common/run_all_class_evidence_ranker.py && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python common/run_all_class_evidence_ranker.py --help`

Expected: exit code 0; help contains `--seed` and `--run-id`.

- [ ] **Step 6: Commit**

```bash
git add experiments/gs/notebooks/exp_model_004/common/candidate_evidence_ranker.py experiments/gs/notebooks/exp_model_004/common/run_all_class_evidence_ranker.py experiments/gs/notebooks/exp_model_004/common/test_candidate_evidence_ranker.py
git commit -m "feat: add nested candidate evidence ranker runner"
```

### Task 3: Reproducible notebook and result contract

**Files:**
- Create: `common/create_all_class_evidence_ranker_notebook.py`
- Create: `exp/exp-all-class-evidence-ranker-01.ipynb`
- Modify: `common/test_notebook_contract.py`

**Interfaces:**
- Consumes: `run_all_class_evidence_ranker.py` result CSV/JSON files.
- Produces: a notebook with `RUN_EXPERIMENT=False` default and `tqdm` streamed runner output.

- [ ] **Step 1: Write the failing test**

```python
def test_ranker_notebook_is_safe_and_compilable():
    notebook=EXP/'exp-all-class-evidence-ranker-01.ipynb'
    assert notebook.exists()
    assert 'test.csv' not in notebook.read_text()
    for code_cell in json.loads(notebook.read_text())['cells']:
        if code_cell['cell_type']=='code':
            compile(''.join(code_cell['source']),'ranker-notebook','exec')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_notebook_contract.py`

Expected: FAIL because the ranker notebook is missing.

- [ ] **Step 3: Write minimal implementation**

```python
# Notebook cells must:
# 1. locate runner and set RUN_EXPERIMENT=False
# 2. stream CLI output with tqdm when enabled
# 3. load summary/fold/class/low-margin/OOf/audit outputs
# 4. assert test_read is False, leakage true, nan mutation count 0
# 5. plot Macro F1, fold delta, Top-k and low-margin recovery
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_notebook_contract.py`

Expected: PASS; notebook has valid JSON and every code cell compiles.

- [ ] **Step 5: Run full pre-execution verification**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_candidate_evidence_ranker.py common/test_notebook_contract.py && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m py_compile common/candidate_evidence_ranker.py common/run_all_class_evidence_ranker.py common/create_all_class_evidence_ranker_notebook.py && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python common/run_all_class_evidence_ranker.py --help`

Expected: all tests pass, source compiles, CLI exits 0. Do not run OOF experiment in this verification step.

- [ ] **Step 6: Commit**

```bash
git add experiments/gs/notebooks/exp_model_004/common/create_all_class_evidence_ranker_notebook.py experiments/gs/notebooks/exp_model_004/exp/exp-all-class-evidence-ranker-01.ipynb experiments/gs/notebooks/exp_model_004/common/test_notebook_contract.py
git commit -m "feat: add candidate evidence ranker experiment notebook"
```

## Plan Self-Review

- Spec coverage: candidate representation, nested leakage prevention, result artifacts, screen criteria, no-test contract, and notebook observability map to Tasks 1–3.
- No placeholders: all file names, function interfaces, test commands, and expected outcomes are explicit.
- Type consistency: candidate row builders return NumPy arrays; runner consumes candidate matrices and emits normalized `(n_samples, 26)` probabilities; notebook consumes named CSV/JSON artifacts.
- Scope: this plan covers the Logistic screen only. Ranker LightGBM and 3-seed confirmation are intentionally excluded until the screen evidence exists.
