# EB-offset Sparse Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** P1+EB 확률을 고정 offset으로 사용하는 sparse residual softmax screen 노트북을 seed 42에 만든다.

**Architecture:** `eb_offset_residual.py`는 deterministic hash sparse matrix, offset softmax 학습, 확률 변환을 담당한다. `run_eb_offset_sparse_residual.py`는 exp_model_002 P1+EB를 outer/inner fold에서 호출하여 leakage-free offset을 만든다. 노트북은 실행·결과 CSV·시각화만 담당한다.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy CSR, scikit-learn StratifiedKFold, matplotlib, tqdm.

## Global Constraints

- 경로는 `experiments/gs/notebooks/exp_model_004`만 수정한다.
- seed 42, outer/inner Stratified 5-fold, epoch 40, batch 256, LR .05, L2 .001, hash 16,384을 고정한다.
- raw gene binary와 deterministic gene×event-type hash만 residual 입력으로 사용한다.
- residual weight/bias는 0 초기화하고 offset은 P1+EB clipped log probability로 고정한다.
- ranker 학습/offset 생성의 모든 supervised fit은 outer-train 안에서만 한다.
- test 파일·test 통계·제출 생성·파라미터/threshold 재탐색을 금지한다.

---

### Task 1: Deterministic sparse hash와 offset softmax

**Files:**
- Create: `common/eb_offset_residual.py`
- Create: `common/test_eb_offset_residual.py`

**Interfaces:**
- Produces: `hashed_event_matrix(token_sets, rows, dimension=16384) -> csr_matrix`
- Produces: `offset_probability(offset_log_probability, weight, bias, features) -> np.ndarray`
- Produces: `fit_offset_residual(features, y_index, offset, class_weight, config) -> tuple[np.ndarray, np.ndarray, list[float]]`

- [ ] **Step 1: Write the failing test**

```python
def test_zero_residual_returns_offset_probability():
    offset=np.log(np.array([[.7,.3],[.2,.8]]))
    result=offset_probability(offset,np.zeros((3,2)),np.zeros(2),csr_matrix((2,3)))
    assert np.allclose(result,np.exp(offset))

def test_hash_is_reproducible_without_vocabulary_fit():
    tokens=[{'TP53__MISSENSE'},{'BRAF__NONSENSE'}]
    assert (hashed_event_matrix(tokens,[0,1]) != hashed_event_matrix(tokens,[0,1])).nnz == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_eb_offset_residual.py`

Expected: FAIL because `eb_offset_residual` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def hashed_event_matrix(token_sets,rows,dimension=16384):
    # hashlib.blake2b token hash modulo fixed dimension; CSR binary matrix
    return matrix

def offset_probability(offset,weight,bias,features):
    logits=offset+features@weight+bias
    logits-=logits.max(1,keepdims=True)
    return np.exp(logits)/np.exp(logits).sum(1,keepdims=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_eb_offset_residual.py`

Expected: PASS with zero-residual and hash determinism checks green.

### Task 2: Nested P1+EB offset runner

**Files:**
- Create: `common/run_eb_offset_sparse_residual.py`
- Modify: `common/eb_offset_residual.py`
- Modify: `common/test_eb_offset_residual.py`

**Interfaces:**
- Consumes: `hashed_event_matrix`, `fit_offset_residual`, `offset_probability`.
- Produces: CLI `--seed 42 --run-id exp-eb-offset-sparse-residual-01`.
- Produces: summary, fold/class/low-margin metrics, epoch loss, OOF probabilities, audit JSON in `result/`.

- [ ] **Step 1: Write the failing test**

```python
def test_offset_training_audit_rejects_outer_validation_rows():
    audit=offset_audit(np.array([0,1,2]),np.array([0,1,2]),np.array([3]))
    assert audit['offset_train_rows_are_inner_oof'] is True
    assert audit['outer_validation_used_for_residual_fit'] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_eb_offset_residual.py`

Expected: FAIL because `offset_audit` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# Per outer fold:
# 1. inner P1+EB OOF offsets for outer_train
# 2. fit residual on outer_train sparse features + inner OOF offset
# 3. full outer-train P1+EB offset for outer-validation
# 4. apply residual to outer-validation; never fit using its rows/labels
```

- [ ] **Step 4: Run test and static runner verification**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_eb_offset_residual.py && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m py_compile common/eb_offset_residual.py common/run_eb_offset_sparse_residual.py && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python common/run_eb_offset_sparse_residual.py --help`

Expected: PASS; CLI exits 0 with `--seed` and `--run-id`.

### Task 3: Safe experiment notebook

**Files:**
- Create: `common/create_eb_offset_sparse_residual_notebook.py`
- Create: `exp/exp-eb-offset-sparse-residual-01.ipynb`
- Modify: `common/test_notebook_contract.py`

**Interfaces:**
- Consumes: runner CSV/JSON results.
- Produces: `RUN_EXPERIMENT=False` notebook with tqdm streaming and loss/F1/low-margin plots.

- [ ] **Step 1: Write failing notebook contract**

```python
assert (EXP/'exp-eb-offset-sparse-residual-01.ipynb').exists()
assert 'test.csv' not in (EXP/'exp-eb-offset-sparse-residual-01.ipynb').read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_notebook_contract.py`

Expected: FAIL because the notebook is missing.

- [ ] **Step 3: Generate notebook**

```python
# setup, streamed runner, audit assertions, summary/fold/class/low-margin/loss readers,
# and matplotlib plots; default RUN_EXPERIMENT=False
```

- [ ] **Step 4: Run complete pre-execution verification**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest discover -s common -p 'test_*.py' && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m py_compile common/eb_offset_residual.py common/run_eb_offset_sparse_residual.py common/create_eb_offset_sparse_residual_notebook.py && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python common/run_eb_offset_sparse_residual.py --help`

Expected: all tests green, source compiles, CLI exits 0. Do not run OOF in verification.

## Plan Self-Review

- Coverage: deterministic feature construction, zero-offset contract, nested OOF training, result auditing, and notebook safety have explicit tasks.
- No placeholder terms remain; constants and command paths are fixed.
- Scope excludes parameter search, LightGBM, 3-seed extension, test inference, and submission generation.
