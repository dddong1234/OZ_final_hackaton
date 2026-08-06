# Lightweight Exact-event EB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a train-only, single-LR Exact-event EB model bundle with reproducible CV, product inference, and submission CSV commands.

**Architecture:** `lightweight_exact_eb_core.py` owns parsing, train-fitted feature state, model serialization, and inference. Thin CLI files call that core for training, CV evaluation, product batch inference, or Dacon submission generation. No CLI imports code outside this directory.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy sparse, scikit-learn Logistic Regression, joblib.

## Global Constraints

- Work only under `experiments/gs/notebooks/submission/lightweight_exact_eb`.
- Use `train.csv` only to fit vocabulary, recurrent events, EB statistics, scaling, and LR.
- Do not use fixed cancer, gene, or exact-mutation lists.
- WT, blank, and NaN produce zero events.
- CV uses Stratified 5-fold and seeds `42/777/2024`; test is not read in CV.
- Production inference loads a saved train-fitted bundle and never refits.
- LR is fixed at `solver='lbfgs'`, `C=0.07`, `max_iter=2000`, `class_weight='balanced'`.

---

### Task 1: Core parser, feature state, and bundle contract

**Files:**
- Create: `experiments/gs/notebooks/submission/lightweight_exact_eb/lightweight_exact_eb_core.py`
- Test: `experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py`

**Interfaces:**
- Produces `fit_bundle(train: pd.DataFrame, seed: int) -> LightweightBundle`.
- Produces `predict_proba(bundle: LightweightBundle, frame: pd.DataFrame) -> np.ndarray`.
- Produces `save_bundle(bundle: LightweightBundle, path: Path) -> None` and `load_bundle(path: Path) -> LightweightBundle`.

- [ ] **Step 1: Write failing parser and projection tests**

```python
def test_missing_values_never_create_events():
    assert normalise_cell(np.nan) == ()
    assert normalise_cell('WT') == ()

def test_apply_ignores_test_only_exact_event():
    bundle = fit_bundle(train_frame, seed=42)
    probability = predict_proba(bundle, test_only_event_frame)
    assert probability.shape == (1, 2)
```

- [ ] **Step 2: Run the test module directly and verify it fails before implementation**

```bash
.venv/bin/python -m pytest -q experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py
```

- [ ] **Step 3: Implement the train-only core**

```python
@dataclass
class LightweightBundle:
    classes: np.ndarray
    feature_state: FeatureState
    model: LogisticRegression

def fit_bundle(train: pd.DataFrame, seed: int) -> LightweightBundle:
    # fit parser vocabulary/EB/scaling and LR using train only
    ...

def predict_proba(bundle: LightweightBundle, frame: pd.DataFrame) -> np.ndarray:
    # transform with bundle state only, then call bundle.model.predict_proba
    ...
```

- [ ] **Step 4: Run parser/projection/bundle round-trip tests**

```bash
.venv/bin/python -m pytest -q experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py
```

- [ ] **Step 5: Commit the focused core change**

```bash
git add experiments/gs/notebooks/submission/lightweight_exact_eb/lightweight_exact_eb_core.py experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py
git commit -m "feat: add lightweight exact EB model bundle"
```

### Task 2: Training and product inference CLIs

**Files:**
- Create: `experiments/gs/notebooks/submission/lightweight_exact_eb/train_lightweight_exact_eb.py`
- Create: `experiments/gs/notebooks/submission/lightweight_exact_eb/predict_lightweight_exact_eb.py`
- Modify: `experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py`

**Interfaces:**
- `train_lightweight_exact_eb.py --train-csv /data/train.csv --bundle-out model.joblib` writes a bundle and audit JSON.
- `predict_lightweight_exact_eb.py --bundle model.joblib --input-csv patients.csv --output-csv predictions.csv` writes `ID`, predicted `SUBCLASS`, and per-class probabilities.

- [ ] **Step 1: Write failing CLI smoke tests**

```python
def test_train_then_predict_cli(tmp_path):
    assert run_train_cli(tmp_path) == 0
    assert run_predict_cli(tmp_path) == 0
    assert (tmp_path / 'predictions.csv').exists()
```

- [ ] **Step 2: Run the CLI smoke test and verify it fails**

```bash
.venv/bin/python -m pytest -q experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py::test_train_then_predict_cli
```

- [ ] **Step 3: Implement explicit train and inference commands**

```python
parser.add_argument('--train-csv', type=Path, required=True)
parser.add_argument('--bundle-out', type=Path, required=True)
parser.add_argument('--input-csv', type=Path, required=True)
parser.add_argument('--output-csv', type=Path, required=True)
```

- [ ] **Step 4: Run the train/predict smoke test and verify output schema**

```bash
.venv/bin/python -m pytest -q experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py::test_train_then_predict_cli
```

- [ ] **Step 5: Commit the CLI change**

```bash
git add experiments/gs/notebooks/submission/lightweight_exact_eb/train_lightweight_exact_eb.py experiments/gs/notebooks/submission/lightweight_exact_eb/predict_lightweight_exact_eb.py experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py
git commit -m "feat: add lightweight training and inference commands"
```

### Task 3: CV evaluation and Dacon submission CLIs

**Files:**
- Create: `experiments/gs/notebooks/submission/lightweight_exact_eb/evaluate_lightweight_exact_eb_cv.py`
- Create: `experiments/gs/notebooks/submission/lightweight_exact_eb/generate_lightweight_exact_eb_submission.py`
- Modify: `experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py`

**Interfaces:**
- `evaluate_lightweight_exact_eb_cv.py --train-csv /data/train.csv --result-dir results` writes seed/fold/OOF metric CSVs.
- `generate_lightweight_exact_eb_submission.py --bundle model.joblib --test-csv /data/test.csv --sample-submission /data/sample_submission.csv --output-csv submission.csv` writes valid Dacon CSV.

- [ ] **Step 1: Write failing output-contract tests**

```python
def test_submission_preserves_sample_ids(tmp_path):
    output = run_submission_cli(tmp_path)
    assert list(output.columns) == ['ID', 'SUBCLASS']
    assert output.ID.tolist() == sample_submission.ID.tolist()
```

- [ ] **Step 2: Run the submission output-contract test and verify it fails**

```bash
.venv/bin/python -m pytest -q experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py::test_submission_preserves_sample_ids
```

- [ ] **Step 3: Implement CV metrics and submission writer**

```python
for seed in (42, 777, 2024):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # fit_bundle only on each fold train; predict only on fold validation
```

- [ ] **Step 4: Run static syntax, output-contract tests, and a train-only CV smoke check**

```bash
.venv/bin/python -m py_compile experiments/gs/notebooks/submission/lightweight_exact_eb/*.py
.venv/bin/python -m pytest -q experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py
```

- [ ] **Step 5: Commit the CV/submission change**

```bash
git add experiments/gs/notebooks/submission/lightweight_exact_eb/evaluate_lightweight_exact_eb_cv.py experiments/gs/notebooks/submission/lightweight_exact_eb/generate_lightweight_exact_eb_submission.py experiments/gs/notebooks/submission/lightweight_exact_eb/test_lightweight_exact_eb_core.py
git commit -m "feat: add lightweight CV and submission commands"
```

### Task 4: Documentation and final verification

**Files:**
- Create: `experiments/gs/notebooks/submission/lightweight_exact_eb/README.md`

- [ ] **Step 1: Document local, `/data`, training, inference, CV, and submission commands**

```markdown
Train: python train_lightweight_exact_eb.py --train-csv /data/train.csv --bundle-out model.joblib
Predict: python predict_lightweight_exact_eb.py --bundle model.joblib --input-csv patients.csv --output-csv predictions.csv
```

- [ ] **Step 2: Verify no external imports or test-time fitting exist**

```bash
rg -n 'sys\.path|concat\(\[.*train.*test|fit\(.*test' experiments/gs/notebooks/submission/lightweight_exact_eb
```

- [ ] **Step 3: Commit documentation**

```bash
git add experiments/gs/notebooks/submission/lightweight_exact_eb/README.md
git commit -m "docs: document lightweight exact EB deployment workflow"
```

## Plan Self-Review

- Coverage: train-only FE/model fit, serialized inference, 3-seed CV, submission CSV, documentation, and tests are covered.
- Placeholders: no unresolved implementation decisions remain; model and safety constants are fixed by the accepted experiment contract.
- Type consistency: all CLI files consume `LightweightBundle` through `fit_bundle`, `load_bundle`, and `predict_proba`.
