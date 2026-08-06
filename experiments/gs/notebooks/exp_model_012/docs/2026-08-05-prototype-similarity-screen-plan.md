# Prototype Similarity Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate fold-train class-prototype mutation similarity as an independent signal for the frozen H0 Selective-EB model.

**Architecture:** Each outer fold fits automatic event vocabulary, IDF weights, class prototypes, and priors solely on its train split. The runner evaluates H0, prototype-only, and a predeclared 0.80 H0 + 0.20 prototype probability blend on the identical validation rows.

**Tech Stack:** Python, pandas, NumPy, SciPy sparse, scikit-learn, matplotlib, pytest, Jupyter.

## Global Constraints

- Work only in `experiments/gs`; do not import team directories.
- Seed42 OOF reads `train.csv` only; do not read `test.csv`.
- No train/test concat, fixed class/gene/mutation lists, or test-fitted transforms.
- Fit vocabulary, IDF, class priors, prototypes, and normalization on outer-fold train only.
- WT, blank and NaN yield no events; assert `nan_as_mutation_count == 0`.
- Preserve H0 parameters and compare matched folds.
- Codex validates unit/smoke tests only; user runs full CV.

---

### Task 1: Train-only prototype primitives

**Files:**
- Create: `experiments/gs/notebooks/exp_model_012/common/prototype_similarity_core.py`
- Create: `experiments/gs/notebooks/exp_model_012/common/test_prototype_similarity_core.py`

**Interfaces:**
- `parse_event_tokens(frame, genes) -> list[list[str]]`
- `fit_train_only_prototype(frame, labels, genes, classes) -> PrototypeArtifacts`
- `predict_prototype(frame, genes, artifacts) -> ndarray`

- [ ] **Step 1: Write failing tests**

```python
def test_nan_wt_and_blank_are_not_event_tokens():
    frame = pd.DataFrame({"G": [np.nan, "WT", "", "R132H"]})
    assert parse_event_tokens(frame, ["G"]) == [[], [], [], ["G__MISSENSE"]]

def test_prototype_probability_is_row_normalized():
    artifacts = fit_train_only_prototype(frame, labels, ["G"], classes)
    np.testing.assert_allclose(predict_prototype(frame, ["G"], artifacts).sum(axis=1), 1.0)
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q experiments/gs/notebooks/exp_model_012/common/test_prototype_similarity_core.py`

Expected: FAIL because the core module is absent.

- [ ] **Step 3: Implement minimal sparse primitives**

Use train-only token document frequency for IDF; generate automatic `gene__functional_type` plus exact-event tokens; compute L2-normalized class centroids and prior-adjusted cosine-softmax probabilities.

- [ ] **Step 4: Verify green**

Run: `pytest -q experiments/gs/notebooks/exp_model_012/common/test_prototype_similarity_core.py`

Expected: PASS.

### Task 2: Checkpointed H0/prototype screen

**Files:**
- Create: `experiments/gs/notebooks/exp_model_012/common/run_h0_prototype_similarity_screen.py`
- Create: `experiments/gs/notebooks/exp_model_012/common/test_h0_prototype_similarity_screen.py`

**Interfaces:**
- `run(run_id: str) -> None`
- `smoke() -> None`
- Outputs summary, fold/class/low-margin CSV, OOF probability CSV, checkpoint NPZ, audit JSON, and matplotlib images under `result/`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_runner_has_no_test_csv_or_team_import():
    source = RUNNER.read_text()
    assert "test.csv" not in source
    assert "experiments/SDH" not in source

def test_smoke_contract_reports_train_only():
    output = subprocess.check_output([sys.executable, str(RUNNER), "--smoke"], text=True)
    assert '"test_read": false' in output
    assert '"nan_as_mutation_count": 0' in output
```

- [ ] **Step 2: Run failing tests**

Run: `pytest -q experiments/gs/notebooks/exp_model_012/common/test_h0_prototype_similarity_screen.py`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement fold-local runner**

For each matched outer fold, call the GS H0 self-contained `fit_fold`, build prototype artifacts only from fold-train, predict validation, form the fixed 0.80/0.20 blend, checkpoint, free matrices, and record leakage/NaN assertions.

- [ ] **Step 4: Verify runner tests and smoke**

Run: `pytest -q experiments/gs/notebooks/exp_model_012/common/test_h0_prototype_similarity_screen.py && python experiments/gs/notebooks/exp_model_012/common/run_h0_prototype_similarity_screen.py --smoke`

Expected: PASS and safe-contract JSON.

### Task 3: Runner notebook and handoff

**Files:**
- Create: `experiments/gs/notebooks/exp_model_012/exp/exp-h0-prototype-similarity-01.ipynb`
- Create: `experiments/gs/notebooks/exp_model_012/docs/2026-08-05-h0-prototype-similarity.md`

- [ ] **Step 1: Write a failing notebook contract test**

```python
def test_notebook_uses_runner_and_tqdm():
    source = "\n".join(c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells)
    assert "run_h0_prototype_similarity_screen.py" in source
    assert "tqdm" in source and "RUN_EXPERIMENT" in source
```

- [ ] **Step 2: Run the failing test**

Run: `pytest -q experiments/gs/notebooks/exp_model_012/common/test_h0_prototype_similarity_screen.py::test_notebook_uses_runner_and_tqdm`

Expected: FAIL because the notebook is absent.

- [ ] **Step 3: Create notebook and document**

Notebook cells cover experiment contract, subprocess+tqdm execution, summary/fold/class/low-margin tables, plots, and automatic promotion decision. The document records fixed settings and promotion conditions: H0 delta >= 0.015, four positive folds, no low-margin collapse, then 3-seed validation.

- [ ] **Step 4: Validate structure without full CV**

Run: `python -c "import nbformat; nbformat.read('experiments/gs/notebooks/exp_model_012/exp/exp-h0-prototype-similarity-01.ipynb', as_version=4)" && pytest -q experiments/gs/notebooks/exp_model_012/common`

Expected: PASS.

## Self-Review

- The plan covers train-only feature fitting, matched-fold comparison, safe checkpointing, outputs, notebook handoff, and no-CV validation.
- No threshold or blend grid search is included.
- The runner API names and file paths are consistent across tasks.

