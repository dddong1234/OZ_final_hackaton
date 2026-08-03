# exp-cv-audit-01 CV Robustness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the current 08-spec Logistic Regression OOF is materially optimistic under random Stratified CV by comparing it with a train-only profile-grouped CV.

**Architecture:** Build sample groups from train mutation profiles only: exact binary mutation-profile duplicates are unioned first, then each sample's nearest cosine-profile neighbour is unioned only if its fixed similarity threshold is met. Fit the same fold-train-only 08-spec feature builder and LR in ordinary StratifiedKFold and StratifiedGroupKFold, then record the OOF gap and group diagnostics. The test dataset is never read.

**Tech Stack:** Python 3.12, pandas, SciPy sparse matrices, scikit-learn, matplotlib, existing project-local sparse_fm_runner utilities.

## Global Constraints

- Work only below `experiments/gs/notebooks/exp_model`.
- Read only `data/raw/train.csv`; do not load test.csv.
- All model feature selection, contrast selection, and LR fitting are fold-train only.
- Use fixed `SEED=42`, `StratifiedKFold-5`, and `StratifiedGroupKFold-5`.
- Set the near-profile threshold before execution: `PROFILE_COSINE_THRESHOLD=0.90`.
- Keep `leakage_check=True` and `nan_as_mutation_count=0` in every result row.

---

### Task 1: Profile grouping contract

**Files:**
- Create: `common/test_exp_cv_audit.py`
- Create: `common/exp_cv_audit_runner.py`

**Interfaces:**
- Produces `build_profile_groups(matrix, threshold) -> tuple[np.ndarray, dict]`.
- Consumes a train-only binary CSR mutation matrix.

- [ ] **Step 1: Write the failing test**

```python
matrix = csr_matrix([[1, 0], [1, 0], [0, 1]])
groups, diagnostics = build_profile_groups(matrix, threshold=0.90)
assert groups[0] == groups[1]
assert groups[0] != groups[2]
assert diagnostics['exact_duplicate_rows'] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python experiments/gs/notebooks/exp_model/common/test_exp_cv_audit.py`

Expected: failure because the audit runner does not exist.

- [ ] **Step 3: Implement minimal profile grouping**

```python
def build_profile_groups(matrix, threshold):
    # union exact CSR row signatures; then union top-1 cosine neighbours >= threshold
    return groups, diagnostics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python experiments/gs/notebooks/exp_model/common/test_exp_cv_audit.py`

Expected: `CV audit contracts passed`.

### Task 2: Fixed CV comparison runner

**Files:**
- Create: `common/exp_cv_audit_runner.py`

**Interfaces:**
- Consumes `build_profile_groups`, `sparse_fm_runner.Cache`, and `_matrix`.
- Produces `exp-cv-audit-01_seed42_summary.csv`, `exp-cv-audit-01_seed42_folds.csv`, and `exp-cv-audit-01_seed42_groups.json`.

- [ ] **Step 1: Build two fold iterators**

```python
regular = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
robust = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

- [ ] **Step 2: Fit identical fold-train-only LR in both iterators**

```python
features, _ = base._matrix(cache, train_index, labels[train_index], contrast=True,
                           functional=False, scale_numeric=False)
model = LogisticRegression(solver='lbfgs', C=.07, max_iter=2000,
                           class_weight='balanced', random_state=42)
```

- [ ] **Step 3: Persist audit data**

Record OOF Macro F1, fold Macro F1, feature count, runtime, group diagnostics, leakage and NaN assertions. Do not select any model from this audit.

- [ ] **Step 4: Verify syntax and tests**

Run:

```bash
.venv/bin/python experiments/gs/notebooks/exp_model/common/test_exp_cv_audit.py
python3 -m py_compile experiments/gs/notebooks/exp_model/common/exp_cv_audit_runner.py
```

Expected: both pass.

### Task 3: Reader-facing notebook

**Files:**
- Create: `exp/exp-cv-audit-01.ipynb`

**Interfaces:**
- Consumes the runner CSV/JSON only after execution.
- Produces a two-CV comparison table and Macro F1 bar chart.

- [ ] **Step 1: Add fixed assumptions cell**

Expose seed, threshold, data policy, and the statement that this is a diagnostic rather than a model-selection score.

- [ ] **Step 2: Add sequential runner cell with live tqdm output**

Use a subprocess pipe so a runner failure includes its final traceback lines.

- [ ] **Step 3: Add bounded results cell**

Read the summary and group diagnostics, assert safety fields, then plot regular vs profile-grouped Macro F1.

- [ ] **Step 4: Validate notebook JSON**

Run: `python3 -m json.tool experiments/gs/notebooks/exp_model/exp/exp-cv-audit-01.ipynb >/dev/null`

Expected: exit code 0.

## Self-Review

- The plan has no test-data read path.
- The group threshold and seed are fixed before execution.
- Grouping uses train features only and no labels; labels are used only by the stratified splitter and the fold-local model.
- The grouped CV is a robustness diagnostic and does not replace the existing random CV benchmark.
