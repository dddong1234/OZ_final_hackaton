# Equal H0 / non-EB 3-seed submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a regulation-safe 3-seed test submission from the fixed `equal_H0_non_EB` candidate.

**Architecture:** A small GS-only runner will import the existing GS H0 submission primitives, build the Selective-EB and non-EB final probabilities for each fixed seed, average them at 0.50/0.50, then average the three seed probabilities. It writes a submission CSV and audit JSON but does not alter existing submission files.

**Tech Stack:** Python 3.12, NumPy, pandas, existing GS sklearn/LightGBM H0 pipeline.

## Global Constraints

- All generated code stays under `experiments/gs/notebooks/submission/`.
- Fixed seeds are exactly `42`, `777`, `2024`; weights are exactly `0.50/0.50` and `1/3` per seed.
- No external-data or test-fitted preprocessing; test is transform-and-predict only.
- No class, gene, or mutation identifier is hard-coded.
- `nan_as_mutation_count == 0` and `leakage_check == True` are required before writing output.
- The user, not the agent, runs full-train submission generation.

---

### Task 1: Test the fixed equal-probability contract

**Files:**
- Create: `experiments/gs/notebooks/submission/test_generate_submission_equal_h0_non_eb.py`
- Create: `experiments/gs/notebooks/submission/generate_submission_equal_h0_non_eb_3seed.py`

**Interfaces:**
- Produces: `equal_probability(left: np.ndarray, right: np.ndarray) -> np.ndarray`
- Produces: `average_seed_probabilities(probabilities: list[np.ndarray]) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from generate_submission_equal_h0_non_eb_3seed import equal_probability


def test_equal_probability_uses_fixed_half_weights_and_normalizes_rows():
    left = np.array([[0.8, 0.2]], dtype=np.float32)
    right = np.array([[0.2, 0.8]], dtype=np.float32)
    actual = equal_probability(left, right)
    np.testing.assert_allclose(actual, [[0.5, 0.5]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/gs/notebooks/submission/test_generate_submission_equal_h0_non_eb.py -v`

Expected: FAIL because `generate_submission_equal_h0_non_eb_3seed` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def equal_probability(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("probability shape mismatch")
    probability = 0.5 * np.asarray(left, dtype=np.float64) + 0.5 * np.asarray(right, dtype=np.float64)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)
```

Also expose an equal seed average helper with shape and normalization validation.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/gs/notebooks/submission/test_generate_submission_equal_h0_non_eb.py -v`

Expected: PASS.

### Task 2: Implement full-train probability generation

**Files:**
- Modify: `experiments/gs/notebooks/submission/generate_submission_equal_h0_non_eb_3seed.py`
- Modify: `experiments/gs/notebooks/submission/test_generate_submission_equal_h0_non_eb.py`

**Interfaces:**
- Consumes: GS `build_design_matrices`, `empirical_bayes_features`, `_hard_specialist`, `_aligned_probability`, and fixed branch helpers.
- Produces: `build_equal_probability(train, test, model_seed) -> tuple[np.ndarray, np.ndarray, dict]`
- Produces: `run_seed_bagged(output_name, seeds=(42, 777, 2024)) -> Path`

- [ ] **Step 1: Write failing safety test**

```python
from generate_submission_equal_h0_non_eb_3seed import run_seed_bagged


def test_rejects_any_seed_contract_other_than_validated_three_seed_tuple():
    try:
        run_seed_bagged(seeds=(42,))
    except ValueError as error:
        assert "exactly" in str(error)
    else:
        raise AssertionError("invalid seed contract was accepted")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/gs/notebooks/submission/test_generate_submission_equal_h0_non_eb.py -v`

Expected: FAIL because `run_seed_bagged` does not exist.

- [ ] **Step 3: Write minimal implementation**

For each seed, fit on full train only:

```python
non_eb = fit_structured_lr(...)
eb = fit_structured_plus_eb_lr(...)
selective = selective_probability(non_eb, eb)
specialist = fit_full_train_automatic_specialist(...)
h0_selective = fixed_branch_replacement(selective, specialist)
h0_non_eb = fixed_branch_replacement(non_eb, specialist)
per_seed = equal_probability(h0_selective, h0_non_eb)
```

Average `per_seed` matrices only after all three seed fits. Validate output IDs against `sample_submission.csv`. Save the CSV as `submission_equal_h0_non_eb_seed42_777_2024_bagged.csv` and save the audit JSON beside it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/gs/notebooks/submission/test_generate_submission_equal_h0_non_eb.py -v`

Expected: PASS.

### Task 3: Verify static safety and user-facing execution

**Files:**
- Modify: `experiments/gs/notebooks/submission/generate_submission_equal_h0_non_eb_3seed.py`
- Modify: `experiments/gs/notebooks/submission/test_generate_submission_equal_h0_non_eb.py`

**Interfaces:**
- Produces: CLI `python generate_submission_equal_h0_non_eb_3seed.py --smoke`.

- [ ] **Step 1: Write failing smoke test**

```python
from generate_submission_equal_h0_non_eb_3seed import smoke


def test_smoke_reports_train_only_contract_without_generating_submission():
    audit = smoke()
    assert audit["test_role"] == "not_read"
    assert audit["nan_as_mutation_count"] == 0
    assert audit["seed_contract"] == [42, 777, 2024]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/gs/notebooks/submission/test_generate_submission_equal_h0_non_eb.py -v`

Expected: FAIL because `smoke` does not exist.

- [ ] **Step 3: Write minimal implementation**

`smoke()` reads only a small train schema subset, checks the train no-NaN contract and the fixed blend helpers, then returns an audit dictionary. The default CLI performs full generation only when not passed `--smoke`.

- [ ] **Step 4: Run final verification**

Run:

```bash
python -m pytest experiments/gs/notebooks/submission/test_generate_submission_equal_h0_non_eb.py -v
python -m py_compile experiments/gs/notebooks/submission/generate_submission_equal_h0_non_eb_3seed.py
python experiments/gs/notebooks/submission/generate_submission_equal_h0_non_eb_3seed.py --smoke
```

Expected: all tests pass, syntax succeeds, and smoke output reports `test_role: not_read`, `nan_as_mutation_count: 0`, and the exact three-seed contract.
