# H0 Selective EB Branch Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compare the faithful H0 ensemble against the same ensemble with only its 80% LR branch replaced by the fixed, train-only Selective Empirical-Bayes gate.

**Architecture:** Every outer fold first creates the unchanged H0 specialist-LGBM probability. Independently, the fold creates P1 non-EB and P1+EB probabilities using only outer-fold training data; the fixed 0.05 margin rule selects the LR-side probability. The candidate is `0.80 * gated_LR + 0.20 * unchanged_H0_specialist`.

**Tech Stack:** Python, pandas, SciPy sparse, scikit-learn, LightGBM, matplotlib, unittest, tqdm.

## Global Constraints

- Write only under `experiments/gs`.
- No test read in OOF mode, no train/test concat, and no test-fitted transform.
- No fixed class, gene, or mutation identifiers; all vocabularies/statistics are fold-train only.
- Use `StratifiedKFold(5, shuffle=True, random_state=seed)`; screen seed is 42.
- Use the already fixed Selective EB margin threshold `0.05`; do not search weights or thresholds.
- H0 remains `0.80 * LR + 0.20 * LGBM hard-specialist`; only the LR-side probability is replaced.
- Every fold writes an atomic checkpoint under `exp_model_007/result`.
- Require `leakage_check=True`, `nan_as_mutation_count=0`, and zero convergence warnings.

### Task 1: Define the fixed branch-replacement contract

**Files:**
- Create: `common/h0_selective_eb_replacement.py`
- Test: `common/test_h0_selective_eb_replacement.py`

- [ ] Write a failing test that verifies `replace_lr_branch(h0_specialist, selective_lr)` returns an exactly normalized `0.80/0.20` probability blend and that `selective_probability` keeps the predeclared 0.05 rule.
- [ ] Implement the fixed constants and pure blend helper.
- [ ] Run `python -m unittest common/test_h0_selective_eb_replacement.py -v` and require PASS.

### Task 2: Implement fold-safe runner with resume checkpoints

**Files:**
- Create: `common/run_h0_selective_eb_replacement.py`
- Test: `common/test_h0_selective_eb_replacement_runner.py`

- [ ] Write failing tests for result-directory resolution, no test read, and candidate/H0 fold-output schema.
- [ ] Implement one-fold-at-a-time H0 + EB/gate fitting, atomic checkpoint persistence, per-fold garbage collection, and final summary/audit outputs.
- [ ] Run the focused tests plus the existing checkpoint tests; require PASS.

### Task 3: Create the reproducible notebook and documentation

**Files:**
- Create: `exp/exp-h0-selective-eb-branch-replacement-01.ipynb`
- Create: `docs/2026-08-05-h0-selective-eb-branch-replacement.md`

- [ ] Generate a notebook with scope/rule cells, smoke-only default, tqdm streamed runner output, CSV loaders, fold and class plots, and automatic screen decision.
- [ ] Keep full CV disabled by default; user changes only `RUN_EXPERIMENT=True`.
- [ ] Validate notebook JSON, runner imports, train schema, parser NaN unit test, and a small smoke test without executing full CV.

### Task 4: Verification

- [ ] Run `python -m py_compile` on each new module.
- [ ] Run all new focused unittests and `git diff --check -- experiments/gs/notebooks/exp_model_007`.
- [ ] Confirm no generated code reads `test.csv` in OOF mode and result paths resolve to `exp_model_007/result`.
