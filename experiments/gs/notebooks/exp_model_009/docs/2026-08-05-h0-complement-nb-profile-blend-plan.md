# H0 Complement NB Profile Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a train-fitted Complement NB mutation-profile model safely complements the accepted Selective-EB H0 ensemble.

**Architecture:** Each outer fold reuses the GS-only H0 Selective-EB fold implementation. A separate Complement NB is trained on only the fold-train gene-mutation binary matrix, then a predeclared 0.80 H0 / 0.20 NB probability blend is evaluated on the outer validation split. No test file is read in OOF mode.

**Tech Stack:** Python, pandas, scipy sparse, scikit-learn ComplementNB/Logistic Regression, LightGBM (H0 only), matplotlib, pytest.

## Global Constraints

- All created code stays under `experiments/gs/notebooks/exp_model_009`.
- No imports from non-GS team directories; GS H0 implementation is read-only dependency.
- No fixed cancer, gene, or exact-mutation lists.
- Outer/inner train only fits vocabulary, EB, recurrence, scaling, specialist discovery, and NB.
- Seed-42 screen reads `train.csv` only; `test.csv` is never read.
- NaN, WT, and blanks create zero events; `nan_as_mutation_count == 0`.
- Fixed experiment: ComplementNB(alpha=1.0, norm=True), H0=0.80, NB=0.20; no blend search.

### Task 1: Core blend contract

**Files:**
- Create: `common/h0_complement_nb_profile.py`
- Test: `common/test_h0_complement_nb_profile.py`

- [ ] Write failing tests for probability normalization, shape rejection, and fixed 0.80/0.20 blending.
- [ ] Implement `profile_blend(h0_probability, nb_probability)` with shape and normalization assertions.
- [ ] Run the focused pytest file.

### Task 2: Checkpointed fold runner

**Files:**
- Create: `common/run_h0_complement_nb_profile_blend.py`
- Test: `common/test_h0_complement_nb_profile_runner.py`

- [ ] Write failing tests for train-only parser behavior and summary schema.
- [ ] Implement a 5-fold checkpointed runner that calls the GS-only H0 fold fit, separately fits ComplementNB on fold-train mutation binary features, writes OOF/audit/results, and never reads test in OOF mode.
- [ ] Add `--smoke` for train schema, NaN parser contract, split disjointness, output schema, and small binary-matrix NB fit.
- [ ] Run focused pytest and `--smoke` only; do not run full CV.

### Task 3: Reader-facing notebook and handoff documentation

**Files:**
- Create: `common/create_h0_complement_nb_profile_blend_notebook.py`
- Create: `exp/exp-h0-complement-nb-profile-blend-01.ipynb`
- Create: `docs/2026-08-05-h0-complement-nb-profile-blend.md`

- [ ] Generate a notebook with fixed config, rules, `tqdm` subprocess output, result tables, fold/class charts, and automated decision.
- [ ] Validate Python syntax, notebook JSON structure, static train-only contract, focused tests, and smoke execution.

## Validation Record

- Full OOF CV is intentionally not run by the agent.
- The user runs seed 42; only if predeclared screen criteria pass should 42/777/2024 be launched.
