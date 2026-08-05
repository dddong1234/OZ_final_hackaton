# Faithful H0 Candidate Evidence Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a train-only seed42 screen that residual-reranks every cancer candidate using event-level Empirical-Bayes evidence while reproducing the accepted H0 Selective-EB base inside every fold.

**Architecture:** Each outer fold creates the accepted H0 Selective-EB probability. A nested inner 3-fold produces H0 probabilities and candidate-wise evidence-shape features for outer-train rows. A shared symmetric pairwise logistic ranker learns a residual score and applies one inner-selected correction strength to outer validation.

**Tech Stack:** Python, pandas, scipy sparse, scikit-learn LogisticRegression/StratifiedKFold, LightGBM, matplotlib, nbformat.

## Global Constraints

- Work only below `experiments/gs/notebooks/exp_model_008`.
- All vocabulary, EB weights, standardization, meta-features, and correction selection are outer-train only.
- Seed42 OOF must not read `test.csv`; no train/test concat.
- No fixed cancer, gene, or exact-mutation identifiers.
- NaN, WT, and blank produce zero events.
- Base: fixed `0.80 × Selective-EB LR + 0.20 × automatic LGBM specialist`, margin `0.05`.
- Pairwise C=`0.035`; correction candidates are inner-only `{0.10, 0.20}`.
- Screen pass: delta `>= +0.015`, 4/5 positive folds, low-margin delta `>= -0.003`, no one fold/class dominance.

### Task 1: Core primitives

**Files:**
- Create: `common/faithful_h0_ranker_core.py`
- Test: `common/test_faithful_h0_ranker_core.py`

- [ ] Test that shape features are finite, pair directions are symmetric, and corrected probabilities remain normalized.
- [ ] Implement candidate evidence shape, symmetric pair construction, and log-probability residual correction.
- [ ] Run core tests.

### Task 2: Fold-safe runner

**Files:**
- Create: `common/run_faithful_h0_candidate_ranker.py`
- Test: `common/test_faithful_h0_candidate_ranker.py`

- [ ] Test test-input exclusion, inner/outer index separation, and result schema.
- [ ] Create inner OOF H0 Selective-EB probabilities and train-fitted EB shapes; fit shared ranker only on inner OOF rows.
- [ ] Fit final ranker per outer fold; apply residual only to outer validation; checkpoint and collect audit/metrics.
- [ ] Run unit tests and smoke only, never full CV.

### Task 3: Notebook

**Files:**
- Create: `common/create_faithful_h0_candidate_ranker_notebook.py`
- Create: `exp/exp-faithful-h0-candidate-ranker-01.ipynb`
- Create: `docs/2026-08-05-faithful-h0-candidate-ranker.md`

- [ ] Generate notebook with contracts, tqdm, results tables/plots, and automatic decision.
- [ ] Validate notebook JSON, imports, smoke, and output paths.
