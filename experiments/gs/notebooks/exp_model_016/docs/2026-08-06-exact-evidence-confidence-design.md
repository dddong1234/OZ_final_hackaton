# Exact Evidence Confidence EB Screen — Design

## Goal

Test whether the accepted exact-event EB replacement branch can use the shape
and reliability of automatic exact-event evidence without changing H0,
manually selecting biology, or retuning the fixed gate/specialist settings.

## Fixed comparison

- Baseline: accepted `exact_event_EB` H0 branch from `exp_model_015`.
- Candidate: same branch plus confidence features from exact-event EB.
- CV: seed 42, stratified outer 5-fold; identical folds.
- LR: `lbfgs`, `C=0.07`, `max_iter=2000`, `class_weight='balanced'`.
- H0 blend remains `0.80 × LR + 0.20 × automatic LGBM specialist`.
- Selective margin remains `0.05`; no threshold or blend search.

## New automatic features

For every class and sample, reduce train-fitted exact-event EB contributions
to: positive/negative/absolute sums, strongest positive/negative evidence,
top-1/top-3 absolute-evidence shares, normalized entropy, and posterior
reliability-weighted evidence. This produces `26 × 9 = 234` dense features.
No gene, class, event, support cutoff, position bin, or biological list is
declared. Top-1/top-3 are generic aggregation statistics, not event selection.

Training rows use inner 5-fold OOF EB states. Outer validation uses an EB state
fitted on its outer train only. Each block is standardized from inner OOF rows.

## Safety

- Screen reads `train.csv` only; test is never opened.
- Vocabulary, EB posterior, reliability and scaling use the current
  outer/inner train rows only.
- WT, blank and NaN yield zero events; `nan_as_mutation_count == 0`.
- No train/test concat, fixed biological identifiers, or non-GS imports.

## Decision

Promote only if seed42 improves exact-event EB by `+0.008` Macro F1 with at
least 4/5 positive folds and no class F1 decrease below `-0.05`. Otherwise
close this axis with no statistic subset or coefficient-grid follow-up.

## Implementation tasks

1. Test contribution-shape/reliability output before implementation.
2. Build a memory-bounded runner, writing summary/fold/class/audit artifacts.
3. Add a runnable notebook with tqdm, result loading and an automatic decision.
4. Run syntax, tests, train-only smoke and diff validation; do not run full CV.
