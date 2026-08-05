# Equal H0 / non-EB 3-seed submission

## Goal

Create one regulation-safe comparison submission from the fixed candidate selected in
`exp-h0-component-complement-audit-01`:

`0.50 * H0 Selective-EB + 0.50 * H0 non-EB`.

Both terms retain the same automatic LGBM specialist.  The only difference is their
LR branch: Selective-EB switches to non-EB below the fixed margin 0.05; non-EB always
uses the structured LR branch.

## Fixed inference contract

For each predeclared seed `42`, `777`, and `2024`:

1. Fit structured features, vocabulary, recurrent-event selection, Empirical-Bayes
   statistics, scaling and automatic specialist pairs on full train only.
2. Apply the fitted transformations to test without fitting any test statistics.
3. Generate the two final probabilities, each with its 0.80 LR / 0.20 specialist
   composition.
4. Average these two probabilities at fixed 0.50 / 0.50.
5. Average the three seed probabilities equally and write the predicted class to the
   sample-submission ID order.

No blend weight, margin threshold, seed, class, gene, or mutation list is tuned.

## Implementation

Add a self-contained submission runner under `experiments/gs/notebooks/submission/`.
It may reuse GS-internal H0 modules, but will not import another team directory.
It writes the CSV and adjacent audit JSON.  The runner validates shape, class order,
probability normalization, sample ID order, no train/test concatenation, and
`nan_as_mutation_count == 0`.

## Verification

Before the user runs the full full-train generation, run only:

- a failing unit test for the equal blend contract;
- the unit test after implementation;
- syntax/import checks;
- a no-test-fit static/smoke check that does not run full CV or full final fitting.

The full submission generation remains user-triggered.
