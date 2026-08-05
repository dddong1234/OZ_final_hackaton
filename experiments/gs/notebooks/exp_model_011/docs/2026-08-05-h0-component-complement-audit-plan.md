# H0 Component Complement Audit Implementation Plan

**Goal:** Audit whether already rule-safe H0 probability branches have complementary errors before any new high-upside experiment.

**Scope:** Seed42, same five outer folds, train-only H0 fits. Fixed equal-probability blends only; no weight search, no test read, no submission.

## Fixed variants

- `H0_selective_EB`: current H0 candidate probability.
- `H0_non_EB`: non-EB LR plus the same automatic specialist.
- `H0_EB`: EB LR plus the same automatic specialist.
- `equal_H0_non_EB`: 0.5 H0 + 0.5 non-EB.
- `equal_H0_EB`: 0.5 H0 + 0.5 EB.

## Gate

Only a blend with all of the following becomes a 3-seed candidate: seed42 delta at least +0.003, at least 4/5 positive folds, recovered H0 errors exceed newly broken H0-correct rows, and no low-margin Macro F1 drop below -0.003.

## Tests and verification

1. Unit-test blend normalization and recovery/broken accounting.
2. Run parser/schema smoke test using train only.
3. Compile runner and validate notebook JSON.
4. Static scan confirms no test CSV read and no hard-coded cancer/gene/mutation lists.
