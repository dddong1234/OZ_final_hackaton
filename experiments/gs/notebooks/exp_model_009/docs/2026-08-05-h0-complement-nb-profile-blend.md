# H0 + Complement NB Mutation-profile Blend

## Hypothesis

The accepted H0 uses a discriminative Selective-EB logistic branch and an automatically discovered LGBM specialist. Complement NB estimates how compatible a sample's binary mutated-gene profile is with each class. Its independent probability errors may recover H0 mistakes.

## Fixed experiment

- H0: unchanged Selective-EB LR + automatic LGBM specialist.
- Complement NB: binary mutation matrix, `alpha=1.0`, `norm=True`.
- Final probability: `0.80 × H0 + 0.20 × Complement NB`.
- No parameter or blend-weight search.

## Leakage contract

For every outer fold, H0 vocabulary/EB/specialists and Complement NB vocabulary/model fit only on outer-fold train rows. Validation is transform/predict only. The seed-42 runner reads only `train.csv`; test is not read, concatenated, encoded, scaled, or used for feature selection. WT, blank, and NaN produce zero mutation events.

## Decision

Seed-42 is a screen only. Promote to fixed 42/777/2024 validation only if the blend improves H0 by at least `+0.003` and improves at least four of five folds. Otherwise this axis is rejected without weight or alpha retuning.
