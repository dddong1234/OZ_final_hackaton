# Safe 3-way Ensemble Implementation Plan

**Goal:** Create a train-only, rule-safe fixed 0.55/0.30/0.15 multinomial LR, OVR LR, LightGBM ensemble screen on the faithful H0 feature contract.

**Scope:** All created files remain under `experiments/gs/notebooks/exp_model_006`. The prior team 3-way code is read-only reference only because it embeds fixed exact-event rules that are prohibited by the current team contract.

## Fixed contract

- Seed42 Stratified 5-fold screen; test is never read.
- H0 feature vocabulary, recurrence, enrichment, and automatic specialist discovery are outer-fold-train-only.
- The new 3-way models use the same sparse fold matrices: multinomial LR 0.55, OVR LR 0.30, base LightGBM 0.15.
- No fixed cancer name, gene name, hotspot, exact-mutation, contrast-pair, or blend-weight search.
- WT, blank, NaN are not events. Result audit records `leakage_check=True` and `nan_as_mutation_count=0`.
- Selective EB gate is intentionally deferred: it is not valid to claim the original EB gate without reproducing its separate train-only EB representation in this safe contract.

## File structure

- `common/safe_3way_ensemble.py`: probability alignment and fixed 3-way blend.
- `common/run_safe_3way_ensemble.py`: fold-local fits, CSV/JSON/PNG outputs, smoke mode.
- `common/test_safe_3way_ensemble.py`: blend and static safety checks.
- `exp/exp-safe-3way-ensemble-01.ipynb`: user-run notebook with tqdm and plots.
- `docs/2026-08-04-safe-3way-ensemble.md`: experiment rationale and decision rule.

## Verification plan

1. Write and run a blend unit test before implementing the blend helper.
2. Compile runner and notebook JSON.
3. Run the dedicated unit tests and the train-only eight-row smoke mode; do not execute full CV.
4. Static-scan runner/core for `test.csv`, `pd.concat`, and fixed-event-rule names.
5. Check the diff only under `experiments/gs/notebooks/exp_model_006`.
