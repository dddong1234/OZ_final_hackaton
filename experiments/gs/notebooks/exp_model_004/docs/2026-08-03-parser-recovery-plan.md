# Parser Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` task-by-task.

**Goal:** Recover deterministic mutation grammar from train-only raw strings and test its isolated effect on the fixed P1+Empirical-Bayes Logistic Regression baseline.

**Architecture:** A grammar module first audits and canonicalizes event segments without labels. A runner then evaluates G0, G1, and conditionally G2 with unchanged fold construction, LR parameters, and EB estimation. Notebook cells invoke only the runner and visualize saved CSV artifacts.

**Tech Stack:** Python 3.12, pandas, numpy, scipy sparse, scikit-learn, matplotlib, tqdm.

## Global Constraints

- Work only under `experiments/gs/notebooks/exp_model_004`.
- Read train only; never read test data in audit, fitting, vocabulary, scaling, or selection.
- Preserve WT/empty/NaN as zero events; `nan_as_mutation_count=0`.
- P1+EB comparator: fixed outer Stratified 5-fold, seeds 42/777/2024, LR lbfgs C=.07 max_iter=2000 class_weight=balanced.
- All supervised token weights and normalization are fold-train only; inner cross-fitting is retained.
- G1 uses recovered segmentation but maps canonical types to legacy parent types; G2 alone uses canonical types.

---

### Task 1: Build deterministic grammar and audit contracts

**Files:**
- Create: `common/parser_recovery.py`
- Create: `common/test_parser_recovery.py`

**Interfaces:**
- `parse_cell(gene: str, value: object) -> list[CanonicalEvent]`
- `audit_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]`
- `event_tokens(events, mode: Literal['legacy','canonical']) -> list[set[str]]`

- [ ] Write failing tests for WT/NaN exclusion, semicolon/slash multi-event splitting, no duplicate segments, unknown retention, and stable canonical type mapping.
- [ ] Implement regex grammar and explicit `OTHER_VALID`/`UNKNOWN` output; return each raw non-WT segment exactly once.
- [ ] Save audit tables: raw pattern frequency, delimiter frequency, canonical type frequency, unknown patterns, and contract JSON.
- [ ] Run parser tests.

### Task 2: Implement fixed P1+EB parser comparison

**Files:**
- Create: `common/run_parser_recovery.py`
- Test: `common/test_parser_recovery.py`

**Interfaces:**
- CLI: `--axis audit|g1|g2 --seed INT --run-id TEXT`
- Outputs: seed summary, fold metrics, class metrics, OOF probabilities, grammar audit, leakage audit JSON.

- [ ] Write failing test asserting G1 token mode maps expanded types to legacy parents while G2 retains canonical types.
- [ ] Reuse the project-internal P1+EB baseline and fixed folds; generate G0 baseline probability.
- [ ] Evaluate G1 first with recovered legacy tokenization; run G2 only when requested after G1 screen review.
- [ ] Record fold-train-only, no-test-read, convergence count, and NaN event count in each result artifact.
- [ ] Run unit tests, syntax check, CLI `--help`, and a non-OOF import smoke check.

### Task 3: Create runnable notebooks

**Files:**
- Create: `exp/exp-parser-grammar-audit-01.ipynb`
- Create: `exp/exp-parser-recovery-g1-01.ipynb`
- Create: `exp/exp-parser-recovery-g2-01.ipynb`
- Create: `common/create_notebooks.py`

- [ ] Audit notebook: runs raw grammar audit only, shows coverage and unknown patterns.
- [ ] G1 notebook: fixed seed 42 run with streaming tqdm; compares G0/P1+EB and G1.
- [ ] G2 notebook: defaults `RUN_EXPERIMENT=False` and explains it may run only after G1 passes.
- [ ] Write notebook contract test: valid JSON, compiling code cells, no test data reference, and f-string seed paths.
- [ ] Run contract test.

### Task 4: Handoff criteria

- [ ] G1 screen promotion requires +.010 OOF, 4/5 folds positive, 8+ class F1 improvements, no class F1 decline below -.05, warnings 0, leakage True, NaN events 0.
- [ ] Only G1 promotion permits G2; only 3-seed positive G2/G1 permits the new parser contract as baseline.
