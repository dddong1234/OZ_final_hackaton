# H2-S Evidence-shape Pairwise Implementation Plan

**Goal:** 규정 안전 H0 확률을 유지하면서 fold-train EB evidence-shape를 공유형 pairwise residual ranker로 결합하는 seed42 OOF 실험을 만든다.

**Architecture:** GS 내부 self-contained parser와 H0 feature builder가 outer/inner fold 별로 train-only state를 만든다. inner OOF H0/EB meta-feature로 shared pairwise LR을 cross-fit해 correction strength를 고르고, outer validation에는 재학습한 변환만 적용한다.

**Global constraints:** test 미열람, 고정 암종/유전자/exact mutation 목록 금지, C/Dexact 금지, NaN event 0, train/test 결합 금지, full CV는 사용자가 노트북에서만 실행.

### Task 1: Safe parsing and fold contracts

- Create `common/h2_evidence_shape_core.py` and `common/test_h2_evidence_shape_core.py`.
- Test NaN/WT/blank event exclusion, train-fitted vocabulary projection, and inner/outer split disjointness.
- Implement deterministic parser, canonical event type, sparse row-local matrices, and fold audits.

### Task 2: H0 and evidence-shape transforms

- Create `common/h2_safe_h0.py` and `common/test_h2_safe_h0.py`.
- Reimplement only GS-local, safe G/B/V/T/R/A/S and train-fitted gene-type enrichment blocks; no imports from SDH or other team directories.
- Fit LR and balanced LGBM with fixed H0 settings, derive automatic specialist pairs from outer-train only, and blend `0.80/0.20`.
- Build 19 candidate-wise EB/H0 evidence-shape features from fit rows only.

### Task 3: Shared pairwise residual ranker

- Create `common/h2_pairwise_ranker.py` and `common/test_h2_pairwise_ranker.py`.
- Produce symmetric true-vs-negative candidate difference rows, fit one `C=0.035` LR, and convert 26 candidate corrections into a log-probability residual softmax.
- Cross-fit the ranker within outer train to select only alpha `.10` or `.20`; refit on all outer-train meta rows before outer validation prediction.

### Task 4: Memory-safe runner and notebook

- Create `common/run_h2_evidence_shape_pairwise.py`, `common/create_h2_evidence_shape_notebook.py`, and `exp/exp-h2-evidence-shape-01.ipynb`.
- Write partial fold artifacts, release matrices/models each fold, and emit requested CSV/JSON/PNG outputs.
- Notebook defaults `RUN_EXPERIMENT=False`, streams output with tqdm, and loads/plots saved artifacts.

### Task 5: Verification

- Run unit tests, py_compile/import/CLI checks, schema checks, notebook compile, no-test-read static scan, and a synthetic subset smoke run.
- Do not run the full seed42 CV.
