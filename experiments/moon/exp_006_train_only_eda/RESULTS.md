# Results — moon-exp-006

## Run record

- Date: 2026-07-31
- Data scope: `train.csv` only. `test.csv` was not opened.
- Validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Model held fixed for every candidate: `LogisticRegression(solver="lbfgs",
  max_iter=1000, class_weight="balanced", random_state=42)`.
- Primary comparison: pooled out-of-fold (OOF) Macro F1.
- Runtime environment: Python 3.12.8, pandas 3.0.5, scikit-learn 1.9.0.

## Train-only descriptive EDA

| Observation | Result | Feature-design implication |
| --- | ---: | --- |
| Training samples / classes | 6,201 / 26 | Use stratified CV and Macro F1 rather than accuracy alone. |
| Mutation-gene columns | 4,384 | Preserve gene-level non-WT flags; do not discard rare genes globally. |
| Missing cells | 0 | No imputation is needed; missingness is not a model feature. |
| Non-WT gene cells / parsed events | 218,893 / 249,064 | A cell can contain more than one event, so event burden is distinct from mutated-gene burden. |
| Mutated-gene count, median / mean / max | 14 / 35.30 / 2,393 | The distribution is strongly right-skewed; retain row-wise burden counts instead of choosing a test-derived high-burden cutoff. |
| Missense event share | 64.66% | Keep a missense count and learn recurrent exact missense strings only inside folds. |
| Synonymous event share | 26.04% | Keep only its aggregate count; do not create sparse raw-string flags. |
| Nonsense + frameshift share | 9.23% | Keep truncating-like counts and gene flags as a functional-impact proxy. |
| Splice / in-frame indel | 0 / 3 events | Their aggregate columns are fold-wise constants or nearly constant and are automatically dropped if constant. |

The most commonly mutated gene was `TP53` (1,770/6,201; 28.54%).  This is a
frequency observation, not a pathogenicity assertion or a reason to restrict
the feature space to named genes.  The full local EDA tables are deliberately
ignored by Git because they are generated artifacts.

## Fixed-condition feature comparison

| Candidate | Features | OOF accuracy | OOF Macro F1 | Delta vs. WT OOF |
| --- | --- | ---: | ---: | ---: |
| `wt_binary` | WT/non-WT gene flags | 0.34285 | 0.34452 | — |
| `gene_burden` | Gene flags + row-wise event/class burden | 0.38768 | 0.38289 | +0.03837 |
| `functional_recurrent` | Gene/burden + truncating gene flags + fold-learned exact recurrent missense flags | 0.40526 | 0.40813 | +0.06360 |

`functional_recurrent` is the next feature representation to carry forward.
Its number of features varied from 7,885 to 7,967 between folds, which is
expected: active genes, non-constant summaries, and recurrent strings are
learned from each fold's training partition only.

## Interpretation and limitations

- `truncating-like` groups nonsense, frameshift, and splice-like notation
  because each can disrupt the protein product.  It does **not** establish loss
  of function for any individual variant.
- An exact event occurring at least five times in a fold's training split is a
  **recurrent-event proxy**, not an external clinical hotspot label.
- Several LR fits reached the fixed `max_iter=1000` limit.  The score is a fair
  fixed-protocol ablation, but the selected representation must be rechecked
  with an independently specified model/regularisation experiment before it is
  treated as a final modelling choice.
- No raw test-event vocabulary, test distribution, external pathway, cancer-gene
  list, clinical annotation, or other team code was used.
