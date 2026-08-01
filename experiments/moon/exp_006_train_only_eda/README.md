# moon-exp-006: Train-only EDA and leak-safe mutation features

## Purpose

This experiment turns the supplied compact protein-mutation strings into
auditable features and tests them under the team's fixed 5-fold Logistic
Regression comparison protocol.  It is a research classification pipeline, not
a clinical diagnostic or treatment-support tool.

## Competition-safety boundary

- Only `train.csv` is opened by `run_train_only_eda.py` and `run_cv.py`.
- `test.csv` is not read for EDA, vocabulary construction, thresholds, feature
  selection, or validation.
- No external gene, pathway, hotspot, oncogene/TSG, clinical, or TCGA resource
  is used.
- Recurrent-event dictionaries and constant-column removal are fitted separately
  inside each training fold; validation folds are transformed only.
- The final submission step is intentionally out of scope for this experiment.

## Biological interpretation in plain language

A gene column records whether a sample has a compact protein-level mutation
string.  The model uses this as a molecular pattern, similar to a list of parts
whose instruction sheets contain changes.  A notation class is an estimate of
how the protein instruction changed, not evidence that the change is pathogenic.

| Notation example | Syntax class | Plain-language interpretation |
| --- | --- | --- |
| `R132H` | missense | One amino-acid letter is replaced. |
| `L21L` | synonymous | The written DNA-level change may not alter the protein letter. |
| `L21*`, `L21X` | nonsense | A stop signal appears early. |
| `P403fs` | frameshift | The later protein reading frame is altered. |
| `Q120_Q121insPA`, `AL34del` | in-frame indel-like | Amino acids are inserted or removed without an explicit `fs` token. |
| unrecognised text | other | Retained only in an aggregate count, not interpreted clinically. |

`truncating-like = nonsense + frameshift + splice-like` is a pragmatic feature
group: all three syntaxes can plausibly shorten or disrupt the normal protein
product.  It is **not** a statement that every such call causes loss of function;
the supplied data have no transcript, genomic coordinate, read depth, VAF, or
clinical annotation with which to establish that.

## Feature candidates

All candidate features are created from one row's supplied strings.

1. `wt_binary` — reference baseline: each gene is `0` for WT/missing and `1`
   for non-WT.
2. `gene_burden` — gene-level non-WT flags plus sample summaries: mutated-gene
   count, event count, multi-event-gene count, and counts for seven notation
   classes.  These are panel-level mutation-burden proxies, not mut/Mb TMB.
3. `functional_recurrent` — candidate 2 plus gene-level truncating-like flags
   and exact recurrent missense-event flags.  An exact event must occur at least
   five times in that fold's training split to receive a separate flag.

Features deliberately excluded from model input:

- an all-raw-event one-hot matrix (too sparse and rare; it would overfit);
- external clinical hotspot/pathway/oncogene/TSG lists (forbidden external data);
- test-derived event vocabulary, quantiles, or distribution checks (leakage);
- missingness counts (audited, but not mutation biology);
- constants in each fold (removed only after that fold is fitted).

## Reproduce

Run from the repository root with Python 3.12.  Substitute the path if the raw
data reside in another approved local clone.

```powershell
py -3.12 -m experiments.moon.exp_006_train_only_eda.run_train_only_eda `
  --train-path data/raw/train.csv `
  --output-dir experiments/moon/exp_006_train_only_eda/results/eda

py -3.12 -m experiments.moon.exp_006_train_only_eda.run_cv `
  --train-path data/raw/train.csv `
  --output-dir experiments/moon/exp_006_train_only_eda/results/cv
```

The generated CSVs are local artifacts and must not be committed.  Commit only
code, configuration, this rationale, and lightweight `metrics.json` when a run
is accepted as reproducible.
