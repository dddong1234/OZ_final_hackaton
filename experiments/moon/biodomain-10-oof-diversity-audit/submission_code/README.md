# B10 clean inference entrypoint

This directory is the code-review entrypoint for the B10 submission. It has no
fixed mutation list, fixed cancer-pair list, external annotation, or test-wide
EDA/summary statistic.

## Safety invariants

- The active candidate is constructed with empty `exact_events`, `gene_pairs`,
  `gene_groups`, `hotspot_top_k`, and `contrast_pairs`.
- Contrast pairs are discovered only from the supplied train labels, using a
  3-fold proxy model on the train rows.
- Train and test are parsed separately. The test parser receives the train
  event vocabulary; unseen test event tokens do not become columns.
- Test rows contribute only their own deterministic string-derived aggregates
  during transform. No across-test-row statistic is calculated.
- The weight report is produced by train-only inner OOF selection and must
  certify that scope before inference starts.
- `--allow-test` is required explicitly; without it the script exits before
  opening `test.csv`.

## Usage

```powershell
py -3.12 .\experiments\moon\biodomain-10-oof-diversity-audit\submission_code\clean_inference.py `
  --allow-test `
  --weight-report .\experiments\moon\biodomain-10-oof-diversity-audit\outputs\final_weight_selection_train_only.json `
  --outdir .\experiments\moon\biodomain-10-oof-diversity-audit\outputs\clean_final
```

Do not use this entrypoint for development, EDA, or model selection. It is for
the one final full-train fit and inference after the candidate is locked.
