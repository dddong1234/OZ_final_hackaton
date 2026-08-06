# Exact-event EB 3-seed standalone reproduction

`reproduce_h0_exact_event_eb_3seed_standalone.py` is a self-contained final pipeline for the validated Exact-event Empirical-Bayes submission branch.

It performs these stages in one file:

1. reads train/test/sample submission separately;
2. reports train-only EDA fields in the audit JSON;
3. parses WT, blank and NaN as zero events;
4. learns all mutation vocabularies, recurrent events, gene×event-type EB and exact-event EB only from train;
5. trains the structured logistic branches and fold-train-discovered LGBM specialists;
6. averages predeclared full-train seed `42/777/2024` probabilities and writes a submission CSV.

No cancer, gene or exact-mutation names are fixed in the script. Test is never concatenated with train or used to fit vocabulary, statistics, scaling, feature selection or specialist pairs.

The script has no pretrained model, checkpoint (`.pth`), or externally downloaded model file. It trains every LR/LGBM component from the supplied `data/raw/train.csv` during execution. The generated audit JSON records macOS/OS, Python, NumPy, pandas, SciPy, scikit-learn and LightGBM versions.

## Run

From the project root:

```bash
.venv/bin/python experiments/gs/notebooks/submission/reproduce_h0_exact_event_eb_3seed_standalone.py --smoke
.venv/bin/python experiments/gs/notebooks/submission/reproduce_h0_exact_event_eb_3seed_standalone.py
```

For the evaluator layout where input files are directly in `/data`, no path
option is needed. To use another data directory explicitly:

```bash
python reproduce_h0_exact_event_eb_3seed_standalone.py --data-dir /data
```

If the file is evaluated outside the repository, set an explicit result path:

```bash
python reproduce_h0_exact_event_eb_3seed_standalone.py --data-dir /data --output-dir /output
```

If copied elsewhere, pass the repository path explicitly:

```bash
python reproduce_h0_exact_event_eb_3seed_standalone.py --root /path/to/OZ_fianl_hackaton --smoke
```

The full run writes `submission_h0_exact_event_eb_seed42_777_2024_bagged.csv` and a sibling `*.audit.json` under `experiments/gs/notebooks/submission/`.

The smoke mode reads only `train.csv`; it does not read test or fit a model.
