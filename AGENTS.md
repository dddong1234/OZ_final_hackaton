# Repository Working Rules

## Project purpose

This repository is a shared workspace where four members independently run AI experiments and later combine reproducible results.

## Default workflow

- Use Python 3.12.
- Treat JupyterLab notebooks as the primary beginner workflow.
- Start JupyterLab from the repository root so project-relative paths work consistently.
- SDH begins with `experiments/SDH/notebooks/00_quick_start.ipynb`.
- Other members begin with `experiments/member_<name>/notebooks/00_quick_start.ipynb`.
- SDH keeps EDA work under `experiments/SDH/exp_001_EDA/`.
- The command-line baseline under other members' `exp_001_baseline` folders is optional reference code.

## Ownership and freedom

- Each member primarily edits their own directory under `experiments/`.
- Members may organize personal notebooks and experiment code however they prefer.
- Do not force every personal experiment to mirror the baseline directory structure.
- Changes to shared files such as `common/`, `configs/`, `requirements*.txt`, and the root README should be reviewed for effects on every member.

## Data and artifact rules

- Shared source data belongs in `data/raw/` and must not be committed.
- Reusable processed data may be placed in `data/processed/` and must not be committed.
- Do not commit trained models, checkpoints, submissions, notebook outputs, OOF probabilities, or test probabilities.
- Commit lightweight experiment metadata such as `metrics.json` and explanatory README files.
- Never overwrite or modify the original files in `data/raw/`.

## Shared result contract

- Classification experiments should record both `accuracy` and `f1_macro`; Macro F1 is the primary comparison metric unless the team decides otherwise.
- Record enough metadata to reproduce a result: owner, experiment ID, model, seed, validation method, preprocessing summary, and parameters.
- Test probabilities intended for later ensembling must include `ID` plus one column per class.
- OOF probabilities intended for stacking must include `ID`, the true target, fold information, and one probability column per class.
- Keep class names and probability-column order explicit.

## Reproducibility

- Store experiment-specific settings with the experiment or notebook instead of relying on one fixed global seed or model configuration.
- Seeds, validation ratios, folds, and model parameters are experimental variables and may change between runs.
- For seed ensembles, keep each run's settings and probabilities separately before combining them.
- Clear notebook cell outputs before committing unless an output is deliberately required for documentation.

## Environment and checks

- Use `requirements.txt` for installing direct dependencies on Windows or macOS.
- Use `requirements-lock.txt` when reproducing the verified WSL Python 3.12 environment.
- Before sharing structural changes, check that notebooks are valid, imports compile, and installed packages have no dependency conflicts.
- Update `docs/PROJECT_LOG.md` when making a project-wide decision or changing a shared contract.

