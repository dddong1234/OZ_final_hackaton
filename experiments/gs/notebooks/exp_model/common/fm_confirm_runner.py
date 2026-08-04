"""FM-04: fixed three-seed confirmation of the FM-03 screening winner.

No candidate, feature, optimizer, or blend-weight search is performed here.
Only train.csv is read; test is intentionally unsupported during OOF validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sparse_fm_runner as base
import fm_screen_runner as screen


def fixed_candidate() -> screen.Candidate:
    return screen.Candidate(
        candidate_id="FM-04_fixed_rank8_lr3e-4_balanced",
        kind="fm",
        rank=8,
        learning_rate=3e-4,
        class_weight="balanced",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", default="FM-04")
    args = parser.parse_args()

    root = base.find_root(Path.cwd())
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert train[genes].isna().sum().sum() == 0
    assert all(not base.normalise_cell(value) for gene in genes for value in train.loc[train[gene].isna(), gene])

    cache = base.Cache.build(train[genes], genes)
    labels = train[base.CFG.target_col].to_numpy()
    folds, _, classes = screen.build_folds(cache, labels, args.seed)
    baseline_probability, baseline_folds = screen.sklearn_baseline(folds, labels, classes, args.seed)
    summary, fold_rows, loss_rows = screen.run_candidate(
        fixed_candidate(), folds, labels, classes, baseline_probability, baseline_folds, args.seed
    )
    summary["experiment_id"] = "FM-04"
    summary["blend_weight_fm"] = 0.25
    summary["blend_weight_lr"] = 0.75
    output = root / "experiments" / "gs" / "notebooks" / "exp_model" / "result"
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run_id}_seed{args.seed}"
    pd.DataFrame([summary]).to_csv(output / f"{stem}_oof.csv", index=False)
    fold_rows.to_csv(output / f"{stem}_folds.csv", index=False)
    loss_rows.to_csv(output / f"{stem}_loss.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
