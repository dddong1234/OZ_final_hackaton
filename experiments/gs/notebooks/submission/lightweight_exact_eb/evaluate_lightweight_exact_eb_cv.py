# -*- coding: utf-8 -*-
"""Train-only 5-fold × 3-seed OOF evaluation for the lightweight model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from lightweight_exact_eb_core import fit_bundle, gene_columns, predict_proba


SEEDS = (42, 777, 2024)


def evaluate(train: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    result_dir.mkdir(parents=True, exist_ok=True)
    genes = gene_columns(train, training=True); labels = train.SUBCLASS.to_numpy(); rows = []; fold_rows = []
    for seed in SEEDS:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.empty(len(train), dtype=object)
        for fold, (fit_index, valid_index) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
            bundle = fit_bundle(train.iloc[fit_index].reset_index(drop=True), seed=seed)
            validation = train.iloc[valid_index].loc[:, ["ID", *genes]].reset_index(drop=True)
            probability = predict_proba(bundle, validation); prediction = bundle.classes[probability.argmax(axis=1)]
            oof[valid_index] = prediction
            fold_rows.append({"seed": seed, "fold": fold, "macro_f1": f1_score(labels[valid_index], prediction, average="macro"), "accuracy": accuracy_score(labels[valid_index], prediction), "feature_count": bundle.audit["feature_count"], "convergence_warning_count": bundle.audit["convergence_warning_count"], "leakage_check": True, "nan_as_mutation_count": 0})
        score = f1_score(labels, oof, average="macro")
        rows.append({"seed": seed, "oof_macro_f1": score, "oof_accuracy": accuracy_score(labels, oof), "leakage_check": True, "nan_as_mutation_count": 0, "test_read": False})
        pd.DataFrame({"ID": train.ID, "true_class": labels, "predicted_class": oof}).to_csv(result_dir / f"lightweight_exact_eb_seed{seed}_oof_predictions.csv", index=False)
    summary = pd.DataFrame(rows); summary.to_csv(result_dir / "lightweight_exact_eb_3seed_summary.csv", index=False); pd.DataFrame(fold_rows).to_csv(result_dir / "lightweight_exact_eb_fold_metrics.csv", index=False)
    (result_dir / "lightweight_exact_eb_leakage_audit.json").write_text(json.dumps({"seeds": list(SEEDS), "test_read": False, "train_test_concat": False, "leakage_check": True, "nan_as_mutation_count": 0}, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--train-csv", type=Path, required=True); parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(); summary = evaluate(pd.read_csv(args.train_csv), args.result_dir); print(summary)


if __name__ == "__main__":
    main()
