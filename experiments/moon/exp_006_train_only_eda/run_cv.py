"""Compare mutation feature sets under the common fixed 5-fold LR protocol."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from experiments.moon.exp_006_train_only_eda.variant_features import (
    VariantFeatureTransformer,
    WTBinaryTransformer,
)


SEED = 42
N_SPLITS = 5
EXPERIMENT_ID = "moon-exp-006-train-only-eda"


def fixed_model() -> LogisticRegression:
    return LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=SEED)


def make_transformer(name: str):
    if name == "wt_binary":
        return WTBinaryTransformer()
    return VariantFeatureTransformer(feature_set=name, recurrent_min_count=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["wt_binary", "gene_burden", "functional_recurrent"])
    args = parser.parse_args()
    train = pd.read_csv(args.train_path, low_memory=False)
    X = train.drop(columns=["ID", "SUBCLASS"])
    y = train["SUBCLASS"]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    result_rows: list[dict[str, object]] = []
    candidate_oof: list[dict[str, object]] = []
    for candidate in args.candidates:
        fold_scores: list[dict[str, object]] = []
        oof_prediction = np.empty(len(train), dtype=object)
        for fold, (fit_index, valid_index) in enumerate(splitter.split(X, y), start=1):
            started = time.perf_counter()
            pipeline = Pipeline([("features", clone(make_transformer(candidate))), ("model", fixed_model())])
            pipeline.fit(X.iloc[fit_index], y.iloc[fit_index])
            prediction = pipeline.predict(X.iloc[valid_index])
            oof_prediction[valid_index] = prediction
            transformer = pipeline.named_steps["features"]
            feature_count = len(transformer.get_feature_names_out()) if hasattr(transformer, "get_feature_names_out") else X.shape[1]
            fold_scores.append(
                {
                    "candidate": candidate,
                    "fold": fold,
                    "train_rows": len(fit_index),
                    "valid_rows": len(valid_index),
                    "feature_count": int(feature_count),
                    "accuracy": float(accuracy_score(y.iloc[valid_index], prediction)),
                    "f1_macro": float(f1_score(y.iloc[valid_index], prediction, average="macro", zero_division=0)),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                }
            )
            row = fold_scores[-1]
            print(f"{candidate} fold {fold}/{N_SPLITS}: macro_f1={row['f1_macro']:.5f}, features={feature_count}, sec={row['elapsed_seconds']}", flush=True)
        result_rows.extend(fold_scores)
        candidate_oof.append(
            {
                "candidate": candidate,
                "oof_accuracy": float(accuracy_score(y, oof_prediction)),
                "oof_f1_macro": float(f1_score(y, oof_prediction, average="macro", zero_division=0)),
            }
        )
    folds = pd.DataFrame(result_rows)
    summary = folds.groupby("candidate").agg(
        fold_accuracy_mean=("accuracy", "mean"),
        fold_accuracy_std=("accuracy", "std"),
        fold_f1_macro_mean=("f1_macro", "mean"),
        fold_f1_macro_std=("f1_macro", "std"),
        feature_count_min=("feature_count", "min"),
        feature_count_max=("feature_count", "max"),
    ).reset_index().merge(pd.DataFrame(candidate_oof), on="candidate", how="left")
    base = summary.loc[summary.candidate.eq("wt_binary"), "oof_f1_macro"]
    if not base.empty:
        summary["delta_vs_wt_binary_oof"] = summary.oof_f1_macro - float(base.iloc[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    metrics = {
        "experiment": EXPERIMENT_ID,
        "scope": "train.csv only; test.csv was not opened",
        "validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "model": {"name": "LogisticRegression", "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced", "random_state": 42},
        "candidates": summary.to_dict(orient="records"),
        "environment": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
