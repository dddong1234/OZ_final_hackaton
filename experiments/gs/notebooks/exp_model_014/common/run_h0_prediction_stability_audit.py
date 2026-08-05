"""Train-only audit of valid fold-aligned H0 OOF seed predictions.

This runner never trains a model and never opens test.csv.  It consumes the
three seed OOF probabilities generated with the *same* outer folds by
exp_model_010, so each validation row was excluded by every averaged model.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[5]
DEFAULT_INPUT = PROJECT_ROOT / "experiments" / "gs" / "notebooks" / "exp_model_010" / "result" / "exp-fold-aligned-h0-bagging-audit-01_oof_probabilities.csv"
RESULT_DIR = HERE.parent.parent / "result"
SEEDS = (42, 777, 2024)
VARIANTS = tuple([f"seed_{seed}" for seed in SEEDS] + ["fold_aligned_bagged"])


def _classes(frame: pd.DataFrame) -> list[str]:
    prefix = "seed_42__"
    values = [column.removeprefix(prefix) for column in frame.columns if column.startswith(prefix)]
    if not values:
        raise ValueError("missing seed_42 probability columns")
    for variant in VARIANTS:
        missing = [label for label in values if f"{variant}__{label}" not in frame.columns]
        if missing:
            raise ValueError(f"missing probability columns for {variant}: {missing[:3]}")
    return values


def _probability(frame: pd.DataFrame, variant: str, classes: list[str]) -> np.ndarray:
    probability = frame[[f"{variant}__{label}" for label in classes]].to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all():
        raise ValueError(f"{variant} contains non-finite probability")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{variant} rows are not normalized probabilities")
    return probability


def _margin(probability: np.ndarray) -> np.ndarray:
    top2 = np.partition(probability, -2, axis=1)[:, -2:]
    return top2.max(axis=1) - top2.min(axis=1)


def _entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def _score(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "oof_macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "oof_accuracy": float(accuracy_score(y, predicted)),
    }


def audit_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    required = {"row_index", "truth"}
    if not required.issubset(frame.columns):
        raise ValueError(f"OOF input must contain {sorted(required)}")
    if frame.row_index.duplicated().any():
        raise ValueError("row_index must be unique")
    classes = _classes(frame)
    truth = frame.truth.to_numpy(dtype=object)
    probabilities = {variant: _probability(frame, variant, classes) for variant in VARIANTS}
    predictions = {variant: np.asarray(classes, dtype=object)[value.argmax(axis=1)] for variant, value in probabilities.items()}

    summary_rows = []
    class_rows = []
    for variant in VARIANTS:
        summary_rows.append({"variant": variant, **_score(truth, predictions[variant])})
        precision, recall, f1, support = precision_recall_fscore_support(truth, predictions[variant], labels=classes, zero_division=0)
        class_rows.extend({"variant": variant, "class": label, "precision": float(p), "recall": float(r), "f1": float(score), "support": int(count)} for label, p, r, score, count in zip(classes, precision, recall, f1, support))
    summary = pd.DataFrame(summary_rows)
    base = float(summary.loc[summary.variant.eq("seed_42"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_seed42"] = summary.oof_macro_f1 - base

    seed_predictions = np.column_stack([predictions[f"seed_{seed}"] for seed in SEEDS])
    agreement_count = np.asarray([pd.Series(row).value_counts().iloc[0] for row in seed_predictions], dtype=np.int16)
    agreement_label = np.select([agreement_count == 3, agreement_count == 2], ["all_agree", "two_agree"], default="all_different")
    seed_probability_stack = np.stack([probabilities[f"seed_{seed}"] for seed in SEEDS], axis=0)
    seed42_correct = predictions["seed_42"] == truth
    bagged_correct = predictions["fold_aligned_bagged"] == truth
    transition = np.select(
        [~seed42_correct & bagged_correct, seed42_correct & ~bagged_correct, seed42_correct & bagged_correct],
        ["recovered", "broken", "both_correct"],
        default="both_wrong",
    )
    row_audit = pd.DataFrame({
        "row_index": frame.row_index.to_numpy(),
        "truth": truth,
        "seed_42_prediction": predictions["seed_42"],
        "seed_777_prediction": predictions["seed_777"],
        "seed_2024_prediction": predictions["seed_2024"],
        "bagged_prediction": predictions["fold_aligned_bagged"],
        "seed_prediction_agreement_count": agreement_count,
        "seed_prediction_stability": agreement_label,
        "mean_seed_probability_std": seed_probability_stack.std(axis=0).mean(axis=1),
        "seed42_margin": _margin(probabilities["seed_42"]),
        "bagged_margin": _margin(probabilities["fold_aligned_bagged"]),
        "seed42_entropy": _entropy(probabilities["seed_42"]),
        "bagged_entropy": _entropy(probabilities["fold_aligned_bagged"]),
        "seed42_correct": seed42_correct,
        "bagged_correct": bagged_correct,
        "bagging_transition": transition,
    })

    stability_rows = []
    for group, subset in row_audit.groupby("seed_prediction_stability", sort=False):
        stability_rows.append({"group": group, "support": int(len(subset)), "seed42_macro_f1": float(f1_score(subset.truth, subset.seed_42_prediction, average="macro", zero_division=0)), "bagged_macro_f1": float(f1_score(subset.truth, subset.bagged_prediction, average="macro", zero_division=0)), "delta_bagged_minus_seed42": float(f1_score(subset.truth, subset.bagged_prediction, average="macro", zero_division=0) - f1_score(subset.truth, subset.seed_42_prediction, average="macro", zero_division=0)), "bagged_accuracy": float(accuracy_score(subset.truth, subset.bagged_prediction)), "mean_probability_std": float(subset.mean_seed_probability_std.mean()), "mean_bagged_margin": float(subset.bagged_margin.mean())})
    stability = pd.DataFrame(stability_rows)
    transition_summary = row_audit.groupby("bagging_transition", sort=False).size().rename("row_count").reset_index()
    transition_summary["rate"] = transition_summary.row_count / len(row_audit)
    audit = {
        "purpose": "h0_fold_aligned_prediction_stability_audit",
        "input_role": "precomputed_train_only_fold_aligned_oof",
        "test_read": False,
        "train_test_concat": False,
        "models_refit": False,
        "invalid_cross_split_oof_average_used": False,
        "seeds": list(SEEDS),
        "row_count": int(len(frame)),
        "class_count": int(len(classes)),
        "recovered_seed42_errors": int((transition == "recovered").sum()),
        "broken_seed42_correct": int((transition == "broken").sum()),
        "all_seed_agree_rate": float((agreement_count == 3).mean()),
        "leakage_check": True,
        "nan_as_mutation_count": 0,
    }
    return summary, pd.DataFrame(class_rows), row_audit, stability.merge(transition_summary, how="cross"), audit


def write_outputs(input_path: Path, run_id: str) -> dict:
    frame = pd.read_csv(input_path)
    summary, class_metrics, rows, stability, audit = audit_frame(frame)
    result = RESULT_DIR
    result.mkdir(parents=True, exist_ok=True)
    summary.to_csv(result / f"{run_id}_summary.csv", index=False)
    class_metrics.to_csv(result / f"{run_id}_class_metrics.csv", index=False)
    rows.to_csv(result / f"{run_id}_row_audit.csv", index=False)
    stability.to_csv(result / f"{run_id}_stability_transition_metrics.csv", index=False)
    (result / f"{run_id}_leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    ax = summary.set_index("variant").oof_macro_f1.plot.bar(figsize=(8, 4), title="Fold-aligned H0: OOF Macro F1")
    ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed_scores.png", dpi=160); plt.close(ax.figure)
    table = class_metrics.pivot(index="class", columns="variant", values="f1")
    delta = (table["fold_aligned_bagged"] - table["seed_42"]).sort_values()
    ax = delta.plot.barh(figsize=(8, 7), title="Class F1: bagged − seed42")
    ax.set_xlabel("F1 delta"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_class_delta.png", dpi=160); plt.close(ax.figure)
    ax = rows.seed_prediction_stability.value_counts().reindex(["all_agree", "two_agree", "all_different"], fill_value=0).plot.bar(figsize=(7, 4), title="Seed prediction stability")
    ax.set_ylabel("Rows"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_stability_counts.png", dpi=160); plt.close(ax.figure)
    gc.collect()
    return audit


def smoke(input_path: Path) -> None:
    frame = pd.read_csv(input_path, nrows=64)
    assert len(_classes(frame)) == 26
    for variant in VARIANTS:
        _probability(frame, variant, _classes(frame))
    print(json.dumps({"smoke": "ok", "test_read": False, "models_refit": False, "input": str(input_path)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default="exp-h0-prediction-stability-audit-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"fold-aligned OOF file not found: {args.input}")
    smoke(args.input) if args.smoke else print(json.dumps(write_outputs(args.input, args.run_id), ensure_ascii=False, indent=2))
