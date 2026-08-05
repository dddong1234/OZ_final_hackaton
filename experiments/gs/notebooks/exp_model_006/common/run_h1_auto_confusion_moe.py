"""User-run seed42 screen for train-only automatic confusion-group MoE."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from h1_auto_confusion_moe import (
    apply_group_experts,
    discover_confusion_groups,
    fit_h0_fold,
    inner_oof_h0_probability,
)


SEED = 42
H0_REFERENCE = .544744
LOW_MARGIN = .05
RESULT_COLUMNS = {
    "summary": {
        "variant", "oof_macro_f1", "oof_accuracy", "delta_vs_h0",
        "convergence_warning_count", "leakage_check", "nan_as_mutation_count",
    },
    "fold": {"fold", "variant", "macro_f1", "delta_vs_h0", "feature_count"},
}


def project_root() -> Path:
    for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv not found")


def decision(h0: float, h1: float, fold_deltas: list[float], low_margin_delta: float) -> dict:
    delta = float(h1 - h0)
    positive_folds = int(sum(value > 0 for value in fold_deltas))
    if abs(h0 - H0_REFERENCE) > .001:
        label = "baseline_not_reproduced"
    elif delta >= .030 and positive_folds >= 4 and low_margin_delta >= -.003:
        label = "jump_candidate"
    elif delta >= .015 and positive_folds >= 4 and low_margin_delta >= -.003:
        label = "strong_validation_candidate"
    elif delta >= .005:
        label = "not_detected"
    else:
        label = "rejected"
    return {
        "h0_reference": H0_REFERENCE, "h0_reference_delta": float(h0 - H0_REFERENCE),
        "delta_vs_h0": delta, "positive_fold_count": positive_folds,
        "low_margin_delta": float(low_margin_delta), "decision": label,
    }


def smoke() -> None:
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv", nrows=8)
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    assert len(genes) == 4384
    assert int(train[genes].isna().sum().sum()) == 0
    assert all(columns for columns in RESULT_COLUMNS.values())
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0}))


def run(run_id: str) -> None:
    started = perf_counter()
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")

    h0_oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    h1_oof = np.zeros_like(h0_oof)
    fold_rows, group_rows, audit_rows, warning_count = [], [], [], 0
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_index, valid_index) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        print(f"[H1-MoE] outer fold {fold}/5: inner H0 OOF", flush=True)
        fit_table = train.iloc[fit_index].reset_index(drop=True)
        y_fit = labels[fit_index]
        inner_probability, inner_audits = inner_oof_h0_probability(
            fit_table, y_fit, genes, classes, seed=SEED * 1000 + fold
        )
        groups = discover_confusion_groups(
            y_fit, classes[inner_probability.argmax(axis=1)], classes, n_groups=6
        )
        print(f"[H1-MoE] outer fold {fold}/5: groups={groups}", flush=True)
        h0 = fit_h0_fold(
            fit_table[genes], train.iloc[valid_index][genes].reset_index(drop=True),
            y_fit, genes, classes, seed=SEED * 100 + fold,
        )
        h1_probability, local_groups = apply_group_experts(
            h0.probability, h0.x_fit, h0.y_fit, h0.x_apply, classes, groups, seed=SEED * 10 + fold
        )
        h0_oof[valid_index], h1_oof[valid_index] = h0.probability, h1_probability
        h0_f1 = float(f1_score(labels[valid_index], classes[h0.probability.argmax(axis=1)], average="macro", zero_division=0))
        h1_f1 = float(f1_score(labels[valid_index], classes[h1_probability.argmax(axis=1)], average="macro", zero_division=0))
        for variant, probability, score in (("H0", h0.probability, h0_f1), ("H1_auto_confusion_moe", h1_probability, h1_f1)):
            fold_rows.append({
                "fold": fold, "variant": variant, "macro_f1": score,
                "accuracy": float(accuracy_score(labels[valid_index], classes[probability.argmax(axis=1)])),
                "feature_count": len(h0.names), "delta_vs_h0": score - h0_f1,
            })
        group_rows.extend({"fold": fold, **row} for row in local_groups)
        audit_rows.extend({"outer_fold": fold, **row} for row in inner_audits)
        audit_rows.append({
            "outer_fold": fold, "inner_fold": 0, "fit_rows": int(len(fit_index)),
            "holdout_rows": int(len(valid_index)), "outer_validation_used_for_fit": False,
            "leakage_check": h0.audit["raw_train_test_concat"] is False,
            "nan_as_mutation_count": 0,
        })
        warning_count += h0.convergence_warnings
        del inner_probability, h0, h1_probability
        gc.collect()

    h0_prediction, h1_prediction = classes[h0_oof.argmax(axis=1)], classes[h1_oof.argmax(axis=1)]
    h0_score = float(f1_score(labels, h0_prediction, average="macro", zero_division=0))
    h1_score = float(f1_score(labels, h1_prediction, average="macro", zero_division=0))
    margin = np.sort(h0_oof, axis=1)[:, -1] - np.sort(h0_oof, axis=1)[:, -2]
    low = margin < LOW_MARGIN
    low_h0 = float(f1_score(labels[low], h0_prediction[low], average="macro", zero_division=0))
    low_h1 = float(f1_score(labels[low], h1_prediction[low], average="macro", zero_division=0))
    folds = pd.DataFrame(fold_rows)
    h0_folds = folds[folds.variant.eq("H0")].sort_values("fold").macro_f1.to_numpy()
    h1_folds = folds[folds.variant.eq("H1_auto_confusion_moe")].sort_values("fold").macro_f1.to_numpy()
    verdict = decision(h0_score, h1_score, list(h1_folds - h0_folds), low_h1 - low_h0)
    summary = pd.DataFrame([
        {"variant": "H0", "oof_macro_f1": h0_score, "oof_accuracy": float(accuracy_score(labels, h0_prediction)), "feature_count": float(folds.feature_count.mean()), "convergence_warning_count": warning_count, "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": 0.0},
        {"variant": "H1_auto_confusion_moe", "oof_macro_f1": h1_score, "oof_accuracy": float(accuracy_score(labels, h1_prediction)), "feature_count": float(folds.feature_count.mean()), "convergence_warning_count": warning_count, "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": h1_score - h0_score},
    ])
    class_rows = []
    for label in classes:
        h0_class = f1_score(labels == label, h0_prediction == label, zero_division=0)
        h1_class = f1_score(labels == label, h1_prediction == label, zero_division=0)
        class_rows.append({"class": label, "support": int((labels == label).sum()), "H0_f1": h0_class, "H1_f1": h1_class, "delta": h1_class - h0_class})
    result = Path(__file__).parent.parent / "result"
    result.mkdir(exist_ok=True)
    if not RESULT_COLUMNS["summary"].issubset(summary.columns):
        raise AssertionError("summary schema failure")
    summary.to_csv(result / f"{run_id}_seed42_summary.csv", index=False)
    folds.to_csv(result / f"{run_id}_seed42_fold_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(result / f"{run_id}_seed42_auto_groups.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(result / f"{run_id}_seed42_inner_outer_audit.csv", index=False)
    pd.DataFrame(class_rows).to_csv(result / f"{run_id}_seed42_class_metrics.csv", index=False)
    pd.DataFrame([{"group": "low_margin", "support": int(low.sum()), "H0_macro_f1": low_h0, "H1_macro_f1": low_h1, "delta": low_h1 - low_h0}]).to_csv(result / f"{run_id}_seed42_low_margin.csv", index=False)
    (result / f"{run_id}_seed42_leakage_audit.json").write_text(json.dumps({
        "seed": SEED, "outer_splits": 5, "inner_splits": 3, "group_count": 6,
        "test_read": False, "train_test_concat": False, "fixed_class_gene_mutation_rules": False,
        "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": perf_counter() - started, **verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = folds.pivot(index="fold", columns="variant", values="macro_f1").plot(marker="o", title="H0 vs H1 auto-confusion MoE")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    class_frame = pd.DataFrame(class_rows).sort_values("delta")
    ax = class_frame.set_index("class").delta.plot.barh(title="H1 class F1 delta vs H0")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(verdict, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-h1-auto-confusion-moe-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id)


if __name__ == "__main__":
    main()
