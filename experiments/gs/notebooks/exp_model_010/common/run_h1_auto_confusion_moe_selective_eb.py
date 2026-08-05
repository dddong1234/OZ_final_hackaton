"""H1: fold-train automatic confusion-group MoE over final Selective-EB H0.

Seed-42 screen only. The runner reads train.csv only and never contains fixed
class, gene, or mutation identifiers.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve()
ROOT = HERE.parents[5]
TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
RESULT = HERE.parent.parent / "result"
SEED = 42
INNER_SPLITS = 3
GROUP_COUNT = 6
LOW_MARGIN = 0.05
H0_REFERENCE = 0.547915


def _add_source(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_add_source(HERE.parents[2] / "exp_model_007" / "common")
_add_source(HERE.parents[2] / "exp_model_006" / "common")
from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402
from h0_faithful_pipeline import build_design_matrices  # noqa: E402


def _metric(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[np.asarray(probability).argmax(axis=1)]
    return float(f1_score(y, prediction, average="macro", zero_division=0)), float(accuracy_score(y, prediction)), prediction


def discover_groups(truth: np.ndarray, prediction: np.ndarray, classes: np.ndarray) -> tuple[tuple[str, ...], ...]:
    """Merge the most bidirectionally confused groups; no identifier is fixed."""
    labels = [str(label) for label in classes]
    lookup = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        matrix[lookup[str(actual)], lookup[str(predicted)]] += 1
    groups = [(label,) for label in labels]
    while len(groups) > GROUP_COUNT:
        candidates = []
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                score = sum(matrix[lookup[a], lookup[b]] + matrix[lookup[b], lookup[a]] for a in groups[left] for b in groups[right])
                candidates.append((-int(score), left, right))
        _, left, right = min(candidates)
        merged = tuple(sorted(groups[left] + groups[right]))
        groups = [group for index, group in enumerate(groups) if index not in (left, right)] + [merged]
        groups.sort()
    return tuple(groups)


def apply_group_experts(base_probability: np.ndarray, x_fit, y_fit: np.ndarray, x_valid, classes: np.ndarray, groups: tuple[tuple[str, ...], ...], seed: int) -> tuple[np.ndarray, list[dict]]:
    """Replace only internal group ratios; every group retains its H0 mass."""
    output = np.asarray(base_probability, dtype=np.float64).copy()
    index = {str(label): position for position, label in enumerate(classes)}
    records = []
    for group_id, group in enumerate(groups):
        if len(group) < 2:
            continue
        mask = np.isin(y_fit, group)
        if int(mask.sum()) < len(group) * 2:
            records.append({"group_id": group_id, "classes": "|".join(group), "size": len(group), "fit_support": int(mask.sum()), "used": False})
            continue
        expert = LGBMClassifier(
            objective="multiclass", num_class=len(group), n_estimators=100, learning_rate=.02,
            num_leaves=20, min_child_samples=10, reg_alpha=0.0, reg_lambda=0.0,
            class_weight="balanced", random_state=seed + group_id, n_jobs=-1,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )
        expert.fit(x_fit[mask], y_fit[mask])
        local = {str(label): position for position, label in enumerate(expert.classes_)}
        local_probability = expert.predict_proba(x_valid)[:, [local[label] for label in group]]
        columns = [index[label] for label in group]
        mass = output[:, columns].sum(axis=1, keepdims=True)
        output[:, columns] = mass * local_probability
        np.testing.assert_allclose(output[:, columns].sum(axis=1), mass.ravel(), atol=1e-6)
        records.append({"group_id": group_id, "classes": "|".join(group), "size": len(group), "fit_support": int(mask.sum()), "used": True})
        del expert
        gc.collect()
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)
    return output.astype(np.float32), records


def inner_h0_oof(frame: pd.DataFrame, labels: np.ndarray, genes: list[str], classes: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict]]:
    oof = np.zeros((len(frame), len(classes)), dtype=np.float32)
    rows = []
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    for inner_fold, (fit_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(frame)), labels), 1):
        result = fit_fold(frame.iloc[fit_idx][genes].reset_index(drop=True), frame.iloc[valid_idx][genes].reset_index(drop=True), labels[fit_idx], genes, classes, seed=seed * 100 + inner_fold)
        oof[valid_idx] = result["candidate"]
        rows.append({"inner_fold": inner_fold, "fit_rows": int(len(fit_idx)), "valid_rows": int(len(valid_idx)), "outer_validation_used_for_fit": False, "leakage_check": not bool(result["audit"]["raw_train_test_concat"]), "nan_as_mutation_count": int(result["audit"]["nan_as_mutation_count"])})
        del result
        gc.collect()
    return oof, rows


def run(run_id: str) -> None:
    started = perf_counter()
    train = pd.read_csv(TRAIN_CSV)
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    h0_oof = np.zeros((len(train), len(classes)), dtype=np.float32)
    h1_oof = np.zeros_like(h0_oof)
    fold_rows, group_rows, audit_rows = [], [], []
    warning_count = 0
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_idx, valid_idx) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        print(f"[H1] fold {fold}/5: inner OOF for automatic groups", flush=True)
        fit_frame = train.iloc[fit_idx][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_idx][genes].reset_index(drop=True)
        y_fit = labels[fit_idx]
        inner_probability, inner_audit = inner_h0_oof(fit_frame, y_fit, genes, classes, SEED * 1000 + fold)
        groups = discover_groups(y_fit, classes[inner_probability.argmax(axis=1)], classes)
        print(f"[H1] fold {fold}/5 groups={groups}", flush=True)
        # The exact final H0 candidate; it reads no test data.
        h0_result = fit_fold(fit_frame, valid_frame, y_fit, genes, classes, seed=SEED * 100 + fold)
        # This matrix builder is called with the same outer-fold train only.
        x_fit, x_valid, names, matrix_audit = build_design_matrices(fit_frame, valid_frame, y_fit, genes, seed=SEED * 100 + fold)
        h1_probability, local_groups = apply_group_experts(h0_result["candidate"], x_fit, y_fit, x_valid, classes, groups, SEED * 10 + fold)
        h0_oof[valid_idx], h1_oof[valid_idx] = h0_result["candidate"], h1_probability
        h0_f1, h0_accuracy, _ = _metric(labels[valid_idx], h0_result["candidate"], classes)
        h1_f1, h1_accuracy, _ = _metric(labels[valid_idx], h1_probability, classes)
        fold_rows.extend([
            {"fold": fold, "variant": "H0_selective_EB", "macro_f1": h0_f1, "accuracy": h0_accuracy, "feature_count": int(h0_result["candidate_feature_count"]), "delta_vs_h0": 0.0},
            {"fold": fold, "variant": "H1_auto_confusion_moe", "macro_f1": h1_f1, "accuracy": h1_accuracy, "feature_count": int(h0_result["candidate_feature_count"]), "delta_vs_h0": h1_f1 - h0_f1},
        ])
        group_rows.extend({"fold": fold, **row} for row in local_groups)
        audit_rows.extend({"outer_fold": fold, **row} for row in inner_audit)
        audit_rows.append({"outer_fold": fold, "inner_fold": 0, "fit_rows": int(len(fit_idx)), "valid_rows": int(len(valid_idx)), "outer_validation_used_for_fit": False, "leakage_check": not bool(h0_result["audit"]["raw_train_test_concat"]) and not bool(matrix_audit["raw_train_test_concat"]), "nan_as_mutation_count": int(h0_result["audit"]["nan_as_mutation_count"])})
        warning_count += int(h0_result["h0_warning"] + h0_result["eb_warning"])
        del h0_result, x_fit, x_valid, inner_probability, h1_probability
        gc.collect()

    h0_f1, h0_accuracy, h0_prediction = _metric(labels, h0_oof, classes)
    h1_f1, h1_accuracy, h1_prediction = _metric(labels, h1_oof, classes)
    margin = np.sort(h0_oof, axis=1)[:, -1] - np.sort(h0_oof, axis=1)[:, -2]
    low = margin < LOW_MARGIN
    low_h0 = float(f1_score(labels[low], h0_prediction[low], average="macro", zero_division=0))
    low_h1 = float(f1_score(labels[low], h1_prediction[low], average="macro", zero_division=0))
    folds = pd.DataFrame(fold_rows)
    pivot = folds.pivot(index="fold", columns="variant", values="macro_f1")
    fold_delta = pivot["H1_auto_confusion_moe"] - pivot["H0_selective_EB"]
    delta = h1_f1 - h0_f1
    baseline_match = abs(h0_f1 - H0_REFERENCE) <= .001
    positive_folds = int((fold_delta > 0).sum())
    if not baseline_match:
        verdict = "baseline_not_reproduced"
    elif delta >= .030 and positive_folds >= 4 and low_h1 - low_h0 >= -.003:
        verdict = "jump_candidate"
    elif delta >= .015 and positive_folds >= 4 and low_h1 - low_h0 >= -.003:
        verdict = "strong_validation_candidate"
    elif delta >= .005:
        verdict = "not_detected"
    else:
        verdict = "rejected"
    precision_h0, recall_h0, class_h0, support = precision_recall_fscore_support(labels, h0_prediction, labels=classes, zero_division=0)
    precision_h1, recall_h1, class_h1, _ = precision_recall_fscore_support(labels, h1_prediction, labels=classes, zero_division=0)
    class_frame = pd.DataFrame({"class": classes, "support": support, "H0_precision": precision_h0, "H0_recall": recall_h0, "H0_f1": class_h0, "H1_precision": precision_h1, "H1_recall": recall_h1, "H1_f1": class_h1, "f1_delta": class_h1 - class_h0})
    audits = pd.DataFrame(audit_rows)
    summary = pd.DataFrame([
        {"variant": "H0_selective_EB", "oof_macro_f1": h0_f1, "oof_accuracy": h0_accuracy, "feature_count": float(folds.feature_count.mean()), "convergence_warning_count": warning_count, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started, "delta_vs_h0": 0.0},
        {"variant": "H1_auto_confusion_moe", "oof_macro_f1": h1_f1, "oof_accuracy": h1_accuracy, "feature_count": float(folds.feature_count.mean()), "convergence_warning_count": warning_count, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started, "delta_vs_h0": delta},
    ])
    audit = {"run_id": run_id, "h0_reference": H0_REFERENCE, "h0_reference_match": baseline_match, "h0_reference_delta": h0_f1 - H0_REFERENCE, "outer_splits": 5, "inner_splits": INNER_SPLITS, "automatic_group_count": GROUP_COUNT, "positive_fold_count": positive_folds, "low_margin_h0": low_h0, "low_margin_h1": low_h1, "low_margin_delta": low_h1 - low_h0, "test_read": False, "train_test_concat": False, "fixed_class_gene_mutation_rules": False, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "decision": verdict}
    RESULT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT / f"{run_id}_seed42_summary.csv", index=False)
    folds.to_csv(RESULT / f"{run_id}_seed42_fold_metrics.csv", index=False)
    class_frame.to_csv(RESULT / f"{run_id}_seed42_class_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(RESULT / f"{run_id}_seed42_auto_groups.csv", index=False)
    audits.to_csv(RESULT / f"{run_id}_seed42_inner_outer_audit.csv", index=False)
    pd.DataFrame([{"group": "low_margin_<0.05", "support": int(low.sum()), "H0_macro_f1": low_h0, "H1_macro_f1": low_h1, "delta": low_h1 - low_h0}]).to_csv(RESULT / f"{run_id}_seed42_low_margin.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(train)), "truth": labels, **{f"h0__{label}": h0_oof[:, index] for index, label in enumerate(classes)}, **{f"h1__{label}": h1_oof[:, index] for index, label in enumerate(classes)}}).to_csv(RESULT / f"{run_id}_seed42_oof_probabilities.csv", index=False)
    (RESULT / f"{run_id}_seed42_leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = pivot.plot(marker="o", title="H0 Selective-EB vs H1 automatic confusion MoE"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(RESULT / f"{run_id}_seed42_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    ax = class_frame.sort_values("f1_delta").set_index("class").f1_delta.plot.barh(title="H1 class F1 delta vs H0"); ax.figure.tight_layout(); ax.figure.savefig(RESULT / f"{run_id}_seed42_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def smoke() -> None:
    train = pd.read_csv(TRAIN_CSV, nrows=16)
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    assert len(genes) == 4384 and int(train[genes].isna().sum().sum()) == 0
    groups = discover_groups(np.array(["A", "A", "B", "B", "C", "C"]), np.array(["B", "A", "A", "B", "C", "C"]), np.array(["A", "B", "C"], dtype=object))
    assert len(groups) == 3 and all(group for group in groups)
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "fixed_rules": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-h1-auto-confusion-moe-selective-eb-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id)
