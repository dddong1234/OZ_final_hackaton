"""Seed-42 screen for generic exact-event evidence-confidence features.

The runner reads train.csv only.  It compares the accepted exact-event EB H0
branch with the same branch augmented by fold-train contribution-shape scores.
No gene, class, event, or mutation list is fixed in this file.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[5]
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"
RESULT_DIR = HERE.parent.parent / "result"
SEED = 42
SCREEN_DELTA = 0.008
FEATURES_PER_CLASS = 9


def _add_common(model: str) -> None:
    path = HERE.parents[2] / model / "common"
    if not path.exists():
        raise FileNotFoundError(f"GS dependency missing: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


for _model in ("exp_model_006", "exp_model_007"):
    _add_common(_model)

from h0_faithful_pipeline import _aligned_probability, build_design_matrices, fit_vocabulary, transform_rows  # noqa: E402
from h0_selective_eb_replacement import EB_ALPHA, EB_CLIP, EB_SHRINKAGE, empirical_bayes_features, fixed_branch_replacement, selective_probability  # noqa: E402
from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402


@dataclass(frozen=True)
class ExactEvidenceState:
    selected: np.ndarray
    weights: np.ndarray
    reliability: np.ndarray


def fit_exact_evidence(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> ExactEvidenceState:
    """Fit posterior-shrunk exact-event evidence and generic support reliability."""
    matrix = matrix.tocsr()
    support = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    selected = np.flatnonzero((support > 0) & (support < matrix.shape[0]))
    if not len(selected):
        return ExactEvidenceState(selected, np.zeros((len(classes), 0), np.float32), np.zeros(0, np.float32))
    x = matrix[:, selected]
    support = support[selected]
    prior = (support + EB_ALPHA) / (len(labels) + 2.0 * EB_ALPHA)
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, label in enumerate(classes):
        positive_mask = labels == label
        positive = np.asarray(x[positive_mask].getnnz(axis=0)).ravel().astype(np.float64)
        negative = support - positive
        positive_rate = (positive + EB_SHRINKAGE * prior) / (positive_mask.sum() + EB_SHRINKAGE)
        negative_rate = (negative + EB_SHRINKAGE * prior) / ((~positive_mask).sum() + EB_SHRINKAGE)
        positive_rate = np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
        negative_rate = np.clip(negative_rate, 1e-6, 1.0 - 1e-6)
        weights[class_index] = np.log(positive_rate / (1.0 - positive_rate)) - np.log(negative_rate / (1.0 - negative_rate))
    reliability = support / (support + EB_SHRINKAGE)
    return ExactEvidenceState(selected, np.clip(weights, -EB_CLIP, EB_CLIP).astype(np.float32), reliability.astype(np.float32))


def apply_exact_scores(matrix: sparse.csr_matrix, state: ExactEvidenceState, class_count: int) -> np.ndarray:
    if not len(state.selected):
        return np.zeros((matrix.shape[0], class_count), dtype=np.float32)
    selected = matrix[:, state.selected]
    score = np.asarray(selected @ state.weights.T, dtype=np.float32)
    denominator = np.sqrt(np.maximum(np.asarray(selected.getnnz(axis=1)).ravel(), 1.0))
    return score / denominator[:, None]


def _shape_for_class(matrix: sparse.csr_matrix, weights: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    """Summarise sparse event contributions without materialising a dense event matrix."""
    matrix = matrix.tocsr()
    row_count = matrix.shape[0]
    output = np.zeros((row_count, FEATURES_PER_CLASS), dtype=np.float32)
    for row in range(row_count):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        if start == stop:
            continue
        contribution = matrix.data[start:stop] * weights[matrix.indices[start:stop]]
        absolute = np.abs(contribution)
        total_abs = float(absolute.sum())
        positive = contribution[contribution > 0]
        negative = contribution[contribution < 0]
        output[row, 0] = positive.sum() if len(positive) else 0.0
        output[row, 1] = negative.sum() if len(negative) else 0.0
        output[row, 2] = total_abs
        output[row, 3] = positive.max() if len(positive) else 0.0
        output[row, 4] = negative.min() if len(negative) else 0.0
        if total_abs > 0:
            ordered = np.sort(absolute)[::-1]
            output[row, 5] = ordered[0] / total_abs
            output[row, 6] = ordered[:3].sum() / total_abs
            proportions = absolute / total_abs
            entropy = -float((proportions * np.log(np.maximum(proportions, 1e-12))).sum())
            output[row, 7] = entropy / np.log(max(len(proportions), 2))
            output[row, 8] = float((absolute * reliability[matrix.indices[start:stop]]).sum())
    return output


def apply_confidence_features(matrix: sparse.csr_matrix, state: ExactEvidenceState, class_count: int) -> np.ndarray:
    """Return generic 9-statistic evidence shapes for every class."""
    if class_count != state.weights.shape[0]:
        raise ValueError("class count does not match fitted state")
    if not len(state.selected):
        return np.zeros((matrix.shape[0], class_count * FEATURES_PER_CLASS), dtype=np.float32)
    selected = matrix[:, state.selected]
    blocks = [_shape_for_class(selected, state.weights[index], state.reliability) for index in range(class_count)]
    return np.hstack(blocks).astype(np.float32)


def cross_fitted_exact_features(
    train_exact: sparse.csr_matrix, apply_exact: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build scaled exact scores and shape features with strict inner OOF fitting."""
    class_count = len(classes)
    train_scores = np.zeros((train_exact.shape[0], class_count), dtype=np.float32)
    train_shape = np.zeros((train_exact.shape[0], class_count * FEATURES_PER_CLASS), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_index, valid_index in splitter.split(np.zeros(len(labels)), labels):
        state = fit_exact_evidence(train_exact[fit_index], labels[fit_index], classes)
        train_scores[valid_index] = apply_exact_scores(train_exact[valid_index], state, class_count)
        train_shape[valid_index] = apply_confidence_features(train_exact[valid_index], state, class_count)
    state = fit_exact_evidence(train_exact, labels, classes)
    apply_scores = apply_exact_scores(apply_exact, state, class_count)
    apply_shape = apply_confidence_features(apply_exact, state, class_count)
    score_std = np.maximum(train_scores.std(axis=0, keepdims=True), 1e-6)
    shape_std = np.maximum(train_shape.std(axis=0, keepdims=True), 1e-6)
    return (
        ((train_scores - train_scores.mean(axis=0, keepdims=True)) / score_std).astype(np.float32),
        ((apply_scores - train_scores.mean(axis=0, keepdims=True)) / score_std).astype(np.float32),
        ((train_shape - train_shape.mean(axis=0, keepdims=True)) / shape_std).astype(np.float32),
        ((apply_shape - train_shape.mean(axis=0, keepdims=True)) / shape_std).astype(np.float32),
    )


def exact_evidence_features(
    fit_frame: pd.DataFrame, apply_frame: pd.DataFrame, labels: np.ndarray, classes: np.ndarray, genes: list[str], seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    vocabulary = fit_vocabulary(fit_frame, genes)
    parsed_fit = transform_rows(fit_frame, genes, vocabulary)
    parsed_apply = transform_rows(apply_frame, genes, vocabulary)
    scores_fit, scores_apply, shape_fit, shape_apply = cross_fitted_exact_features(parsed_fit.exact, parsed_apply.exact, labels, classes, seed)
    return scores_fit, scores_apply, shape_fit, shape_apply, int(parsed_fit.exact.shape[1])


def fit_lr(x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_apply: sparse.csr_matrix, classes: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_fit, y_fit)
    warnings_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return _aligned_probability(model, model.predict_proba(x_apply), classes).astype(np.float32), int(warnings_count)


def _metric(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[probability.argmax(axis=1)]
    return float(f1_score(labels, prediction, average="macro", zero_division=0)), float(accuracy_score(labels, prediction)), prediction


def run(run_id: str, seed: int) -> None:
    train = pd.read_csv(TRAIN_CSV)
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train no-NaN contract failed")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    oof = {"exact": np.zeros((len(train), len(classes)), np.float32), "confidence": np.zeros((len(train), len(classes)), np.float32)}
    fold_rows: list[dict] = []
    audit_rows: list[dict] = []
    started = perf_counter()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (fit_index, valid_index) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        print(f"[exact evidence confidence] seed {seed}, fold {fold}/5", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        fold_seed = seed * 100 + fold
        baseline = fit_fold(fit_frame, valid_frame, labels[fit_index], genes, classes, seed=fold_seed)
        x_fit, x_valid, _, design_audit = build_design_matrices(fit_frame, valid_frame, labels[fit_index], genes, seed=fold_seed)
        type_fit, type_valid = empirical_bayes_features(fit_frame, valid_frame, labels[fit_index], classes, genes, seed=fold_seed)
        exact_fit, exact_valid, shape_fit, shape_valid, vocabulary_size = exact_evidence_features(fit_frame, valid_frame, labels[fit_index], classes, genes, fold_seed)
        base_fit = sparse.hstack([x_fit, sparse.csr_matrix(type_fit), sparse.csr_matrix(exact_fit)], format="csr")
        base_valid = sparse.hstack([x_valid, sparse.csr_matrix(type_valid), sparse.csr_matrix(exact_valid)], format="csr")
        exact_probability, exact_warning = fit_lr(base_fit, labels[fit_index], base_valid, classes, fold_seed)
        confidence_fit = sparse.hstack([base_fit, sparse.csr_matrix(shape_fit)], format="csr")
        confidence_valid = sparse.hstack([base_valid, sparse.csr_matrix(shape_valid)], format="csr")
        confidence_probability, confidence_warning = fit_lr(confidence_fit, labels[fit_index], confidence_valid, classes, fold_seed)
        exact_gated, _ = selective_probability(baseline["h0_lr"], exact_probability)
        confidence_gated, fallback_mask = selective_probability(baseline["h0_lr"], confidence_probability)
        oof["exact"][valid_index] = fixed_branch_replacement(exact_gated, baseline["specialist"])
        oof["confidence"][valid_index] = fixed_branch_replacement(confidence_gated, baseline["specialist"])
        for key, label, feature_count in (("exact", "exact_event_EB", base_fit.shape[1]), ("confidence", "exact_event_EB_confidence", confidence_fit.shape[1])):
            macro, accuracy, _ = _metric(labels[valid_index], oof[key][valid_index], classes)
            fold_rows.append({"seed": seed, "fold": fold, "variant": label, "macro_f1": macro, "accuracy": accuracy, "feature_count": int(feature_count), "exact_vocabulary_size": vocabulary_size})
        audit_rows.append({"seed": seed, "fold": fold, "test_read": False, "train_test_concat": False, "outer_validation_used_for_fit": False, "inner_crossfit_only": True, "fixed_class_gene_exact_mutation_rules": False, "exact_support_cutoff": None, "confidence_statistics_per_class": FEATURES_PER_CLASS, "leakage_check": not bool(design_audit["raw_train_test_concat"]), "nan_as_mutation_count": int(design_audit["nan_as_mutation_count"]), "convergence_warning_count": int(baseline["h0_warning"] + baseline["eb_warning"] + exact_warning + confidence_warning), "confidence_gate_non_eb_rows": int(fallback_mask.sum())})
        del baseline, x_fit, x_valid, type_fit, type_valid, exact_fit, exact_valid, shape_fit, shape_valid, base_fit, base_valid, confidence_fit, confidence_valid
        gc.collect()
    folds, audits = pd.DataFrame(fold_rows), pd.DataFrame(audit_rows)
    summary_rows, class_rows = [], []
    for key, label in (("exact", "exact_event_EB"), ("confidence", "exact_event_EB_confidence")):
        macro, accuracy, prediction = _metric(labels, oof[key], classes)
        summary_rows.append({"seed": seed, "variant": label, "oof_macro_f1": macro, "oof_accuracy": accuracy, "feature_count_mean": float(folds.loc[folds.variant.eq(label), "feature_count"].mean()), "convergence_warning_count": int(audits.convergence_warning_count.sum()), "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started})
        precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"seed": seed, "variant": label, "class": item, "precision": p, "recall": r, "f1": score, "support": int(count)} for item, p, r, score, count in zip(classes, precision, recall, f1, support))
    summary = pd.DataFrame(summary_rows)
    reference = float(summary.loc[summary.variant.eq("exact_event_EB"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_exact_event_eb"] = summary.oof_macro_f1 - reference
    pivot = folds.pivot(index="fold", columns="variant", values="macro_f1")
    confidence_delta = float(summary.loc[summary.variant.eq("exact_event_EB_confidence"), "delta_vs_exact_event_eb"].iloc[0])
    class_matrix = pd.DataFrame(class_rows).pivot(index="class", columns="variant", values="f1")
    class_delta = class_matrix["exact_event_EB_confidence"] - class_matrix["exact_event_EB"]
    decision = {"run_id": run_id, "seed": seed, "delta_required": SCREEN_DELTA, "delta_vs_exact_event_eb": confidence_delta, "positive_fold_count": int((pivot["exact_event_EB_confidence"] > pivot["exact_event_EB"]).sum()), "minimum_class_delta": float(class_delta.min()), "screen_pass": bool(confidence_delta >= SCREEN_DELTA and (pivot["exact_event_EB_confidence"] > pivot["exact_event_EB"]).sum() >= 4 and class_delta.min() >= -0.05), "test_read": False, "train_test_concat": False, "fixed_class_gene_exact_mutation_rules": False, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max())}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_DIR / f"{run_id}_seed{seed}_summary.csv", index=False)
    folds.to_csv(RESULT_DIR / f"{run_id}_seed{seed}_fold_metrics.csv", index=False)
    audits.to_csv(RESULT_DIR / f"{run_id}_seed{seed}_fold_audit.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULT_DIR / f"{run_id}_seed{seed}_class_metrics.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(train)), "truth": labels, **{f"exact__{label}": oof["exact"][:, index] for index, label in enumerate(classes)}, **{f"confidence__{label}": oof["confidence"][:, index] for index, label in enumerate(classes)}}).to_csv(RESULT_DIR / f"{run_id}_seed{seed}_oof_probabilities.csv", index=False)
    (RESULT_DIR / f"{run_id}_seed{seed}_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    axis = pivot.plot(marker="o", figsize=(8, 4), title="Exact-event EB vs confidence EB"); axis.set_ylabel("Macro F1"); axis.figure.tight_layout(); axis.figure.savefig(RESULT_DIR / f"{run_id}_seed{seed}_fold_macro_f1.png", dpi=160); plt.close(axis.figure)
    axis = class_delta.sort_values().plot.barh(figsize=(8, 7), title="Class F1: confidence − exact-event EB"); axis.figure.tight_layout(); axis.figure.savefig(RESULT_DIR / f"{run_id}_seed{seed}_class_f1_delta.png", dpi=160); plt.close(axis.figure)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def smoke() -> dict:
    labels = np.asarray(["A", "A", "B", "B", "C", "C"])
    matrix = sparse.csr_matrix(np.asarray([[1, 0], [1, 1], [0, 1], [0, 1], [1, 0], [0, 0]], dtype=np.float32))
    state = fit_exact_evidence(matrix, labels, np.asarray(["A", "B", "C"]))
    shape = apply_confidence_features(matrix, state, 3)
    if shape.shape != (6, 27) or not np.isfinite(shape).all():
        raise AssertionError("confidence shape smoke contract failed")
    return {"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "fixed_class_gene_exact_mutation_rules": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-exact-evidence-confidence-01")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(smoke(), ensure_ascii=False), flush=True) if arguments.smoke else run(arguments.run_id, arguments.seed)
