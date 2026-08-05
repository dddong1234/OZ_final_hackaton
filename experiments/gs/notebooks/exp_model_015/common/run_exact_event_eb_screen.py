"""H0 Selective-EB versus automatic exact-event Empirical-Bayes screen.

The added representation is every fold-train gene__normalized_event token.
There are no fixed class, gene, allele, support, position-bin, or top-k rules.
This screen reads train.csv only; test data is never opened.
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
SCREEN_DELTA = 0.015


def _gs_common(model: str) -> Path:
    path = HERE.parents[2] / model / "common"
    if not path.exists():
        raise FileNotFoundError(f"required GS H0 module missing: {path}")
    return path


for _path in (_gs_common("exp_model_006"), _gs_common("exp_model_007")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from h0_faithful_pipeline import _aligned_probability, build_design_matrices, fit_vocabulary, transform_rows  # noqa: E402
from h0_selective_eb_replacement import (  # noqa: E402
    EB_ALPHA, EB_CLIP, EB_SHRINKAGE, empirical_bayes_features,
    fixed_branch_replacement, selective_probability,
)
from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402


@dataclass(frozen=True)
class ExactEBState:
    selected: np.ndarray
    weights: np.ndarray


def fit_exact_eb(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> ExactEBState:
    """Fit posterior-shrunk class evidence for every observed exact token."""
    matrix = matrix.tocsr()
    support = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    selected = np.flatnonzero((support > 0) & (support < matrix.shape[0]))
    if not len(selected):
        return ExactEBState(selected, np.zeros((len(classes), 0), dtype=np.float32))
    x = matrix[:, selected]
    support = support[selected]
    prior = (support + EB_ALPHA) / (len(labels) + 2.0 * EB_ALPHA)
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, label in enumerate(classes):
        positive_mask = labels == label
        positive = np.asarray(x[positive_mask].getnnz(axis=0)).ravel().astype(np.float64)
        negative = support - positive
        pos_rate = (positive + EB_SHRINKAGE * prior) / (positive_mask.sum() + EB_SHRINKAGE)
        neg_rate = (negative + EB_SHRINKAGE * prior) / ((~positive_mask).sum() + EB_SHRINKAGE)
        pos_rate = np.clip(pos_rate, 1e-6, 1.0 - 1e-6)
        neg_rate = np.clip(neg_rate, 1e-6, 1.0 - 1e-6)
        weights[class_index] = np.log(pos_rate / (1.0 - pos_rate)) - np.log(neg_rate / (1.0 - neg_rate))
    return ExactEBState(selected, np.clip(weights, -EB_CLIP, EB_CLIP).astype(np.float32))


def apply_exact_eb(matrix: sparse.csr_matrix, state: ExactEBState, class_count: int) -> np.ndarray:
    if not len(state.selected):
        return np.zeros((matrix.shape[0], class_count), dtype=np.float32)
    selected = matrix[:, state.selected]
    evidence = np.asarray(selected @ state.weights.T, dtype=np.float32)
    scale = np.sqrt(np.maximum(np.asarray(selected.getnnz(axis=1)).ravel(), 1.0))
    return evidence / scale[:, None]


def cross_fitted_exact_eb(
    fit_exact: sparse.csr_matrix, apply_exact: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-train OOF standardization; outer validation is transform-only."""
    train_score = np.zeros((fit_exact.shape[0], len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for inner_fit, inner_valid in splitter.split(np.zeros(len(labels)), labels):
        state = fit_exact_eb(fit_exact[inner_fit], labels[inner_fit], classes)
        train_score[inner_valid] = apply_exact_eb(fit_exact[inner_valid], state, len(classes))
    final_state = fit_exact_eb(fit_exact, labels, classes)
    apply_score = apply_exact_eb(apply_exact, final_state, len(classes))
    mean = train_score.mean(axis=0, keepdims=True)
    std = np.maximum(train_score.std(axis=0, keepdims=True), 1e-6)
    return ((train_score - mean) / std).astype(np.float32), ((apply_score - mean) / std).astype(np.float32)


def exact_eb_features(fit_frame: pd.DataFrame, apply_frame: pd.DataFrame, labels: np.ndarray, classes: np.ndarray, genes: list[str], seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    vocabulary = fit_vocabulary(fit_frame, genes)
    fit = transform_rows(fit_frame, genes, vocabulary)
    apply = transform_rows(apply_frame, genes, vocabulary)
    train_score, valid_score = cross_fitted_exact_eb(fit.exact, apply.exact, labels, classes, seed)
    return train_score, valid_score, int(fit.exact.shape[1])


def fit_lr(x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_valid: sparse.csr_matrix, classes: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_fit, y_fit)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return _aligned_probability(model, model.predict_proba(x_valid), classes).astype(np.float32), int(warning_count)


def metric(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[probability.argmax(axis=1)]
    return float(f1_score(labels, prediction, average="macro", zero_division=0)), float(accuracy_score(labels, prediction)), prediction


def run(run_id: str, seed: int) -> None:
    train = pd.read_csv(TRAIN_CSV)
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    oof = {name: np.zeros((len(train), len(classes)), dtype=np.float32) for name in ("h0", "exact")}
    fold_rows: list[dict] = []
    audit_rows: list[dict] = []
    started = perf_counter()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (fit_index, valid_index) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        print(f"[exact-event EB] seed {seed}, fold {fold}/5", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        fold_seed = seed * 100 + fold
        baseline = fit_fold(fit_frame, valid_frame, labels[fit_index], genes, classes, seed=fold_seed)
        x_fit, x_valid, _, design_audit = build_design_matrices(fit_frame, valid_frame, labels[fit_index], genes, seed=fold_seed)
        type_fit, type_valid = empirical_bayes_features(fit_frame, valid_frame, labels[fit_index], classes, genes, seed=fold_seed)
        exact_fit, exact_valid, exact_vocab_size = exact_eb_features(fit_frame, valid_frame, labels[fit_index], classes, genes, fold_seed)
        augmented_fit = sparse.hstack([x_fit, sparse.csr_matrix(type_fit), sparse.csr_matrix(exact_fit)], format="csr")
        augmented_valid = sparse.hstack([x_valid, sparse.csr_matrix(type_valid), sparse.csr_matrix(exact_valid)], format="csr")
        exact_lr, exact_warning = fit_lr(augmented_fit, labels[fit_index], augmented_valid, classes, fold_seed)
        gated_exact_lr, exact_non_eb_mask = selective_probability(baseline["h0_lr"], exact_lr)
        exact_candidate = fixed_branch_replacement(gated_exact_lr, baseline["specialist"])
        oof["h0"][valid_index] = baseline["candidate"]
        oof["exact"][valid_index] = exact_candidate
        for variant, probability, count in (("H0_selective_EB", baseline["candidate"], baseline["candidate_feature_count"]), ("exact_event_EB", exact_candidate, augmented_fit.shape[1])):
            macro, accuracy, _ = metric(labels[valid_index], probability, classes)
            fold_rows.append({"seed": seed, "fold": fold, "variant": variant, "macro_f1": macro, "accuracy": accuracy, "feature_count": int(count), "exact_vocabulary_size": exact_vocab_size})
        audit_rows.append({"seed": seed, "fold": fold, "test_read": False, "train_test_concat": False, "outer_validation_used_for_fit": False, "inner_crossfit_only": True, "fixed_class_gene_exact_mutation_rules": False, "exact_support_cutoff": None, "leakage_check": not bool(design_audit["raw_train_test_concat"]), "nan_as_mutation_count": int(design_audit["nan_as_mutation_count"]), "convergence_warning_count": int(baseline["h0_warning"] + baseline["eb_warning"] + exact_warning), "exact_gate_non_eb_rows": int(exact_non_eb_mask.sum())})
        del baseline, x_fit, x_valid, type_fit, type_valid, exact_fit, exact_valid, augmented_fit, augmented_valid
        gc.collect()
    folds = pd.DataFrame(fold_rows)
    audits = pd.DataFrame(audit_rows)
    summary_rows, class_rows = [], []
    for key, variant in (("h0", "H0_selective_EB"), ("exact", "exact_event_EB")):
        macro, accuracy, prediction = metric(labels, oof[key], classes)
        summary_rows.append({"seed": seed, "variant": variant, "oof_macro_f1": macro, "oof_accuracy": accuracy, "feature_count_mean": float(folds.loc[folds.variant.eq(variant), "feature_count"].mean()), "convergence_warning_count": int(audits.convergence_warning_count.sum()), "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started})
        precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"seed": seed, "variant": variant, "class": label, "precision": p, "recall": r, "f1": score, "support": int(count)} for label, p, r, score, count in zip(classes, precision, recall, f1, support))
    summary = pd.DataFrame(summary_rows)
    baseline_score = float(summary.loc[summary.variant.eq("H0_selective_EB"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_h0"] = summary.oof_macro_f1 - baseline_score
    fold_pivot = folds.pivot(index="fold", columns="variant", values="macro_f1")
    delta = float(summary.loc[summary.variant.eq("exact_event_EB"), "delta_vs_h0"].iloc[0])
    decision = {"run_id": run_id, "seed": seed, "screen_delta_required": SCREEN_DELTA, "delta_vs_h0": delta, "positive_fold_count": int((fold_pivot["exact_event_EB"] > fold_pivot["H0_selective_EB"]).sum()), "screen_pass": bool(delta >= SCREEN_DELTA and (fold_pivot["exact_event_EB"] > fold_pivot["H0_selective_EB"]).sum() >= 4), "test_read": False, "train_test_concat": False, "fixed_class_gene_exact_mutation_rules": False, "exact_support_cutoff": None, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max())}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_DIR / f"{run_id}_seed{seed}_summary.csv", index=False)
    folds.to_csv(RESULT_DIR / f"{run_id}_seed{seed}_fold_metrics.csv", index=False)
    audits.to_csv(RESULT_DIR / f"{run_id}_seed{seed}_fold_audit.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULT_DIR / f"{run_id}_seed{seed}_class_metrics.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(train)), "truth": labels, **{f"h0__{label}": oof["h0"][:, index] for index, label in enumerate(classes)}, **{f"exact__{label}": oof["exact"][:, index] for index, label in enumerate(classes)}}).to_csv(RESULT_DIR / f"{run_id}_seed{seed}_oof_probabilities.csv", index=False)
    (RESULT_DIR / f"{run_id}_seed{seed}_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = fold_pivot.plot(marker="o", figsize=(8, 4), title="H0 vs exact-event EB"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(RESULT_DIR / f"{run_id}_seed{seed}_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    metrics = pd.DataFrame(class_rows).pivot(index="class", columns="variant", values="f1")
    ax = (metrics["exact_event_EB"] - metrics["H0_selective_EB"]).sort_values().plot.barh(figsize=(8, 7), title="Class F1: exact-event EB − H0"); ax.figure.tight_layout(); ax.figure.savefig(RESULT_DIR / f"{run_id}_seed{seed}_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def smoke() -> None:
    labels = np.asarray(["A", "A", "B", "B", "C", "C"])
    matrix = sparse.csr_matrix(np.asarray([[1, 0], [1, 1], [0, 1], [0, 1], [1, 0], [0, 0]], dtype=np.float32))
    state = fit_exact_eb(matrix, labels, np.asarray(["A", "B", "C"]))
    assert len(state.selected) == 2
    assert apply_exact_eb(matrix, state, 3).shape == (6, 3)
    assert not TRAIN_CSV.name == "test.csv"
    print(json.dumps({"smoke": "ok", "test_read": False, "fixed_class_gene_exact_mutation_rules": False, "nan_as_mutation_count": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-exact-event-eb-01")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id, args.seed)
