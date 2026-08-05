"""Train-only error-diversity audit: final H0 versus OVR LR on H0+EB inputs."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve()
ROOT = HERE.parents[5]
TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
RESULT = HERE.parent.parent / "result"
SEED = 42
H0_REFERENCE = 0.547915


def _add_source(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_add_source(HERE.parents[2] / "exp_model_007" / "common")
_add_source(HERE.parents[2] / "exp_model_006" / "common")
from h0_selective_eb_replacement import empirical_bayes_features  # noqa: E402
from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402
from h0_faithful_pipeline import build_design_matrices  # noqa: E402


def _metric(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[np.asarray(probability).argmax(axis=1)]
    return float(f1_score(y, prediction, average="macro", zero_division=0)), float(accuracy_score(y, prediction)), prediction


def _aligned_probability(model: OneVsRestClassifier, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(model.classes_)}
    return np.asarray(probability[:, [lookup[label] for label in classes]], dtype=np.float32)


def _fit_ovr(x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_valid: sparse.csr_matrix, classes: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    # Sequential fitting avoids multiplying 26 sparse LR working sets in memory.
    base = LogisticRegression(solver="lbfgs", C=.07, max_iter=2000, class_weight="balanced", random_state=seed)
    model = OneVsRestClassifier(base, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_fit, y_fit)
    warnings_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return _aligned_probability(model, model.predict_proba(x_valid), classes), int(warnings_count)


def run(run_id: str) -> None:
    started = perf_counter()
    train = pd.read_csv(TRAIN_CSV)  # Seed-42 screen must not read test.
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    h0_oof = np.zeros((len(train), len(classes)), dtype=np.float32)
    ovr_oof = np.zeros_like(h0_oof)
    fold_rows, audit_rows = [], []
    warning_count = 0
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        print(f"[OVR diversity] fold {fold}/5: H0 and same-input OVR", flush=True)
        fit_frame = train.iloc[fit_idx][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_idx][genes].reset_index(drop=True)
        y_fit = labels[fit_idx]
        h0 = fit_fold(fit_frame, valid_frame, y_fit, genes, classes, seed=SEED * 100 + fold)
        x_fit, x_valid, _, feature_audit = build_design_matrices(fit_frame, valid_frame, y_fit, genes, seed=SEED * 100 + fold)
        eb_fit, eb_valid = empirical_bayes_features(fit_frame, valid_frame, y_fit, classes, genes, seed=SEED * 100 + fold)
        ovr_probability, local_warning = _fit_ovr(sparse.hstack([x_fit, sparse.csr_matrix(eb_fit)], format="csr"), y_fit, sparse.hstack([x_valid, sparse.csr_matrix(eb_valid)], format="csr"), classes, SEED * 100 + fold)
        h0_oof[valid_idx], ovr_oof[valid_idx] = h0["candidate"], ovr_probability
        h0_f1, h0_accuracy, _ = _metric(labels[valid_idx], h0["candidate"], classes)
        ovr_f1, ovr_accuracy, _ = _metric(labels[valid_idx], ovr_probability, classes)
        fold_rows.extend([
            {"fold": fold, "variant": "H0_selective_EB", "macro_f1": h0_f1, "accuracy": h0_accuracy, "feature_count": int(h0["candidate_feature_count"]), "delta_vs_h0": 0.0},
            {"fold": fold, "variant": "OVR_EB_LR", "macro_f1": ovr_f1, "accuracy": ovr_accuracy, "feature_count": int(x_fit.shape[1] + eb_fit.shape[1]), "delta_vs_h0": ovr_f1 - h0_f1},
        ])
        audit_rows.append({"fold": fold, "test_read": False, "train_test_concat": False, "outer_validation_used_for_fit": False, "fold_train_only_eb": True, "fixed_class_gene_mutation_rules": False, "leakage_check": not bool(h0["audit"]["raw_train_test_concat"]) and not bool(feature_audit["raw_train_test_concat"]), "nan_as_mutation_count": int(h0["audit"]["nan_as_mutation_count"]), "h0_convergence_warning_count": int(h0["h0_warning"] + h0["eb_warning"]), "ovr_convergence_warning_count": local_warning})
        warning_count += int(h0["h0_warning"] + h0["eb_warning"] + local_warning)
        del h0, x_fit, x_valid, eb_fit, eb_valid, ovr_probability
        gc.collect()

    h0_f1, h0_accuracy, h0_prediction = _metric(labels, h0_oof, classes)
    ovr_f1, ovr_accuracy, ovr_prediction = _metric(labels, ovr_oof, classes)
    h0_correct, ovr_correct = h0_prediction == labels, ovr_prediction == labels
    recovered = (~h0_correct) & ovr_correct
    broken = h0_correct & (~ovr_correct)
    both_wrong = (~h0_correct) & (~ovr_correct)
    disagreement = h0_prediction != ovr_prediction
    # Oracle is diagnostic only: it is never a prediction or submission rule.
    oracle_prediction = np.where(h0_correct, h0_prediction, ovr_prediction)
    oracle_f1 = float(f1_score(labels, oracle_prediction, average="macro", zero_division=0))
    precision_h0, recall_h0, class_h0, support = precision_recall_fscore_support(labels, h0_prediction, labels=classes, zero_division=0)
    precision_ovr, recall_ovr, class_ovr, _ = precision_recall_fscore_support(labels, ovr_prediction, labels=classes, zero_division=0)
    class_frame = pd.DataFrame({"class": classes, "support": support, "H0_precision": precision_h0, "H0_recall": recall_h0, "H0_f1": class_h0, "OVR_precision": precision_ovr, "OVR_recall": recall_ovr, "OVR_f1": class_ovr, "f1_delta": class_ovr - class_h0})
    folds, audits = pd.DataFrame(fold_rows), pd.DataFrame(audit_rows)
    summary = pd.DataFrame([
        {"variant": "H0_selective_EB", "oof_macro_f1": h0_f1, "oof_accuracy": h0_accuracy, "feature_count": float(folds.query("variant == 'H0_selective_EB'").feature_count.mean()), "convergence_warning_count": warning_count, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started, "delta_vs_h0": 0.0},
        {"variant": "OVR_EB_LR", "oof_macro_f1": ovr_f1, "oof_accuracy": ovr_accuracy, "feature_count": float(folds.query("variant == 'OVR_EB_LR'").feature_count.mean()), "convergence_warning_count": warning_count, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started, "delta_vs_h0": ovr_f1 - h0_f1},
    ])
    diversity = {"run_id": run_id, "h0_reference": H0_REFERENCE, "h0_reference_match": abs(h0_f1 - H0_REFERENCE) <= .001, "h0_reference_delta": h0_f1 - H0_REFERENCE, "test_read": False, "train_test_concat": False, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "hard_prediction_disagreement_rate": float(disagreement.mean()), "h0_wrong_ovr_recovered_count": int(recovered.sum()), "h0_wrong_ovr_recovered_rate_of_all_rows": float(recovered.mean()), "h0_correct_ovr_broken_count": int(broken.sum()), "h0_correct_ovr_broken_rate_of_all_rows": float(broken.mean()), "joint_wrong_count": int(both_wrong.sum()), "oracle_macro_f1_diagnostic_only": oracle_f1, "ovr_diversity_candidate": bool(recovered.sum() > broken.sum() and recovered.mean() >= .05)}
    RESULT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT / f"{run_id}_seed42_summary.csv", index=False)
    folds.to_csv(RESULT / f"{run_id}_seed42_fold_metrics.csv", index=False)
    class_frame.to_csv(RESULT / f"{run_id}_seed42_class_metrics.csv", index=False)
    audits.to_csv(RESULT / f"{run_id}_seed42_fold_audit.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(train)), "truth": labels, "h0_prediction": h0_prediction, "ovr_prediction": ovr_prediction, "h0_correct": h0_correct, "ovr_correct": ovr_correct, "recovered_by_ovr": recovered, "broken_by_ovr": broken, "disagreement": disagreement}).to_csv(RESULT / f"{run_id}_seed42_error_overlap.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(train)), "truth": labels, **{f"h0__{label}": h0_oof[:, index] for index, label in enumerate(classes)}, **{f"ovr__{label}": ovr_oof[:, index] for index, label in enumerate(classes)}}).to_csv(RESULT / f"{run_id}_seed42_oof_probabilities.csv", index=False)
    (RESULT / f"{run_id}_seed42_diversity_audit.json").write_text(json.dumps(diversity, ensure_ascii=False, indent=2), encoding="utf-8")
    pivot = folds.pivot(index="fold", columns="variant", values="macro_f1")
    ax = pivot.plot(marker="o", title="H0 vs OVR EB LR"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(RESULT / f"{run_id}_seed42_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    ax = class_frame.sort_values("f1_delta").set_index("class").f1_delta.plot.barh(title="OVR class F1 delta vs H0"); ax.figure.tight_layout(); ax.figure.savefig(RESULT / f"{run_id}_seed42_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(diversity, ensure_ascii=False), flush=True)


def smoke() -> None:
    train = pd.read_csv(TRAIN_CSV, nrows=16)
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    assert len(genes) == 4384 and int(train[genes].isna().sum().sum()) == 0
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "ovr_n_jobs": 1}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-h0-ovr-diversity-audit-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id)
