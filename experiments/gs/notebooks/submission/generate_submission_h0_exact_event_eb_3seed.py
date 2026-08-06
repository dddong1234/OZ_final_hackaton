"""Generate the accepted H0 + automatic exact-event EB 3-seed submission.

The three validated model seeds are fitted independently on the complete
training set.  Every learned artifact (event vocabulary, EB statistics,
standardisation, recurrent features and automatic specialist pairs) is fitted
from ``train.csv`` only.  ``test.csv`` is used only to apply those fitted
transformations and predict.

This file deliberately imports only other code under ``experiments/gs``.
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

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve()
VALIDATED_SEEDS = (42, 777, 2024)
OUTPUT_NAME = "submission_h0_exact_event_eb_seed42_777_2024_bagged.csv"


def project_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "data" / "raw" / "train.csv").exists():
            return candidate
    raise FileNotFoundError("data/raw/train.csv was not found")


def _add_gs_common(relative: str) -> None:
    path = project_root() / "experiments" / "gs" / "notebooks" / relative
    if not path.exists():
        raise FileNotFoundError(f"GS common path missing: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_add_gs_common("exp_model_006/common")
_add_gs_common("exp_model_007/common")

from h0_faithful_pipeline import _aligned_probability, _hard_specialist, fit_vocabulary, transform_rows, build_design_matrices  # noqa: E402
from h0_selective_eb_replacement import (  # noqa: E402
    EB_ALPHA, EB_CLIP, EB_SHRINKAGE, H0_SPECIALIST_WEIGHT, SELECTIVE_LR_WEIGHT,
    SELECTIVE_MARGIN, fixed_branch_replacement, selective_probability,
)


@dataclass(frozen=True)
class ExactEBState:
    """Train-fitted posterior log-odds for all observed exact events."""

    selected: np.ndarray
    weights: np.ndarray


def fit_exact_eb(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> ExactEBState:
    """Fit class evidence without any fixed genes, variants, or support cutoff."""
    matrix = matrix.tocsr()
    support = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    selected = np.flatnonzero((support > 0) & (support < matrix.shape[0]))
    if not len(selected):
        return ExactEBState(selected, np.zeros((len(classes), 0), dtype=np.float32))
    selected_matrix = matrix[:, selected]
    support = support[selected]
    global_prior = (support + EB_ALPHA) / (len(labels) + 2.0 * EB_ALPHA)
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, label in enumerate(classes):
        positive_mask = labels == label
        positive = np.asarray(selected_matrix[positive_mask].getnnz(axis=0)).ravel().astype(np.float64)
        negative = support - positive
        positive_rate = (positive + EB_SHRINKAGE * global_prior) / (positive_mask.sum() + EB_SHRINKAGE)
        negative_rate = (negative + EB_SHRINKAGE * global_prior) / ((~positive_mask).sum() + EB_SHRINKAGE)
        positive_rate = np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
        negative_rate = np.clip(negative_rate, 1e-6, 1.0 - 1e-6)
        weights[class_index] = np.log(positive_rate / (1.0 - positive_rate)) - np.log(negative_rate / (1.0 - negative_rate))
    return ExactEBState(selected, np.clip(weights, -EB_CLIP, EB_CLIP).astype(np.float32))


def apply_exact_eb(matrix: sparse.csr_matrix, state: ExactEBState, class_count: int) -> np.ndarray:
    if not len(state.selected):
        return np.zeros((matrix.shape[0], class_count), dtype=np.float32)
    selected = matrix[:, state.selected]
    evidence = np.asarray(selected @ state.weights.T, dtype=np.float32)
    scale = np.sqrt(np.maximum(np.asarray(selected.getnnz(axis=1)).ravel(), 1.0))
    return evidence / scale[:, None]


def cross_fitted_exact_eb(
    train_exact: sparse.csr_matrix, apply_exact: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Make training EB scores by inner OOF; apply scores use full train only."""
    oof_scores = np.zeros((train_exact.shape[0], len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_index, valid_index in splitter.split(np.zeros(len(labels)), labels):
        state = fit_exact_eb(train_exact[fit_index], labels[fit_index], classes)
        oof_scores[valid_index] = apply_exact_eb(train_exact[valid_index], state, len(classes))
    final_state = fit_exact_eb(train_exact, labels, classes)
    transformed = apply_exact_eb(apply_exact, final_state, len(classes))
    mean = oof_scores.mean(axis=0, keepdims=True)
    std = np.maximum(oof_scores.std(axis=0, keepdims=True), 1e-6)
    return ((oof_scores - mean) / std).astype(np.float32), ((transformed - mean) / std).astype(np.float32)


def exact_eb_features(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame, labels: np.ndarray, classes: np.ndarray, genes: list[str], seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Train-only exact-event vocabulary; test is only projected into it."""
    vocabulary = fit_vocabulary(train_frame, genes)
    train_parsed = transform_rows(train_frame, genes, vocabulary)
    test_parsed = transform_rows(test_frame, genes, vocabulary)
    scores_train, scores_test = cross_fitted_exact_eb(train_parsed.exact, test_parsed.exact, labels, classes, seed)
    return scores_train, scores_test, int(train_parsed.exact.shape[1])


def fit_lr_probability(
    x_train: sparse.csr_matrix, labels: np.ndarray, x_test: sparse.csr_matrix, classes: np.ndarray, model_seed: int,
) -> tuple[np.ndarray, int]:
    model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=model_seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, labels)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return _aligned_probability(model, model.predict_proba(x_test), classes).astype(np.float32), int(warning_count)


def average_seed_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    if not probabilities:
        raise ValueError("at least one seed probability matrix is required")
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in probabilities]
    expected_shape = matrices[0].shape
    if len(expected_shape) != 2 or any(matrix.shape != expected_shape for matrix in matrices):
        raise ValueError("seed probability shapes differ")
    if any(not np.isfinite(matrix).all() or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6) for matrix in matrices):
        raise ValueError("seed probabilities must be finite and normalized")
    return np.mean(matrices, axis=0, dtype=np.float64).astype(np.float32)


def build_exact_probability(train: pd.DataFrame, test: pd.DataFrame, model_seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit the accepted exact-event replacement branch and automatic specialist."""
    if "ID" not in train or "SUBCLASS" not in train or "ID" not in test:
        raise ValueError("train/test schema must contain ID and train must contain SUBCLASS")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if list(test.columns) != ["ID", *genes]:
        raise ValueError("test gene columns must exactly match train order")
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train no-NaN contract failed")

    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    train_frame = train.loc[:, genes]
    test_frame = test.loc[:, genes]

    print(f"[exact-event H0] seed {model_seed}: train-only structured features", flush=True)
    x_train, x_test, names, design_audit = build_design_matrices(train_frame, test_frame, labels, genes, seed=model_seed)
    non_eb_probability, non_eb_warning = fit_lr_probability(x_train, labels, x_test, classes, model_seed)

    print(f"[exact-event H0] seed {model_seed}: train-only exact-event EB", flush=True)
    exact_train, exact_test, exact_vocabulary_size = exact_eb_features(train_frame, test_frame, labels, classes, genes, model_seed)
    augmented_train = sparse.hstack([x_train, sparse.csr_matrix(exact_train)], format="csr")
    augmented_test = sparse.hstack([x_test, sparse.csr_matrix(exact_test)], format="csr")
    exact_probability, exact_warning = fit_lr_probability(augmented_train, labels, augmented_test, classes, model_seed)
    gated_exact_probability, use_non_eb = selective_probability(non_eb_probability, exact_probability)

    print(f"[exact-event H0] seed {model_seed}: train-only automatic LGBM specialist", flush=True)
    lgbm = LGBMClassifier(
        objective="multiclass", boosting_type="gbdt", num_class=len(classes),
        n_estimators=400, learning_rate=0.05, num_leaves=25, min_child_samples=10,
        min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced",
        random_state=model_seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    lgbm.fit(x_train, labels)
    lgbm_probability = _aligned_probability(lgbm, lgbm.predict_proba(x_test), classes)
    specialist_probability, specialist_pairs = _hard_specialist(x_train, labels, x_test, lgbm_probability, classes, names, model_seed)
    probability = fixed_branch_replacement(gated_exact_probability, specialist_probability)

    audit = {
        "model_seed": int(model_seed),
        "seed_weight": 1.0 / len(VALIDATED_SEEDS),
        "lr_weight": SELECTIVE_LR_WEIGHT,
        "specialist_weight": H0_SPECIALIST_WEIGHT,
        "selective_margin": SELECTIVE_MARGIN,
        "threshold_retuned": False,
        "test_role": "transform_and_predict_only",
        "test_read_for_fit_statistics_selection_or_scaling": False,
        "raw_train_test_concat": False,
        "vocabulary_source": "full_train_only",
        "exact_event_vocabulary_source": "full_train_only",
        "exact_event_support_cutoff": None,
        "specialist_pair_source": "full_train_only_automatic_discovery",
        "fixed_cancer_gene_exact_mutation_rules": False,
        "nan_as_mutation_count": int(design_audit["nan_as_mutation_count"]),
        "leakage_check": bool(not design_audit["raw_train_test_concat"]),
        "structured_feature_count": int(x_train.shape[1]),
        "exact_eb_feature_count": int(exact_train.shape[1]),
        "exact_vocabulary_size": int(exact_vocabulary_size),
        "final_feature_count": int(augmented_train.shape[1]),
        "specialist_pairs": [list(pair) for pair in specialist_pairs],
        "selective_non_eb_test_rows": int(use_non_eb.sum()),
        "convergence_warning_count": int(non_eb_warning + exact_warning),
    }
    if audit["nan_as_mutation_count"] != 0 or not audit["leakage_check"]:
        raise AssertionError("submission safety contract failed")
    del lgbm, x_train, x_test, augmented_train, augmented_test, exact_train, exact_test
    gc.collect()
    return probability, classes, audit


def make_submission(sample: pd.DataFrame, test: pd.DataFrame, probability: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    if list(sample.columns) != ["ID", "SUBCLASS"]:
        raise ValueError("sample_submission must have exactly ID and SUBCLASS")
    if not sample.ID.reset_index(drop=True).equals(test.ID.reset_index(drop=True)):
        raise ValueError("sample submission IDs and test IDs are not aligned")
    if probability.shape != (len(test), len(classes)) or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("final probability matrix is invalid")
    submission = sample.copy()
    submission["SUBCLASS"] = classes[probability.argmax(axis=1)]
    return submission


def run_seed_bagged(output_name: str = OUTPUT_NAME, seeds: tuple[int, ...] = VALIDATED_SEEDS) -> Path:
    """Generate one Dacon CSV from the fixed validated 3-seed configuration."""
    if tuple(seeds) != VALIDATED_SEEDS:
        raise ValueError("the validated seed contract is exactly (42, 777, 2024)")
    started = perf_counter()
    root = project_root()
    raw = root / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    sample = pd.read_csv(raw / "sample_submission.csv")
    probabilities: list[np.ndarray] = []
    audits: list[dict] = []
    classes: np.ndarray | None = None
    for seed in seeds:
        probability, current_classes, audit = build_exact_probability(train, test, seed)
        if classes is None:
            classes = current_classes
        elif not np.array_equal(classes, current_classes):
            raise AssertionError("class order changed between seed fits")
        probabilities.append(probability)
        audits.append(audit)
    assert classes is not None
    averaged = average_seed_probabilities(probabilities)
    submission = make_submission(sample, test, averaged, classes)
    destination = root / "experiments" / "gs" / "notebooks" / "submission" / output_name
    submission.to_csv(destination, index=False)
    audit = {
        "run_id": "submission-h0-exact-event-eb-3seed-bagging",
        "seeds": list(seeds),
        "seed_weights": [1.0 / len(seeds)] * len(seeds),
        "weights_tuned": False,
        "test_role": "transform_and_predict_only",
        "raw_train_test_concat": False,
        "leakage_check": bool(all(item["leakage_check"] for item in audits)),
        "nan_as_mutation_count": int(max(item["nan_as_mutation_count"] for item in audits)),
        "per_seed_audits": audits,
        "output_file": str(destination),
        "row_count": int(len(submission)),
        "runtime_seconds": perf_counter() - started,
    }
    audit_path = destination.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not pd.read_csv(destination).equals(submission) or not audit["leakage_check"] or audit["nan_as_mutation_count"] != 0:
        raise AssertionError("submission safety validation failed")
    print(json.dumps({"submission": str(destination), "audit": str(audit_path), "rows": len(submission), "leakage_check": True, "nan_as_mutation_count": 0}, ensure_ascii=False), flush=True)
    return destination


def smoke() -> dict:
    """Run only parser/schema contracts; intentionally never opens test.csv."""
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv", nrows=32)
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if len(genes) != 4384 or int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train schema or NaN contract failed")
    labels = np.asarray(["A", "A", "B", "B", "C", "C"])
    exact = sparse.csr_matrix(np.asarray([[1, 0], [1, 1], [0, 1], [0, 1], [1, 0], [0, 0]], dtype=np.float32))
    state = fit_exact_eb(exact, labels, np.asarray(["A", "B", "C"]))
    if len(state.selected) != 2 or apply_exact_eb(exact, state, 3).shape != (6, 3):
        raise AssertionError("exact EB unit contract failed")
    averaged = average_seed_probabilities([np.array([[0.8, 0.2]], dtype=np.float32)] * 3)
    return {"test_role": "not_read", "nan_as_mutation_count": 0, "seed_contract": list(VALIDATED_SEEDS), "averaged_probability": averaged.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    arguments = parser.parse_args()
    if arguments.smoke:
        print(json.dumps(smoke(), ensure_ascii=False), flush=True)
    else:
        run_seed_bagged(output_name=arguments.output_name)
