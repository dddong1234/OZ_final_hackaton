"""Create the fixed equal H0 / non-EB 3-seed Dacon submission.

For every validated seed, this runner fits all feature construction, Empirical-
Bayes evidence, and automatic LGBM-specialist selection on full train only.
Test is used solely for the resulting transformations and prediction.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve()
VALIDATED_SEEDS = (42, 777, 2024)
OUTPUT_NAME = "submission_equal_h0_non_eb_seed42_777_2024_bagged.csv"


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


_add_gs_common("exp_model_007/common")
_add_gs_common("exp_model_006/common")

from h0_selective_eb_replacement import (  # noqa: E402
    H0_SPECIALIST_WEIGHT,
    SELECTIVE_LR_WEIGHT,
    SELECTIVE_MARGIN,
    empirical_bayes_features,
    fixed_branch_replacement,
    selective_probability,
)
from h0_faithful_pipeline import _aligned_probability, _hard_specialist, build_design_matrices  # noqa: E402


def equal_probability(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the fixed 0.50/0.50 normalized probability blend."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("probability shape mismatch")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("probabilities must be finite")
    blended = 0.5 * left + 0.5 * right
    if np.any(blended.sum(axis=1) <= 0):
        raise ValueError("probability rows must have positive mass")
    return (blended / blended.sum(axis=1, keepdims=True)).astype(np.float32)


def average_seed_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    """Equal-average validated per-seed test probabilities."""
    if not probabilities:
        raise ValueError("at least one seed probability matrix is required")
    arrays = [np.asarray(item, dtype=np.float64) for item in probabilities]
    shape = arrays[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in arrays):
        raise ValueError("all seed probabilities must have the same 2D shape")
    if any(not np.isfinite(item).all() or not np.allclose(item.sum(axis=1), 1.0, atol=1e-6) for item in arrays):
        raise ValueError("each seed probability must be finite and normalized")
    return np.mean(arrays, axis=0, dtype=np.float64).astype(np.float32)


def _fit_lr_probability(
    x_train: sparse.csr_matrix,
    labels: np.ndarray,
    x_test: sparse.csr_matrix,
    classes: np.ndarray,
    model_seed: int,
) -> tuple[np.ndarray, int]:
    model = LogisticRegression(
        solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=model_seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, labels)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    probability = _aligned_probability(model, model.predict_proba(x_test), classes).astype(np.float32)
    return probability, int(warning_count)


def build_equal_probability(train: pd.DataFrame, test: pd.DataFrame, model_seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit both fixed H0 branches on train and apply them to test once."""
    if "ID" not in train or "SUBCLASS" not in train or "ID" not in test:
        raise ValueError("train/test schema must contain ID and train must contain SUBCLASS")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if list(test.columns) != ["ID", *genes]:
        raise ValueError("test gene columns must match train order exactly")
    if int(train[genes].isna().sum().sum()) != 0:
        raise ValueError("train violates the no-NaN mutation contract")

    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    train_frame, test_frame = train.loc[:, genes], test.loc[:, genes]

    print(f"[equal-H0] seed {model_seed}: train-only structured features", flush=True)
    x_train, x_test, names, structured_audit = build_design_matrices(
        train_frame, test_frame, labels, genes, seed=model_seed,
    )
    non_eb_probability, non_eb_warning = _fit_lr_probability(x_train, labels, x_test, classes, model_seed)

    print(f"[equal-H0] seed {model_seed}: train-only EB LR", flush=True)
    eb_train, eb_test = empirical_bayes_features(train_frame, test_frame, labels, classes, genes, seed=model_seed)
    x_train_eb = sparse.hstack([x_train, sparse.csr_matrix(eb_train)], format="csr")
    x_test_eb = sparse.hstack([x_test, sparse.csr_matrix(eb_test)], format="csr")
    eb_probability, eb_warning = _fit_lr_probability(x_train_eb, labels, x_test_eb, classes, model_seed)
    selective_lr_probability, use_non_eb = selective_probability(non_eb_probability, eb_probability)

    print(f"[equal-H0] seed {model_seed}: automatic specialist", flush=True)
    lgbm = LGBMClassifier(
        objective="multiclass", boosting_type="gbdt", num_class=len(classes),
        n_estimators=400, learning_rate=0.05, num_leaves=25, min_child_samples=10,
        min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced",
        random_state=model_seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    lgbm.fit(x_train, labels)
    lgbm_probability = _aligned_probability(lgbm, lgbm.predict_proba(x_test), classes)
    specialist_probability, specialist_pairs = _hard_specialist(
        x_train, labels, x_test, lgbm_probability, classes, names, model_seed,
    )
    h0_selective_eb = fixed_branch_replacement(selective_lr_probability, specialist_probability)
    h0_non_eb = fixed_branch_replacement(non_eb_probability, specialist_probability)
    probability = equal_probability(h0_selective_eb, h0_non_eb)

    audit = {
        "model_seed": int(model_seed),
        "equal_branch_weights": [0.5, 0.5],
        "lr_weight_per_branch": SELECTIVE_LR_WEIGHT,
        "specialist_weight_per_branch": H0_SPECIALIST_WEIGHT,
        "selective_margin": SELECTIVE_MARGIN,
        "threshold_retuned": False,
        "test_role": "transform_and_predict_only",
        "test_read_for_fit_statistics_selection_or_scaling": False,
        "raw_train_test_concat": False,
        "vocabulary_source": "full_train_only",
        "specialist_pair_source": "full_train_only_automatic_discovery",
        "fixed_cancer_gene_exact_mutation_rules": False,
        "nan_as_mutation_count": int(structured_audit["nan_as_mutation_count"]),
        "leakage_check": bool(not structured_audit["raw_train_test_concat"]),
        "structured_feature_count": int(x_train.shape[1]),
        "eb_feature_count": int(eb_train.shape[1]),
        "final_feature_count": int(x_train_eb.shape[1]),
        "specialist_pairs": [list(pair) for pair in specialist_pairs],
        "selective_non_eb_test_rows": int(use_non_eb.sum()),
        "convergence_warning_count": int(non_eb_warning + eb_warning),
    }
    del lgbm, x_train_eb, x_test_eb
    gc.collect()
    if audit["nan_as_mutation_count"] != 0 or not audit["leakage_check"]:
        raise AssertionError("submission safety contract failed")
    return probability, classes, audit


def _make_submission(sample: pd.DataFrame, test: pd.DataFrame, probability: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    if list(sample.columns) != ["ID", "SUBCLASS"]:
        raise ValueError("sample_submission must have exactly ID and SUBCLASS")
    if not sample.ID.reset_index(drop=True).equals(test.ID.reset_index(drop=True)):
        raise ValueError("sample submission IDs do not match test ID order")
    if probability.shape != (len(test), len(classes)) or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("final probability matrix is invalid")
    output = sample.copy()
    output["SUBCLASS"] = classes[probability.argmax(axis=1)]
    return output


def run_seed_bagged(output_name: str = OUTPUT_NAME, seeds: tuple[int, ...] = VALIDATED_SEEDS) -> Path:
    """Generate the fixed 3-seed equal-H0/non-EB final submission."""
    if tuple(seeds) != VALIDATED_SEEDS:
        raise ValueError("the validated seed contract is exactly (42, 777, 2024)")
    started = perf_counter()
    root = project_root()
    raw = root / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    sample = pd.read_csv(raw / "sample_submission.csv")
    per_seed, audits, classes = [], [], None
    for seed in seeds:
        probability, current_classes, audit = build_equal_probability(train, test, model_seed=seed)
        if classes is None:
            classes = current_classes
        elif not np.array_equal(classes, current_classes):
            raise AssertionError("class order changed between seed fits")
        per_seed.append(probability)
        audits.append(audit)
    averaged = average_seed_probabilities(per_seed)
    submission = _make_submission(sample, test, averaged, classes)
    destination = root / "experiments" / "gs" / "notebooks" / "submission" / output_name
    submission.to_csv(destination, index=False)
    audit = {
        "run_id": "submission-equal-h0-non-eb-3seed-bagging",
        "seeds": list(seeds),
        "seed_weights": [1.0 / len(seeds)] * len(seeds),
        "equal_branch_weights": [0.5, 0.5],
        "weight_tuned": False,
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
    """Verify static train-only and fixed-weight contracts without reading test."""
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv", nrows=32)
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if len(genes) != 4384 or int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train schema or NaN contract failed")
    averaged = average_seed_probabilities([np.array([[0.8, 0.2]], dtype=np.float32)] * 3)
    np.testing.assert_allclose(equal_probability(averaged, np.array([[0.2, 0.8]], dtype=np.float32)), [[0.5, 0.5]])
    return {"test_role": "not_read", "nan_as_mutation_count": 0, "seed_contract": list(VALIDATED_SEEDS), "equal_branch_weights": [0.5, 0.5]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), ensure_ascii=False), flush=True)
    else:
        run_seed_bagged(output_name=args.output_name)
