"""Final train-only submission pipeline for the accepted H0 Selective-EB model.

This module intentionally uses no fixed cancer, gene, or mutation identifiers.
All vocabularies, recurrent events, Empirical-Bayes weights, normalisation, and
specialist pairs are fitted from the full training data only.  Test is used only
to apply those fitted transformations and produce predictions.
"""
from __future__ import annotations

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

from h0_selective_eb_replacement import (
    H0_SPECIALIST_WEIGHT,
    SELECTIVE_LR_WEIGHT,
    SELECTIVE_MARGIN,
    empirical_bayes_features,
    fixed_branch_replacement,
    selective_probability,
)

HERE = Path(__file__).resolve()
RUN_ID = "submission-h0-selective-eb-lr-lgbm-specialist"
MODEL_SEED = 42


def project_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "data" / "raw" / "train.csv").exists():
            return candidate
    raise FileNotFoundError("data/raw/train.csv was not found")


def submission_directory() -> Path:
    root = project_root()
    path = root / "experiments" / "gs" / "notebooks" / "submission"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _h0_common() -> Path:
    path = HERE.parents[2] / "exp_model_006" / "common"
    if not path.exists():
        raise FileNotFoundError("GS faithful H0 source was not found")
    return path


if str(_h0_common()) not in sys.path:
    sys.path.insert(0, str(_h0_common()))

from h0_faithful_pipeline import _aligned_probability, _hard_specialist, build_design_matrices  # noqa: E402


def make_submission_frame(
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    probability: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    """Validate sample order and write only the required submission columns."""
    required = ["ID", "SUBCLASS"]
    if list(sample_submission.columns) != required:
        raise ValueError("sample_submission must have exactly ID and SUBCLASS columns")
    if "ID" not in test or not sample_submission.ID.reset_index(drop=True).equals(test.ID.reset_index(drop=True)):
        raise ValueError("sample_submission ID order must match test ID order")
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (len(test), len(classes)):
        raise ValueError("probability shape does not match test rows and train classes")
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("final probability rows must be finite and normalized")
    output = sample_submission.loc[:, required].copy()
    output["SUBCLASS"] = np.asarray(classes, dtype=object)[probability.argmax(axis=1)]
    if output.SUBCLASS.isna().any() or not set(output.SUBCLASS).issubset(set(classes)):
        raise ValueError("submission contains an invalid predicted class")
    return output


def average_seed_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    """Equal-average predeclared full-train seed probabilities."""
    if not probabilities:
        raise ValueError("at least one seed probability matrix is required")
    arrays = [np.asarray(item, dtype=np.float64) for item in probabilities]
    shape = arrays[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in arrays):
        raise ValueError("all seed probability matrices must share shape")
    if any(not np.allclose(item.sum(axis=1), 1.0, atol=1e-6) for item in arrays):
        raise ValueError("each seed probability matrix must contain normalized rows")
    average = np.mean(arrays, axis=0)
    if not np.allclose(average.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("equal seed average must preserve probability rows")
    return average.astype(np.float32)


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
    return _aligned_probability(model, model.predict_proba(x_test), classes).astype(np.float32), int(warning_count)


def build_submission_probability(train: pd.DataFrame, test: pd.DataFrame, *, model_seed: int = MODEL_SEED) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit the accepted full-train model and apply it to test exactly once."""
    if "ID" not in train or "SUBCLASS" not in train or "ID" not in test:
        raise ValueError("train/test schema must include ID and train must include SUBCLASS")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if list(test.columns) != ["ID", *genes]:
        raise ValueError("test gene columns must exactly match train gene order")
    if int(train[genes].isna().sum().sum()) != 0:
        raise ValueError("training gene matrix violates the no-NaN contract")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    train_frame, test_frame = train.loc[:, genes], test.loc[:, genes]

    print("[submission] fit train-only structured mutation features", flush=True)
    x_train, x_test, names, structured_audit = build_design_matrices(
        train_frame, test_frame, labels, genes, seed=model_seed,
    )
    print("[submission] fit H0 multinomial Logistic Regression", flush=True)
    non_eb_probability, non_eb_warnings = _fit_lr_probability(x_train, labels, x_test, classes, model_seed)

    print("[submission] fit train-only Empirical-Bayes evidence branch", flush=True)
    eb_train, eb_test = empirical_bayes_features(
        train_frame, test_frame, labels, classes, genes, seed=model_seed,
    )
    x_train_eb = sparse.hstack([x_train, sparse.csr_matrix(eb_train)], format="csr")
    x_test_eb = sparse.hstack([x_test, sparse.csr_matrix(eb_test)], format="csr")
    eb_probability, eb_warnings = _fit_lr_probability(x_train_eb, labels, x_test_eb, classes, model_seed)
    selective_lr_probability, use_non_eb = selective_probability(non_eb_probability, eb_probability)

    print("[submission] fit full-train LGBM and automatic two-pair specialist", flush=True)
    lgbm = LGBMClassifier(
        objective="multiclass", boosting_type="gbdt", num_class=len(classes),
        n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10,
        min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced",
        random_state=model_seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    lgbm.fit(x_train, labels)
    lgbm_probability = _aligned_probability(lgbm, lgbm.predict_proba(x_test), classes)
    specialist_probability, specialist_pairs = _hard_specialist(
        x_train, labels, x_test, lgbm_probability, classes, names, model_seed,
    )
    final_probability = fixed_branch_replacement(selective_lr_probability, specialist_probability)
    audit = {
        "run_id": RUN_ID,
        "model_seed": model_seed,
        "lr_weight": SELECTIVE_LR_WEIGHT,
        "specialist_weight": H0_SPECIALIST_WEIGHT,
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
        "convergence_warning_count": int(non_eb_warnings + eb_warnings),
    }
    del lgbm, x_train_eb, x_test_eb
    gc.collect()
    if audit["nan_as_mutation_count"] != 0 or not audit["leakage_check"]:
        raise AssertionError("submission safety contract failed")
    return final_probability, classes, audit


def run(output_name: str = "submission_h0_selective_eb_lr_lgbm_specialist_seed42.csv") -> Path:
    started = perf_counter()
    root = project_root()
    raw = root / "data" / "raw"
    print("[submission] read train, test, and sample submission separately", flush=True)
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    sample = pd.read_csv(raw / "sample_submission.csv")
    probability, classes, audit = build_submission_probability(train, test, model_seed=MODEL_SEED)
    submission = make_submission_frame(sample, test, probability, classes)
    destination = submission_directory() / output_name
    submission.to_csv(destination, index=False)
    audit.update({"output_file": str(destination), "row_count": int(len(submission)), "runtime_seconds": perf_counter() - started})
    audit_path = destination.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    reloaded = pd.read_csv(destination)
    if not reloaded.equals(submission):
        raise AssertionError("submission round-trip validation failed")
    print(json.dumps({"submission": str(destination), "audit": str(audit_path), "rows": len(submission), "leakage_check": audit["leakage_check"], "nan_as_mutation_count": audit["nan_as_mutation_count"]}, ensure_ascii=False), flush=True)
    return destination


def run_seed_bagged(
    output_name: str = "submission_h0_selective_eb_lr_lgbm_specialist_seed42_777_2024_bagged.csv",
    seeds: tuple[int, ...] = (42, 777, 2024),
) -> Path:
    """Fit each predeclared seed on full train and equally average test probabilities."""
    if tuple(seeds) != (42, 777, 2024):
        raise ValueError("the validated seed-bagging contract is exactly (42, 777, 2024)")
    started = perf_counter(); root = project_root(); raw = root / "data" / "raw"
    print("[submission] read train, test, and sample submission separately", flush=True)
    train = pd.read_csv(raw / "train.csv"); test = pd.read_csv(raw / "test.csv"); sample = pd.read_csv(raw / "sample_submission.csv")
    probability_rows, audits, classes = [], [], None
    for seed in seeds:
        print(f"[submission] full-train seed {seed}", flush=True)
        probability, current_classes, audit = build_submission_probability(train, test, model_seed=seed)
        if classes is None:
            classes = current_classes
        elif not np.array_equal(classes, current_classes):
            raise AssertionError("class order differs between seed fits")
        probability_rows.append(probability); audits.append(audit)
    averaged = average_seed_probabilities(probability_rows)
    submission = make_submission_frame(sample, test, averaged, classes)
    destination = submission_directory() / output_name; submission.to_csv(destination, index=False)
    audit = {
        "run_id": RUN_ID + "-seed-bagging",
        "seeds": list(seeds),
        "seed_weights": [1.0 / len(seeds)] * len(seeds),
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
    audit_path = destination.with_suffix(".audit.json"); audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not pd.read_csv(destination).equals(submission) or not audit["leakage_check"] or audit["nan_as_mutation_count"] != 0:
        raise AssertionError("seed-bagged submission safety validation failed")
    print(json.dumps({"submission": str(destination), "audit": str(audit_path), "rows": len(submission), "leakage_check": audit["leakage_check"], "nan_as_mutation_count": audit["nan_as_mutation_count"]}, ensure_ascii=False), flush=True)
    return destination


if __name__ == "__main__":
    run()
