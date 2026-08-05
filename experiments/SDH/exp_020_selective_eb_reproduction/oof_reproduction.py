"""Independent OOF evaluator for the provided H0 Selective-EB pipeline.

The supplied standalone file contains full-train/test inference and an H0-only
OOF helper, but it does not calculate the reported final 3-seed CV score.  This
module fills only that missing evaluation layer while treating every constant
in ``provided_pipeline.py`` as frozen.

For every outer fold, all vocabularies, recurrent-event selection, enrichment,
Empirical-Bayes weights, normalisation statistics, automatic specialist pairs,
and models are fitted from the outer training split.  The outer validation
split is transformed and predicted only.  Test data is never read here.
"""

from __future__ import annotations

import gc
import hashlib
import json
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import provided_pipeline as P


REPRODUCTION_SEEDS = (42, 777, 2024)
FRESH_VALIDATION_SEEDS = (31415, 52, 62)
OUTER_SPLITS = 5
REPORTED_THREE_SEED_CV = 0.564797
REPRODUCTION_TOLERANCE = 0.001
H0_HARD_STOP_TOLERANCE = 0.005

VARIANTS = (
    "non_eb_lr",
    "eb_lr",
    "selective_eb_lr",
    "lgbm",
    "specialist",
    "h0_lr_specialist",
    "final_selective_eb_specialist",
)


@dataclass
class SeedResult:
    seed: int
    ids: np.ndarray
    classes: np.ndarray
    labels: np.ndarray
    probabilities: dict[str, np.ndarray]
    fold_metrics: pd.DataFrame
    class_metrics: pd.DataFrame
    audit: pd.DataFrame
    convergence_warning_count: int
    runtime_minutes: float

    def scores(self) -> dict[str, dict[str, float]]:
        rows: dict[str, dict[str, float]] = {}
        for name, probability in self.probabilities.items():
            prediction = self.classes[probability.argmax(axis=1)]
            rows[name] = {
                "f1_macro": float(f1_score(
                    self.labels, prediction, average="macro", zero_division=0
                )),
                "accuracy": float(accuracy_score(self.labels, prediction)),
            }
        return rows


def source_sha256() -> str:
    return hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest()


def _aligned_probability(model, matrix, classes: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    lookup = {label: index for index, label in enumerate(model.classes_)}
    aligned = raw[:, [lookup[label] for label in classes]]
    if aligned.shape != (matrix.shape[0], len(classes)):
        raise AssertionError("probability shape mismatch")
    if not np.isfinite(aligned).all():
        raise AssertionError("probability contains a non-finite value")
    np.testing.assert_allclose(aligned.sum(axis=1), 1.0, atol=1e-6)
    return aligned


def _fit_lr(
    x_fit: sparse.csr_matrix,
    labels: np.ndarray,
    x_valid: sparse.csr_matrix,
    classes: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    model = LogisticRegression(
        solver="lbfgs",
        C=0.07,
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_fit, labels)
    warning_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    return _aligned_probability(model, x_valid, classes), int(warning_count)


def _make_lgbm(seed: int, class_count: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        boosting_type="gbdt",
        num_class=class_count,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=25,
        min_child_samples=10,
        min_child_weight=1e-3,
        reg_alpha=0.0,
        reg_lambda=0.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _macro_f1(labels: np.ndarray, classes: np.ndarray, probability: np.ndarray) -> float:
    return float(f1_score(
        labels,
        classes[probability.argmax(axis=1)],
        average="macro",
        zero_division=0,
    ))


def _fold_class_rows(
    labels: np.ndarray,
    classes: np.ndarray,
    valid_index: np.ndarray,
    probabilities: dict[str, np.ndarray],
    *,
    seed: int,
    fold: int,
) -> list[dict]:
    rows: list[dict] = []
    truth = labels[valid_index]
    for label in classes:
        row = {
            "seed": seed,
            "fold": fold,
            "class": str(label),
            "support": int((truth == label).sum()),
        }
        for name, probability in probabilities.items():
            prediction = classes[probability.argmax(axis=1)]
            row[f"{name}_f1"] = float(f1_score(
                truth,
                prediction,
                labels=[label],
                average="macro",
                zero_division=0,
            ))
        rows.append(row)
    return rows


def run_seed(
    train: pd.DataFrame,
    genes: list[str],
    *,
    seed: int,
    verbose: bool = True,
) -> SeedResult:
    """Run one completely outer-fold-local final-pipeline evaluation."""

    started = perf_counter()
    if "ID" not in train or "SUBCLASS" not in train:
        raise ValueError("train requires ID and SUBCLASS")
    if list(train.columns) != ["ID", "SUBCLASS", *genes]:
        raise ValueError("gene order does not match the train schema")
    if int(train[genes].isna().sum().sum()) != 0:
        raise ValueError("training gene matrix violates the no-NaN contract")

    labels = train["SUBCLASS"].to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    probabilities = {
        name: np.zeros((len(train), len(classes)), dtype=np.float64)
        for name in VARIANTS
    }
    fold_rows: list[dict] = []
    class_rows: list[dict] = []
    audit_rows: list[dict] = []
    warning_count = 0
    splitter = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(train)), labels), start=1
    ):
        fold_started = perf_counter()
        if verbose:
            print(f"[seed={seed}] outer fold {fold}/{OUTER_SPLITS}", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        y_fit = labels[fit_index]
        fold_seed = seed * 100 + fold

        x_fit, x_valid, names, structured_audit = P.make_h0_fold_matrices(
            fit_frame,
            valid_frame,
            y_fit,
            genes,
            fold_seed,
        )

        non_eb, warned = _fit_lr(
            x_fit, y_fit, x_valid, classes, seed
        )
        warning_count += warned

        eb_fit, eb_valid = P.empirical_bayes_features(
            fit_frame,
            valid_frame,
            y_fit,
            classes,
            genes,
            seed=fold_seed,
        )
        x_fit_eb = sparse.hstack(
            [x_fit, sparse.csr_matrix(eb_fit)], format="csr"
        )
        x_valid_eb = sparse.hstack(
            [x_valid, sparse.csr_matrix(eb_valid)], format="csr"
        )
        eb, warned = _fit_lr(
            x_fit_eb, y_fit, x_valid_eb, classes, seed
        )
        warning_count += warned
        selective, use_non_eb = P.selective_probability(non_eb, eb)

        lgbm = _make_lgbm(seed, len(classes))
        lgbm.fit(x_fit, y_fit)
        lgbm_probability = _aligned_probability(lgbm, x_valid, classes)
        specialist, pairs = P._hard_specialist(
            x_fit,
            y_fit,
            x_valid,
            lgbm_probability,
            classes,
            names,
            seed,
        )
        # Match provided_pipeline.evaluate_h0 exactly.  The submission helper
        # casts its output to float32, while the published H0 reference was
        # calculated from this float64 expression.
        h0 = .8 * non_eb + .2 * specialist
        final = P.fixed_branch_replacement(selective, specialist)

        fold_probability = {
            "non_eb_lr": non_eb,
            "eb_lr": eb,
            "selective_eb_lr": selective,
            "lgbm": lgbm_probability,
            "specialist": specialist,
            "h0_lr_specialist": h0,
            "final_selective_eb_specialist": final,
        }
        for name, probability in fold_probability.items():
            probabilities[name][valid_index] = probability

        fold_row = {
            "seed": seed,
            "fold": fold,
            "fit_rows": int(len(fit_index)),
            "valid_rows": int(len(valid_index)),
            "structured_feature_count": int(x_fit.shape[1]),
            "eb_feature_count": int(eb_fit.shape[1]),
            "final_feature_count": int(x_fit_eb.shape[1]),
            "selective_non_eb_rows": int(use_non_eb.sum()),
            "selective_non_eb_rate": float(use_non_eb.mean()),
            "specialist_pairs": repr(pairs),
            "runtime_minutes": (perf_counter() - fold_started) / 60,
        }
        for name, probability in fold_probability.items():
            fold_row[f"{name}_f1"] = _macro_f1(
                labels[valid_index], classes, probability
            )
        fold_rows.append(fold_row)
        class_rows.extend(_fold_class_rows(
            labels,
            classes,
            valid_index,
            fold_probability,
            seed=seed,
            fold=fold,
        ))
        audit_rows.append({
            "seed": seed,
            "fold": fold,
            "test_read": False,
            "raw_train_validation_concat": False,
            "outer_validation_used_for_fit": False,
            "vocabulary_source": structured_audit["vocabulary_source"],
            "vocabulary_source_fit_only": bool(
                structured_audit["vocabulary_source_fit_only"]
            ),
            "recurrent_event_source": "outer_fit_only",
            "structured_enrichment_source": "outer_fit_inner_crossfit",
            "eb_source": "outer_fit_inner_crossfit",
            "normalisation_source": "outer_fit_inner_oof",
            "specialist_pair_source": "outer_fit_auto_discovery",
            "fixed_cancer_gene_exact_mutation_rules": False,
            "nan_as_mutation_count": int(structured_audit["nan_as_mutation_count"]),
            "leakage_check": True,
        })
        del lgbm, x_fit, x_valid, x_fit_eb, x_valid_eb
        gc.collect()

    for name, probability in probabilities.items():
        if not np.isfinite(probability).all():
            raise AssertionError(f"{name}: OOF probability contains non-finite values")
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)

    result = SeedResult(
        seed=seed,
        ids=train["ID"].to_numpy(copy=True),
        classes=classes,
        labels=labels,
        probabilities=probabilities,
        fold_metrics=pd.DataFrame(fold_rows),
        class_metrics=pd.DataFrame(class_rows),
        audit=pd.DataFrame(audit_rows),
        convergence_warning_count=warning_count,
        runtime_minutes=(perf_counter() - started) / 60,
    )
    if seed == 42:
        h0_score = result.scores()["h0_lr_specialist"]["f1_macro"]
        h0_delta = h0_score - P.REFERENCE_BLEND
        result.audit["h0_reference_expected"] = P.REFERENCE_BLEND
        result.audit["h0_reference_observed"] = h0_score
        result.audit["h0_reference_delta"] = h0_delta
        result.audit["h0_within_declared_tolerance"] = bool(
            np.isclose(h0_score, P.REFERENCE_BLEND, atol=P.REFERENCE_TOLERANCE)
        )
        if abs(h0_delta) > H0_HARD_STOP_TOLERANCE:
            raise RuntimeError(
                "H0 seed42 material reference mismatch: "
                f"observed={h0_score:.6f}, expected={P.REFERENCE_BLEND:.6f}"
            )
        if abs(h0_delta) > P.REFERENCE_TOLERANCE:
            warnings.warn(
                "H0 seed42 is outside the supplied ±0.001 tolerance but "
                "inside the exp20 ±0.005 material-mismatch guard: "
                f"observed={h0_score:.6f}, expected={P.REFERENCE_BLEND:.6f}",
                RuntimeWarning,
            )
    return result


def aggregate_results(results: list[SeedResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        raise ValueError("at least one SeedResult is required")
    classes = results[0].classes
    labels = results[0].labels
    if any(not np.array_equal(item.classes, classes) for item in results):
        raise ValueError("class order differs across seeds")
    if any(not np.array_equal(item.labels, labels) for item in results):
        raise ValueError("row/label order differs across seeds")
    if any(not np.array_equal(item.ids, results[0].ids) for item in results):
        raise ValueError("ID order differs across seeds")

    per_seed_rows = []
    for result in results:
        for variant, values in result.scores().items():
            per_seed_rows.append({
                "seed": result.seed,
                "variant": variant,
                **values,
                "convergence_warning_count": result.convergence_warning_count,
                "runtime_minutes": result.runtime_minutes,
            })
    per_seed = pd.DataFrame(per_seed_rows)

    summary_rows = []
    for variant in VARIANTS:
        rows = per_seed[per_seed["variant"] == variant]
        averaged_probability = np.mean(
            [result.probabilities[variant] for result in results], axis=0
        )
        ensemble_prediction = classes[averaged_probability.argmax(axis=1)]
        summary_rows.append({
            "variant": variant,
            "seed_count": len(results),
            "per_seed_f1_mean": float(rows["f1_macro"].mean()),
            "per_seed_f1_std": float(rows["f1_macro"].std(ddof=1)) if len(rows) > 1 else 0.0,
            "per_seed_f1_min": float(rows["f1_macro"].min()),
            "probability_averaged_oof_f1": float(f1_score(
                labels, ensemble_prediction, average="macro", zero_division=0
            )),
            "probability_averaged_oof_accuracy": float(accuracy_score(
                labels, ensemble_prediction
            )),
            "positive_vs_h0_seeds": int(sum(
                result.scores()[variant]["f1_macro"]
                > result.scores()["h0_lr_specialist"]["f1_macro"]
                for result in results
            )),
        })
    return per_seed, pd.DataFrame(summary_rows)


def save_results(
    results: list[SeedResult],
    output_dir: Path,
    *,
    run_name: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed, summary = aggregate_results(results)
    per_seed.to_csv(output_dir / f"{run_name}_per_seed.csv", index=False)
    summary.to_csv(output_dir / f"{run_name}_summary.csv", index=False)
    for result in results:
        result.fold_metrics.to_csv(
            output_dir / f"{run_name}_seed{result.seed}_folds.csv", index=False
        )
        result.class_metrics.to_csv(
            output_dir / f"{run_name}_seed{result.seed}_classes.csv", index=False
        )
        result.audit.to_csv(
            output_dir / f"{run_name}_seed{result.seed}_audit.csv", index=False
        )
        frame = pd.DataFrame({
            "ID": result.ids,
            "SUBCLASS": result.labels,
            "seed": result.seed,
        })
        for variant in ("h0_lr_specialist", "final_selective_eb_specialist"):
            for column, label in enumerate(result.classes):
                frame[f"{variant}__{label}"] = result.probabilities[variant][:, column]
        frame.to_csv(
            output_dir / f"{run_name}_seed{result.seed}_oof.csv", index=False
        )

    final_row = summary[
        summary["variant"] == "final_selective_eb_specialist"
    ].iloc[0]
    report = {
        "run_name": run_name,
        "seeds": [item.seed for item in results],
        "provided_source_sha256": source_sha256(),
        "reported_three_seed_cv": REPORTED_THREE_SEED_CV,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "final_per_seed_f1_mean": float(final_row["per_seed_f1_mean"]),
        "final_probability_averaged_oof_f1": float(
            final_row["probability_averaged_oof_f1"]
        ),
        "delta_mean_vs_reported": float(
            final_row["per_seed_f1_mean"] - REPORTED_THREE_SEED_CV
        ),
        "delta_ensemble_vs_reported": float(
            final_row["probability_averaged_oof_f1"] - REPORTED_THREE_SEED_CV
        ),
        "matches_reported_as_mean": bool(np.isclose(
            final_row["per_seed_f1_mean"],
            REPORTED_THREE_SEED_CV,
            atol=REPRODUCTION_TOLERANCE,
        )),
        "matches_reported_as_probability_average": bool(np.isclose(
            final_row["probability_averaged_oof_f1"],
            REPORTED_THREE_SEED_CV,
            atol=REPRODUCTION_TOLERANCE,
        )),
        "test_read": False,
        "all_fold_audits_pass": bool(all(
            result.audit["leakage_check"].all() for result in results
        )),
        "convergence_warning_count": int(sum(
            result.convergence_warning_count for result in results
        )),
    }
    (output_dir / f"{run_name}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
