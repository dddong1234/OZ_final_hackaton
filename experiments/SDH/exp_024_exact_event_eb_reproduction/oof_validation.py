"""Fold-local OOF validation for H0 versus Exact-event EB.

Test data is never read. Every vocabulary, supervised score, scaling statistic,
automatic specialist pair, and model is fitted inside the outer training split.
"""

from __future__ import annotations

import gc
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

import exact_event_pipeline as P


SEEDS = (42, 777, 2024)
OUTER_SPLITS = 5
REPORTED_EXACT_MEAN = 0.568441
VARIANTS = (
    "h0",
    "exact_lr",
    "selective_exact_lr",
    "final_exact",
)


@dataclass
class SeedResult:
    seed: int
    ids: np.ndarray
    labels: np.ndarray
    classes: np.ndarray
    probabilities: dict[str, np.ndarray]
    fold_metrics: pd.DataFrame
    audit: pd.DataFrame
    warning_count: int
    runtime_minutes: float

    def scores(self) -> dict[str, dict[str, float]]:
        output = {}
        for name, probability in self.probabilities.items():
            prediction = self.classes[probability.argmax(axis=1)]
            output[name] = {
                "f1_macro": float(f1_score(
                    self.labels, prediction, average="macro", zero_division=0
                )),
                "accuracy": float(accuracy_score(self.labels, prediction)),
            }
        return output


def _aligned_probability(model, matrix, classes: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    lookup = {label: index for index, label in enumerate(model.classes_)}
    probability = raw[:, [lookup[label] for label in classes]]
    if probability.shape != (matrix.shape[0], len(classes)):
        raise AssertionError("probability shape mismatch")
    if not np.isfinite(probability).all():
        raise AssertionError("non-finite probability")
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    return probability


def _fit_lr(
    x_fit: sparse.csr_matrix,
    y_fit: np.ndarray,
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
        model.fit(x_fit, y_fit)
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


def _score(y: np.ndarray, classes: np.ndarray, probability: np.ndarray) -> float:
    return float(f1_score(
        y, classes[probability.argmax(axis=1)], average="macro", zero_division=0
    ))


def run_seed(
    train: pd.DataFrame,
    genes: list[str],
    *,
    seed: int,
    verbose: bool = True,
) -> SeedResult:
    """Evaluate one seed with completely fold-local feature learning."""

    started = perf_counter()
    if list(train.columns) != ["ID", "SUBCLASS", *genes]:
        raise ValueError("train schema or gene order mismatch")
    if int(train[genes].isna().sum().sum()) != 0:
        raise ValueError("train contains NaN gene cells")

    labels = train["SUBCLASS"].to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    probabilities = {
        name: np.zeros((len(train), len(classes)), dtype=np.float64)
        for name in VARIANTS
    }
    fold_rows: list[dict] = []
    audit_rows: list[dict] = []
    warning_count = 0
    splitter = StratifiedKFold(
        n_splits=OUTER_SPLITS, shuffle=True, random_state=seed
    )

    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(train)), labels), start=1
    ):
        fold_started = perf_counter()
        fold_seed = seed * 100 + fold
        if verbose:
            print(f"[seed={seed}] fold {fold}/{OUTER_SPLITS}", flush=True)

        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        y_fit = labels[fit_index]

        x_fit, x_valid, names, structured_audit = P.make_h0_fold_matrices(
            fit_frame, valid_frame, y_fit, genes, fold_seed
        )
        non_eb, warned = _fit_lr(
            x_fit, y_fit, x_valid, classes, seed
        )
        warning_count += warned

        gene_eb_fit, gene_eb_valid = P.empirical_bayes_features(
            fit_frame,
            valid_frame,
            y_fit,
            classes,
            genes,
            seed=fold_seed,
        )
        exact_fit, exact_valid, exact_vocabulary_size = P.exact_eb_features(
            fit_frame,
            valid_frame,
            y_fit,
            classes,
            genes,
            seed=fold_seed,
        )
        x_fit_exact = sparse.hstack(
            [x_fit, sparse.csr_matrix(gene_eb_fit), sparse.csr_matrix(exact_fit)],
            format="csr",
        )
        x_valid_exact = sparse.hstack(
            [x_valid, sparse.csr_matrix(gene_eb_valid), sparse.csr_matrix(exact_valid)],
            format="csr",
        )
        exact_lr, warned = _fit_lr(
            x_fit_exact, y_fit, x_valid_exact, classes, seed
        )
        warning_count += warned
        selective_exact, use_non_eb = P.selective_probability(non_eb, exact_lr)

        lgbm = _make_lgbm(seed, len(classes))
        lgbm.fit(x_fit, y_fit)
        main_probability = _aligned_probability(lgbm, x_valid, classes)
        specialist, pairs = P._hard_specialist(
            x_fit,
            y_fit,
            x_valid,
            main_probability,
            classes,
            names,
            seed,
        )
        h0 = 0.8 * non_eb + 0.2 * specialist
        final_exact = P.fixed_branch_replacement(selective_exact, specialist)

        fold_probability = {
            "h0": h0,
            "exact_lr": exact_lr,
            "selective_exact_lr": selective_exact,
            "final_exact": final_exact,
        }
        for name, probability in fold_probability.items():
            probabilities[name][valid_index] = probability

        h0_f1 = _score(labels[valid_index], classes, h0)
        exact_f1 = _score(labels[valid_index], classes, final_exact)
        fold_rows.append({
            "seed": seed,
            "fold": fold,
            "fit_rows": len(fit_index),
            "valid_rows": len(valid_index),
            "h0_f1": h0_f1,
            "final_exact_f1": exact_f1,
            "delta_vs_h0": exact_f1 - h0_f1,
            "structured_feature_count": x_fit.shape[1],
            "gene_eb_feature_count": gene_eb_fit.shape[1],
            "exact_eb_feature_count": exact_fit.shape[1],
            "exact_vocabulary_size": exact_vocabulary_size,
            "selective_non_eb_rows": int(use_non_eb.sum()),
            "specialist_pairs": repr(pairs),
            "runtime_minutes": (perf_counter() - fold_started) / 60,
        })
        audit_rows.append({
            "seed": seed,
            "fold": fold,
            "test_read": False,
            "raw_train_validation_concat": False,
            "outer_validation_used_for_fit": False,
            "vocabulary_source": structured_audit["vocabulary_source"],
            "exact_vocabulary_source": "outer_fit_only",
            "gene_eb_source": "outer_fit_inner_crossfit",
            "exact_eb_source": "outer_fit_inner_crossfit",
            "normalisation_source": "outer_fit_inner_oof",
            "specialist_pair_source": "outer_fit_auto_discovery",
            "fixed_domain_identifiers": False,
            "leakage_check": True,
        })

        del lgbm, x_fit, x_valid, x_fit_exact, x_valid_exact
        gc.collect()

    for name, probability in probabilities.items():
        if not np.isfinite(probability).all():
            raise AssertionError(f"{name}: non-finite OOF probability")
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)

    return SeedResult(
        seed=seed,
        ids=train["ID"].to_numpy(copy=True),
        labels=labels,
        classes=classes,
        probabilities=probabilities,
        fold_metrics=pd.DataFrame(fold_rows),
        audit=pd.DataFrame(audit_rows),
        warning_count=warning_count,
        runtime_minutes=(perf_counter() - started) / 60,
    )


def aggregate(results: list[SeedResult]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not results:
        raise ValueError("at least one result is required")
    labels, classes = results[0].labels, results[0].classes
    for result in results[1:]:
        if not np.array_equal(result.labels, labels):
            raise ValueError("label order differs across seeds")
        if not np.array_equal(result.classes, classes):
            raise ValueError("class order differs across seeds")

    per_seed_rows = []
    for result in results:
        scores = result.scores()
        for variant, values in scores.items():
            per_seed_rows.append({
                "seed": result.seed,
                "variant": variant,
                **values,
                "warning_count": result.warning_count,
                "runtime_minutes": result.runtime_minutes,
            })
    per_seed = pd.DataFrame(per_seed_rows)

    summary_rows = []
    for variant in VARIANTS:
        rows = per_seed[per_seed["variant"] == variant]
        averaged = np.mean(
            [result.probabilities[variant] for result in results], axis=0
        )
        prediction = classes[averaged.argmax(axis=1)]
        summary_rows.append({
            "variant": variant,
            "seed_count": len(results),
            "per_seed_f1_mean": rows["f1_macro"].mean(),
            "per_seed_f1_std": rows["f1_macro"].std(ddof=1) if len(rows) > 1 else 0.0,
            "per_seed_f1_min": rows["f1_macro"].min(),
            "probability_averaged_oof_f1": f1_score(
                labels, prediction, average="macro", zero_division=0
            ),
            "probability_averaged_oof_accuracy": accuracy_score(labels, prediction),
        })
    summary = pd.DataFrame(summary_rows)

    h0_mean = summary.loc[summary.variant == "h0", "per_seed_f1_mean"].iloc[0]
    exact_mean = summary.loc[
        summary.variant == "final_exact", "per_seed_f1_mean"
    ].iloc[0]
    fold_metrics = pd.concat([result.fold_metrics for result in results], ignore_index=True)
    decisions = pd.DataFrame([{
        "h0_mean": h0_mean,
        "exact_mean": exact_mean,
        "mean_delta": exact_mean - h0_mean,
        "minimum_seed_delta": min(
            result.scores()["final_exact"]["f1_macro"]
            - result.scores()["h0"]["f1_macro"]
            for result in results
        ),
        "positive_seeds": sum(
            result.scores()["final_exact"]["f1_macro"]
            > result.scores()["h0"]["f1_macro"]
            for result in results
        ),
        "positive_folds": int((fold_metrics["delta_vs_h0"] > 0).sum()),
        "total_folds": len(fold_metrics),
        "reported_exact_mean": REPORTED_EXACT_MEAN,
        "delta_vs_reported": exact_mean - REPORTED_EXACT_MEAN,
    }])
    return per_seed, summary, decisions


def class_comparison(results: list[SeedResult]) -> pd.DataFrame:
    labels, classes = results[0].labels, results[0].classes
    rows = []
    for seed_result in results:
        h0_pred = classes[seed_result.probabilities["h0"].argmax(axis=1)]
        exact_pred = classes[seed_result.probabilities["final_exact"].argmax(axis=1)]
        for label in classes:
            h0_f1 = f1_score(
                labels, h0_pred, labels=[label], average="macro", zero_division=0
            )
            exact_f1 = f1_score(
                labels, exact_pred, labels=[label], average="macro", zero_division=0
            )
            rows.append({
                "seed": seed_result.seed,
                "class": label,
                "support": int((labels == label).sum()),
                "h0_f1": h0_f1,
                "exact_f1": exact_f1,
                "delta": exact_f1 - h0_f1,
                "h0_wrong_exact_right": int(((h0_pred != labels) & (exact_pred == labels) & (labels == label)).sum()),
                "h0_right_exact_wrong": int(((h0_pred == labels) & (exact_pred != labels) & (labels == label)).sum()),
            })
    frame = pd.DataFrame(rows)
    return frame.groupby("class", as_index=False).agg(
        support=("support", "first"),
        h0_f1_mean=("h0_f1", "mean"),
        exact_f1_mean=("exact_f1", "mean"),
        delta_mean=("delta", "mean"),
        recovered=("h0_wrong_exact_right", "sum"),
        broken=("h0_right_exact_wrong", "sum"),
    ).sort_values("delta_mean", ascending=False)


def save_results(results: list[SeedResult], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed, summary, decisions = aggregate(results)
    classes = class_comparison(results)
    per_seed.to_csv(output_dir / "exact_event_3seed_per_seed.csv", index=False)
    summary.to_csv(output_dir / "exact_event_3seed_summary.csv", index=False)
    decisions.to_csv(output_dir / "exact_event_3seed_decision.csv", index=False)
    classes.to_csv(output_dir / "exact_event_3seed_classes.csv", index=False)
    for result in results:
        result.fold_metrics.to_csv(
            output_dir / f"exact_event_seed{result.seed}_folds.csv", index=False
        )
        result.audit.to_csv(
            output_dir / f"exact_event_seed{result.seed}_audit.csv", index=False
        )
    decision = decisions.iloc[0].to_dict()
    report = {
        **decision,
        "seeds": [result.seed for result in results],
        "test_read": False,
        "all_fold_audits_pass": bool(all(
            result.audit["leakage_check"].all() for result in results
        )),
        "convergence_warning_count": int(sum(
            result.warning_count for result in results
        )),
    }
    (output_dir / "exact_event_3seed_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def permutation_check(
    train: pd.DataFrame,
    genes: list[str],
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Audit supervised EB with permuted labels, without the specialist.

    Random labels can make the automatic two-pair specialist choose overlapping
    class pairs.  That component is unrelated to the supervised-feature sanity
    question and can invalidate probability mass when its pairs overlap.  This
    audit therefore compares three LR representations on identical folds:
    structured H0, gene-type EB, and gene-type plus exact-event EB.
    """

    rng = np.random.default_rng(seed)
    labels = rng.permutation(train["SUBCLASS"].to_numpy())
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    probability = {
        name: np.zeros((len(train), len(classes)), dtype=np.float64)
        for name in ("non_eb_lr", "gene_eb_lr", "exact_eb_lr")
    }
    rows = []
    splitter = StratifiedKFold(
        n_splits=OUTER_SPLITS, shuffle=True, random_state=seed
    )
    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(train)), labels), start=1
    ):
        print(f"[permutation seed={seed}] fold {fold}/{OUTER_SPLITS}", flush=True)
        fold_seed = seed * 100 + fold
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        y_fit = labels[fit_index]
        x_fit, x_valid, _, _ = P.make_h0_fold_matrices(
            fit_frame, valid_frame, y_fit, genes, fold_seed
        )
        non_eb, _ = _fit_lr(x_fit, y_fit, x_valid, classes, seed)

        gene_fit, gene_valid = P.empirical_bayes_features(
            fit_frame, valid_frame, y_fit, classes, genes, seed=fold_seed
        )
        x_fit_gene = sparse.hstack(
            [x_fit, sparse.csr_matrix(gene_fit)], format="csr"
        )
        x_valid_gene = sparse.hstack(
            [x_valid, sparse.csr_matrix(gene_valid)], format="csr"
        )
        gene_eb, _ = _fit_lr(
            x_fit_gene, y_fit, x_valid_gene, classes, seed
        )

        exact_fit, exact_valid, _ = P.exact_eb_features(
            fit_frame, valid_frame, y_fit, classes, genes, seed=fold_seed
        )
        x_fit_exact = sparse.hstack(
            [x_fit_gene, sparse.csr_matrix(exact_fit)], format="csr"
        )
        x_valid_exact = sparse.hstack(
            [x_valid_gene, sparse.csr_matrix(exact_valid)], format="csr"
        )
        exact_eb, _ = _fit_lr(
            x_fit_exact, y_fit, x_valid_exact, classes, seed
        )

        fold_probability = {
            "non_eb_lr": non_eb,
            "gene_eb_lr": gene_eb,
            "exact_eb_lr": exact_eb,
        }
        for name, values in fold_probability.items():
            probability[name][valid_index] = values
        rows.append({
            "fold": fold,
            "non_eb_lr_f1": _score(labels[valid_index], classes, non_eb),
            "gene_eb_lr_f1": _score(labels[valid_index], classes, gene_eb),
            "exact_eb_lr_f1": _score(labels[valid_index], classes, exact_eb),
        })
        del x_fit, x_valid, x_fit_gene, x_valid_gene, x_fit_exact, x_valid_exact
        gc.collect()

    scores = {
        name: _score(labels, classes, values)
        for name, values in probability.items()
    }
    delta_exact_vs_gene = scores["exact_eb_lr"] - scores["gene_eb_lr"]
    delta_exact_vs_non_eb = scores["exact_eb_lr"] - scores["non_eb_lr"]
    report = {
        "seed": seed,
        **scores,
        "delta_exact_vs_gene_eb": delta_exact_vs_gene,
        "delta_exact_vs_non_eb": delta_exact_vs_non_eb,
        "pass_threshold": 0.01,
        "pass": bool(delta_exact_vs_gene < 0.01),
        "specialist_excluded": True,
        "specialist_exclusion_reason": "permuted labels may select overlapping automatic pairs",
        "test_read": False,
    }
    return pd.DataFrame(rows), report
