"""Train-only automatic confusion-group mixture-of-experts utilities."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from h0_faithful_pipeline import build_design_matrices


@dataclass
class H0FoldOutput:
    probability: np.ndarray
    x_fit: sparse.csr_matrix
    x_apply: sparse.csr_matrix
    y_fit: np.ndarray
    names: list[str]
    audit: dict
    specialist_pairs: tuple[tuple[str, str], ...]
    convergence_warnings: int


def _align(model, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(model.classes_)}
    return probability[:, [lookup[label] for label in classes]]


def discover_similarity_pairs(
    x_fit: sparse.csr_matrix, names: list[str], y_fit: np.ndarray, classes: np.ndarray
) -> tuple[tuple[str, str], ...]:
    gene_columns = np.asarray([name.startswith("G__") for name in names])
    matrix = x_fit[:, gene_columns]
    centroids = []
    for label in classes:
        centroid = np.asarray(matrix[y_fit == label].mean(axis=0)).ravel()
        norm = np.linalg.norm(centroid)
        centroids.append(centroid / norm if norm else centroid)
    profile = np.vstack(centroids)
    candidates = [
        (-float(profile[left] @ profile[right]), str(classes[left]), str(classes[right]))
        for left in range(len(classes))
        for right in range(left + 1, len(classes))
    ]
    return tuple((left, right) for _, left, right in sorted(candidates)[:2])


def _h0_specialist_probability(
    main_probability: np.ndarray,
    x_fit: sparse.csr_matrix,
    y_fit: np.ndarray,
    x_apply: sparse.csr_matrix,
    classes: np.ndarray,
    pairs: tuple[tuple[str, str], ...],
    seed: int,
) -> np.ndarray:
    output = main_probability.copy()
    index = {label: position for position, label in enumerate(classes)}
    original_prediction = classes[main_probability.argmax(axis=1)]
    for pair in pairs:
        mask = np.isin(y_fit, pair)
        model = LGBMClassifier(
            objective="binary", boosting_type="gbdt", n_estimators=100,
            learning_rate=.02, num_leaves=20, min_child_samples=10,
            reg_alpha=0.0, reg_lambda=0.0, importance_type="gain",
            class_weight="balanced", random_state=seed, n_jobs=-1,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )
        model.fit(x_fit[mask], y_fit[mask])
        route = np.isin(original_prediction, pair)
        if not route.any():
            continue
        raw = model.predict_proba(x_apply[route])
        local = {label: position for position, label in enumerate(model.classes_)}
        ratio = raw[:, local[pair[0]]]
        columns = [index[pair[0]], index[pair[1]]]
        mass = main_probability[route][:, columns].sum(axis=1)
        output[np.ix_(np.flatnonzero(route), columns)] = np.column_stack(
            (mass * ratio, mass * (1.0 - ratio))
        )
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)
    return output


def fit_h0_fold(
    fit_frame, apply_frame, y_fit: np.ndarray, genes: list[str], classes: np.ndarray, *, seed: int
) -> H0FoldOutput:
    """Fit train-only H0 components once and retain matrices for group specialists."""
    x_fit, x_apply, names, audit = build_design_matrices(
        fit_frame, apply_frame, y_fit, genes, seed=seed
    )
    lr = LogisticRegression(
        solver="lbfgs", C=.07, max_iter=2000, class_weight="balanced", random_state=42
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr.fit(x_fit, y_fit)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    lr_probability = _align(lr, lr.predict_proba(x_apply), classes)
    main = LGBMClassifier(
        objective="multiclass", boosting_type="gbdt", num_class=len(classes),
        n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10,
        min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0,
        class_weight="balanced", random_state=42, n_jobs=-1,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    main.fit(x_fit, y_fit)
    pairs = discover_similarity_pairs(x_fit, names, y_fit, classes)
    specialist_probability = _h0_specialist_probability(
        _align(main, main.predict_proba(x_apply), classes),
        x_fit, y_fit, x_apply, classes, pairs, seed=42,
    )
    probability = .80 * lr_probability + .20 * specialist_probability
    del lr, main
    gc.collect()
    return H0FoldOutput(
        probability=probability, x_fit=x_fit, x_apply=x_apply, y_fit=np.asarray(y_fit),
        names=names, audit=audit, specialist_pairs=pairs,
        convergence_warnings=warning_count,
    )


def discover_confusion_groups(
    truth: np.ndarray, prediction: np.ndarray, classes: np.ndarray, *, n_groups: int = 6
) -> tuple[tuple[str, ...], ...]:
    """Merge labels with highest inner-OOF bidirectional confusion until n_groups."""
    groups = [tuple([str(label)]) for label in classes]
    lookup = {str(label): index for index, label in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        confusion[lookup[str(actual)], lookup[str(predicted)]] += 1
    while len(groups) > min(n_groups, len(classes)):
        candidates = []
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                score = sum(
                    confusion[lookup[a], lookup[b]] + confusion[lookup[b], lookup[a]]
                    for a in groups[left] for b in groups[right]
                )
                candidates.append((-int(score), groups[left], groups[right], left, right))
        _, first, second, left, right = sorted(candidates)[0]
        merged = tuple(sorted((*first, *second)))
        groups = [group for index, group in enumerate(groups) if index not in (left, right)]
        groups.append(merged)
        groups.sort()
    return tuple(groups)


def redistribute_group_probability_mass(
    base_probability: np.ndarray,
    specialist_probability: np.ndarray,
    classes: np.ndarray,
    group: tuple[str, ...],
) -> np.ndarray:
    output = base_probability.copy()
    index = {str(label): position for position, label in enumerate(classes)}
    columns = [index[label] for label in group]
    mass = base_probability[:, columns].sum(axis=1, keepdims=True)
    output[:, columns] = mass * specialist_probability
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(output[:, columns].sum(axis=1), mass.ravel(), atol=1e-6)
    return output


def apply_group_experts(
    base_probability: np.ndarray,
    x_fit: sparse.csr_matrix,
    y_fit: np.ndarray,
    x_apply: sparse.csr_matrix,
    classes: np.ndarray,
    groups: tuple[tuple[str, ...], ...],
    *, seed: int,
) -> tuple[np.ndarray, list[dict]]:
    output = base_probability.copy()
    rows: list[dict] = []
    for group_index, group in enumerate(groups):
        if len(group) < 2:
            continue
        mask = np.isin(y_fit, group)
        model = LGBMClassifier(
            objective="multiclass", boosting_type="gbdt", num_class=len(group),
            n_estimators=100, learning_rate=.02, num_leaves=20, min_child_samples=10,
            reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced",
            random_state=seed + group_index, n_jobs=-1, deterministic=True,
            force_col_wise=True, verbosity=-1,
        )
        model.fit(x_fit[mask], y_fit[mask])
        local = {str(label): position for position, label in enumerate(model.classes_)}
        specialist = model.predict_proba(x_apply)[:, [local[label] for label in group]]
        output = redistribute_group_probability_mass(output, specialist, classes, group)
        rows.append({"group_id": group_index, "classes": "|".join(group), "size": len(group), "fit_support": int(mask.sum())})
        del model
        gc.collect()
    return output, rows


def inner_oof_h0_probability(
    frame, labels: np.ndarray, genes: list[str], classes: np.ndarray, *, seed: int
) -> tuple[np.ndarray, list[dict]]:
    probability = np.zeros((len(frame), len(classes)), dtype=np.float64)
    audits: list[dict] = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (fit_index, holdout_index) in enumerate(splitter.split(np.zeros(len(frame)), labels), 1):
        output = fit_h0_fold(
            frame.iloc[fit_index][genes].reset_index(drop=True),
            frame.iloc[holdout_index][genes].reset_index(drop=True), labels[fit_index], genes, classes,
            seed=seed * 100 + fold,
        )
        probability[holdout_index] = output.probability
        audits.append({
            "inner_fold": fold, "fit_rows": int(len(fit_index)), "holdout_rows": int(len(holdout_index)),
            "outer_validation_used_for_fit": False, "leakage_check": output.audit["raw_train_test_concat"] is False,
            "nan_as_mutation_count": 0,
        })
        del output
        gc.collect()
    return probability, audits
