"""Self-contained regulation-safe H0 for the H2-S screen.

No fixed cancer, gene, or exact mutation names are present in this module.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from h2_evidence_shape_core import EventMatrices, EventState, fit_event_state, parse_frame, transform_event_state


@dataclass
class H0Prediction:
    probability: np.ndarray
    state: EventState
    output_matrices: EventMatrices
    feature_count: int
    convergence_warnings: int
    specialist_pairs: tuple[tuple[str, str], ...]


def _balanced_weight(labels: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    count = np.asarray([(labels == label).sum() for label in classes], dtype=np.float64)
    return {str(label): float(len(labels) / max(len(classes) * value, 1.0)) for label, value in zip(classes, count)}


def _align_probability(model, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((len(probability), len(classes)), dtype=np.float64)
    index = {label: pos for pos, label in enumerate(classes)}
    for pos, label in enumerate(model.classes_):
        out[:, index[label]] = probability[:, pos]
    return out


def _nonconstant(x: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(x.getnnz(axis=0)).ravel().astype(bool) & (np.asarray(x.getnnz(axis=0)).ravel() < x.shape[0])


def _apply_enrichment(matrix: sparse.csr_matrix, weights: np.ndarray) -> np.ndarray:
    scores = np.asarray(matrix @ weights.T, dtype=np.float32)
    denominator = np.sqrt(np.maximum(np.asarray(matrix.getnnz(axis=1)).ravel(), 1)).astype(np.float32)
    return scores / denominator[:, None]


def _cross_fitted_enrichment(frame: pd.DataFrame, genes: list[str], labels: np.ndarray, classes: np.ndarray, seed: int) -> np.ndarray:
    """Generate training-only EB scores for every fit row via inner OOF."""
    scores = np.zeros((len(frame), len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for inner_fit, inner_holdout in splitter.split(np.zeros(len(frame)), labels):
        local_fit = parse_frame(frame.iloc[inner_fit][genes], genes)
        local_state = fit_event_state(local_fit, labels[inner_fit], min_support=1, shrinkage=10.0)
        if tuple(classes) != local_state.classes:
            raise AssertionError("inner fold class order mismatch")
        local_holdout = transform_event_state(parse_frame(frame.iloc[inner_holdout][genes], genes), local_state)
        scores[inner_holdout] = _apply_enrichment(local_holdout.gene_type, local_state.eb_weights)
    return scores


def _design(fit: EventMatrices, out: EventMatrices, state: EventState, enrich_fit: np.ndarray, enrich_out: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    active = _nonconstant(fit.mutation)
    truncating = _nonconstant(fit.truncation)
    recurrent = np.flatnonzero(np.asarray(fit.exact.getnnz(axis=0)).ravel() >= 5)
    dense_fit = np.hstack((np.log1p(fit.burden), np.log1p(fit.type_counts), np.log1p(fit.amino_pair), fit.topology, enrich_fit))
    dense_out = np.hstack((np.log1p(out.burden), np.log1p(out.type_counts), np.log1p(out.amino_pair), out.topology, enrich_out))
    raw_fit = sparse.hstack((fit.mutation[:, active], sparse.csr_matrix(dense_fit[:, :10]), fit.truncation[:, truncating], sparse.csr_matrix(fit.truncation.sum(axis=1)), fit.exact[:, recurrent], sparse.csr_matrix(fit.exact[:, recurrent].sum(axis=1)), sparse.csr_matrix(dense_fit[:, 10:])), format="csr")
    raw_out = sparse.hstack((out.mutation[:, active], sparse.csr_matrix(dense_out[:, :10]), out.truncation[:, truncating], sparse.csr_matrix(out.truncation.sum(axis=1)), out.exact[:, recurrent], sparse.csr_matrix(out.exact[:, recurrent].sum(axis=1)), sparse.csr_matrix(dense_out[:, 10:])), format="csr")
    keep = _nonconstant(raw_fit)
    return raw_fit[:, keep], raw_out[:, keep]


def discover_pairs(mutation: np.ndarray | sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray, *, top_k: int = 2) -> tuple[tuple[str, str], ...]:
    rows = []
    for label in classes:
        mask = labels == label
        mean = np.asarray(mutation[mask].mean(axis=0)).ravel().astype(np.float64)
        norm = np.linalg.norm(mean)
        rows.append(mean / norm if norm else mean)
    profile = np.vstack(rows)
    similarity = profile @ profile.T
    pairs = []
    for left in range(len(classes)):
        for right in range(left + 1, len(classes)):
            pairs.append((float(similarity[left, right]), str(classes[left]), str(classes[right])))
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    return tuple((left, right) for _, left, right in pairs[:top_k])


def _apply_specialists(main_probability: np.ndarray, x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_out: sparse.csr_matrix, classes: np.ndarray, pairs: tuple[tuple[str, str], ...], seed: int) -> np.ndarray:
    out = main_probability.copy()
    index = {label: pos for pos, label in enumerate(classes)}
    for pair_no, pair in enumerate(pairs):
        mask = np.isin(y_fit, pair)
        if mask.sum() < 4 or len(np.unique(y_fit[mask])) < 2:
            continue
        model = LGBMClassifier(objective="binary", boosting_type="gbdt", n_estimators=100, learning_rate=.02, num_leaves=20, min_child_samples=10, min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced", random_state=seed * 10 + pair_no, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1)
        model.fit(x_fit[mask], y_fit[mask])
        predicted = classes[main_probability.argmax(axis=1)]
        route = np.isin(predicted, pair)
        if not route.any():
            continue
        probability = model.predict_proba(x_out[route])
        pair_pos = [index[label] for label in model.classes_]
        mass = main_probability[route][:, [index[pair[0]], index[pair[1]]]].sum(axis=1, keepdims=True)
        out[np.ix_(np.flatnonzero(route), pair_pos)] = mass * probability
    return out


def fit_predict_h0(fit_frame: pd.DataFrame, out_frame: pd.DataFrame, genes: list[str], labels: np.ndarray, *, seed: int) -> H0Prediction:
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    fit_parsed, out_parsed = parse_frame(fit_frame[genes], genes), parse_frame(out_frame[genes], genes)
    state = fit_event_state(fit_parsed, labels, min_support=1, shrinkage=10.0)
    fit_matrix, out_matrix = transform_event_state(fit_parsed, state), transform_event_state(out_parsed, state)
    enrich_fit = _cross_fitted_enrichment(fit_frame.reset_index(drop=True), genes, labels, classes, seed)
    enrich_out = _apply_enrichment(out_matrix.gene_type, state.eb_weights)
    x_fit, x_out = _design(fit_matrix, out_matrix, state, enrich_fit, enrich_out)
    warning_count = 0
    lr = LogisticRegression(solver="lbfgs", C=.07, max_iter=2000, class_weight="balanced", random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr.fit(x_fit, labels)
    warning_count += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    lr_probability = _align_probability(lr, lr.predict_proba(x_out), classes)
    lgbm = LGBMClassifier(objective="multiclass", boosting_type="gbdt", num_class=len(classes), n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10, min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight=_balanced_weight(labels, classes), random_state=seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1)
    lgbm.fit(x_fit, labels)
    lgbm_probability = _align_probability(lgbm, lgbm.predict_proba(x_out), classes)
    pairs = discover_pairs(fit_matrix.mutation, labels, classes, top_k=2)
    specialist_probability = _apply_specialists(lgbm_probability, x_fit, labels, x_out, classes, pairs, seed)
    probability = .80 * lr_probability + .20 * specialist_probability
    probability /= probability.sum(axis=1, keepdims=True)
    return H0Prediction(probability.astype(np.float32), state, out_matrix, int(x_fit.shape[1]), warning_count, pairs)


def build_evidence_shape(gene_type: np.ndarray | sparse.csr_matrix, weights: np.ndarray, priors: np.ndarray, probability: np.ndarray) -> np.ndarray:
    is_sparse = sparse.issparse(gene_type)
    x = gene_type.tocsr() if is_sparse else np.asarray(gene_type)
    n_rows, n_classes = x.shape[0], probability.shape[1]
    out = np.zeros((n_rows, n_classes, 19), dtype=np.float32)
    rank = (-probability).argsort(axis=1).argsort(axis=1) + 1
    top = probability.max(axis=1, keepdims=True)
    sorted_prob = np.sort(probability, axis=1)
    margin = (sorted_prob[:, -1] - sorted_prob[:, -2]).reshape(-1, 1)
    for row in range(n_rows):
        active = x.indices[x.indptr[row]:x.indptr[row + 1]] if is_sparse else np.flatnonzero(x[row])
        for cls in range(n_classes):
            evidence = weights[cls, active] if len(active) else np.empty(0, dtype=np.float32)
            positive, negative = evidence[evidence > 0], evidence[evidence < 0]
            absolute = np.abs(evidence)
            total_abs = float(absolute.sum())
            ordered = np.sort(absolute)[::-1]
            entropy = 0.0 if not total_abs else float(-(absolute / total_abs * np.log(np.maximum(absolute / total_abs, 1e-12))).sum())
            values = (float(evidence.sum()), float(positive.sum()), float(negative.sum()), float(positive.max()) if len(positive) else 0., float(negative.min()) if len(negative) else 0., len(positive), len(negative), total_abs, float(ordered[:1].sum() / total_abs) if total_abs else 0., float(ordered[:3].sum() / total_abs) if total_abs else 0., entropy, float(evidence.sum() - (evidence[np.argmax(absolute)] if len(evidence) else 0.)), len(active), probability[row, cls], np.log(max(probability[row, cls], 1e-12)), rank[row, cls], top[row, 0] - probability[row, cls], margin[row, 0], priors[cls])
            out[row, cls] = values
    return out
