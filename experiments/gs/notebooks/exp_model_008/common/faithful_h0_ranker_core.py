"""Candidate-wise evidence summaries and shared pairwise ranking primitives."""
from __future__ import annotations

import numpy as np
from scipy import sparse


FEATURE_COUNT = 19


def build_evidence_shape(
    gene_type: sparse.csr_matrix,
    weights: np.ndarray,
    priors: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    """Return train-fitted candidate evidence shape `[row, class, 19]`."""
    matrix = gene_type.tocsr()
    probability = np.asarray(probability, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    rows, classes = probability.shape
    if matrix.shape[0] != rows or weights.shape != (classes, matrix.shape[1]) or priors.shape != (classes,):
        raise ValueError("evidence shape inputs do not align")
    result = np.zeros((rows, classes, FEATURE_COUNT), dtype=np.float32)
    rank = (-probability).argsort(axis=1).argsort(axis=1) + 1
    sorted_probability = np.sort(probability, axis=1)
    top_probability = sorted_probability[:, -1]
    margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    for row in range(rows):
        active = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
        for class_index in range(classes):
            evidence = weights[class_index, active] if active.size else np.empty(0, dtype=np.float64)
            positive = evidence[evidence > 0]
            negative = evidence[evidence < 0]
            absolute = np.abs(evidence)
            absolute_sum = float(absolute.sum())
            ordered = np.sort(absolute)[::-1]
            normalized = absolute / absolute_sum if absolute_sum else absolute
            entropy = float(-(normalized * np.log(np.maximum(normalized, 1e-12))).sum()) if absolute_sum else 0.0
            dominant = float(evidence[np.argmax(absolute)]) if absolute.size else 0.0
            result[row, class_index] = (
                float(evidence.sum()),
                float(positive.sum()),
                float(negative.sum()),
                float(positive.max()) if positive.size else 0.0,
                float(negative.min()) if negative.size else 0.0,
                float(positive.size),
                float(negative.size),
                absolute_sum,
                float(ordered[:1].sum() / absolute_sum) if absolute_sum else 0.0,
                float(ordered[:3].sum() / absolute_sum) if absolute_sum else 0.0,
                entropy,
                float(evidence.sum() - dominant),
                float(active.size),
                float(probability[row, class_index]),
                float(np.log(max(probability[row, class_index], 1e-12))),
                float(rank[row, class_index]),
                float(top_probability[row] - probability[row, class_index]),
                float(margin[row]),
                float(priors[class_index]),
            )
    return result


def make_symmetric_pairs(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """True-vs-other candidate differences, plus their exact reverse direction."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=object)
    classes = np.asarray(classes, dtype=object)
    if features.shape[:2] != (len(labels), len(classes)):
        raise ValueError("candidate feature rows must align with labels/classes")
    class_to_index = {label: index for index, label in enumerate(classes)}
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for row, label in enumerate(labels):
        true_index = class_to_index[label]
        for candidate_index in range(len(classes)):
            if candidate_index == true_index:
                continue
            difference = features[row, true_index] - features[row, candidate_index]
            rows.extend((difference, -difference))
            targets.extend((1, 0))
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.int8)


def candidate_residual(shape: np.ndarray, coefficient: np.ndarray, intercept: float) -> np.ndarray:
    """Score every class candidate with one shared linear ranker."""
    shape = np.asarray(shape, dtype=np.float32)
    coefficient = np.asarray(coefficient, dtype=np.float32)
    if shape.shape[2] != coefficient.size:
        raise ValueError("coefficient length does not match evidence features")
    return np.tensordot(shape, coefficient, axes=([2], [0])) + float(intercept)


def apply_residual(probability: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    """Correct base log-probability with a shared candidate residual."""
    if alpha not in (0.10, 0.20):
        raise ValueError("alpha must be one of the fixed inner-OOF candidates")
    probability = np.asarray(probability, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if probability.shape != residual.shape:
        raise ValueError("probability and residual must share shape")
    logits = np.log(np.maximum(probability, 1e-12)) + alpha * residual
    logits -= logits.max(axis=1, keepdims=True)
    output = np.exp(logits)
    output /= output.sum(axis=1, keepdims=True)
    return output.astype(np.float32)
