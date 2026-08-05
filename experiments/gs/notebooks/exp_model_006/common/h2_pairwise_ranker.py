"""Memory-bounded shared pairwise ranker for H2-S."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def make_symmetric_pairs(features: np.ndarray, labels: np.ndarray, classes: np.ndarray, *, max_negatives: int = 25) -> tuple[np.ndarray, np.ndarray]:
    rows, targets = [], []
    index = {label: pos for pos, label in enumerate(classes)}
    for row, label in enumerate(labels):
        truth = index[label]
        negatives = [candidate for candidate in range(len(classes)) if candidate != truth][:max_negatives]
        for candidate in negatives:
            delta = features[row, truth] - features[row, candidate]
            rows.extend((delta, -delta))
            targets.extend((1, 0))
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.int8)


def candidate_residuals(features: np.ndarray, coefficient: np.ndarray, intercept: float = 0.0) -> np.ndarray:
    return np.einsum("rcf,f->rc", features, coefficient, optimize=True) + float(intercept)


def apply_residual_correction(probability: np.ndarray, residual: np.ndarray, strength: float) -> np.ndarray:
    log_probability = np.log(np.clip(probability, 1e-12, 1.0)) + float(strength) * residual
    return np.exp(log_probability - logsumexp(log_probability, axis=1, keepdims=True))
