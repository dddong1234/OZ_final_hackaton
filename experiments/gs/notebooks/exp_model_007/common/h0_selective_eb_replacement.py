"""Fold-safe Empirical-Bayes LR branch for the faithful H0 ensemble.

The module deliberately contains no class, gene, or mutation identifiers.
All event vocabulary and class-conditional weights are learned from the
current outer-fold training data.  It does not read test data.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold


SELECTIVE_LR_WEIGHT = 0.80
H0_SPECIALIST_WEIGHT = 0.20
SELECTIVE_MARGIN = 0.05
EB_ALPHA = 1.0
EB_SHRINKAGE = 20.0
EB_CLIP = 4.0


def _h0_common() -> Path:
    here = Path(__file__).resolve()
    path = here.parents[2] / "exp_model_006" / "common"
    if not path.exists():
        raise FileNotFoundError("GS faithful H0 source was not found")
    return path


if str(_h0_common()) not in sys.path:
    sys.path.insert(0, str(_h0_common()))

from h0_faithful_pipeline import fit_vocabulary, transform_rows  # noqa: E402


@dataclass(frozen=True)
class EBState:
    selected: np.ndarray
    weights: np.ndarray


def fixed_branch_replacement(selective_lr_probability: np.ndarray, specialist_probability: np.ndarray) -> np.ndarray:
    """Preserve the H0 80/20 contract while replacing only its LR branch."""
    selective_lr_probability = np.asarray(selective_lr_probability, dtype=np.float64)
    specialist_probability = np.asarray(specialist_probability, dtype=np.float64)
    if selective_lr_probability.ndim != 2 or selective_lr_probability.shape != specialist_probability.shape:
        raise ValueError("LR and specialist probability matrices must share shape")
    output = SELECTIVE_LR_WEIGHT * selective_lr_probability + H0_SPECIALIST_WEIGHT * specialist_probability
    if not np.allclose(output.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Input probability rows must be normalized")
    return output.astype(np.float32)


def selective_probability(non_eb_probability: np.ndarray, eb_probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the previously fixed margin rule without re-tuning it."""
    non_eb_probability = np.asarray(non_eb_probability, dtype=np.float64)
    eb_probability = np.asarray(eb_probability, dtype=np.float64)
    if non_eb_probability.ndim != 2 or non_eb_probability.shape != eb_probability.shape or eb_probability.shape[1] < 2:
        raise ValueError("Expected equal two-dimensional probability matrices with at least two classes")
    top_two = np.partition(eb_probability, kth=-2, axis=1)[:, -2:]
    use_non_eb = (top_two[:, 1] - top_two[:, 0]) < SELECTIVE_MARGIN
    probability = np.where(use_non_eb[:, None], non_eb_probability, eb_probability)
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Input probability rows must be normalized")
    return probability.astype(np.float32), use_non_eb


def fit_empirical_bayes(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> EBState:
    """Fit only outer/inner-train gene×event-type class evidence weights."""
    matrix = matrix.tocsr()
    support = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    selected = np.flatnonzero((support > 0) & (support < matrix.shape[0]))
    if not len(selected):
        return EBState(selected=selected, weights=np.zeros((len(classes), 0), dtype=np.float32))
    x = matrix[:, selected]
    selected_support = support[selected]
    class_size = np.asarray([(labels == label).sum() for label in classes], dtype=np.float64)
    p0 = (selected_support + EB_ALPHA) / (len(labels) + 2.0 * EB_ALPHA)
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, label in enumerate(classes):
        positive = np.asarray(x[labels == label].getnnz(axis=0)).ravel().astype(np.float64)
        negative = selected_support - positive
        positive_rate = (positive + EB_SHRINKAGE * p0) / (class_size[class_index] + EB_SHRINKAGE)
        negative_rate = (negative + EB_SHRINKAGE * p0) / (len(labels) - class_size[class_index] + EB_SHRINKAGE)
        positive_rate = np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
        negative_rate = np.clip(negative_rate, 1e-6, 1.0 - 1e-6)
        weights[class_index] = np.log(positive_rate / (1.0 - positive_rate)) - np.log(negative_rate / (1.0 - negative_rate))
    return EBState(selected=selected, weights=np.clip(weights, -EB_CLIP, EB_CLIP).astype(np.float32))


def apply_empirical_bayes(matrix: sparse.csr_matrix, state: EBState, class_count: int) -> np.ndarray:
    if not len(state.selected):
        return np.zeros((matrix.shape[0], class_count), dtype=np.float32)
    x = matrix[:, state.selected]
    score = np.asarray(x @ state.weights.T, dtype=np.float32)
    denominator = np.sqrt(np.maximum(np.asarray(x.getnnz(axis=1)).ravel(), 1.0))
    return score / denominator[:, None]


def cross_fitted_eb_scores(
    fit_gene_type: sparse.csr_matrix,
    apply_gene_type: sparse.csr_matrix,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create train OOF scores, then fit outer-train weights for validation."""
    train_score = np.zeros((fit_gene_type.shape[0], len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for inner_fit, inner_valid in splitter.split(np.zeros(len(labels)), labels):
        state = fit_empirical_bayes(fit_gene_type[inner_fit], labels[inner_fit], classes)
        train_score[inner_valid] = apply_empirical_bayes(fit_gene_type[inner_valid], state, len(classes))
    final_state = fit_empirical_bayes(fit_gene_type, labels, classes)
    apply_score = apply_empirical_bayes(apply_gene_type, final_state, len(classes))
    mean = train_score.mean(axis=0, keepdims=True)
    std = np.maximum(train_score.std(axis=0, keepdims=True), 1e-6)
    return ((train_score - mean) / std).astype(np.float32), ((apply_score - mean) / std).astype(np.float32)


def empirical_bayes_features(
    fit_frame: pd.DataFrame,
    apply_frame: pd.DataFrame,
    labels: np.ndarray,
    classes: np.ndarray,
    genes: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fold-train vocabulary then standardized EB scores for one outer fold."""
    vocabulary = fit_vocabulary(fit_frame, genes)
    fit = transform_rows(fit_frame, genes, vocabulary)
    apply = transform_rows(apply_frame, genes, vocabulary)
    return cross_fitted_eb_scores(fit.gene_type, apply.gene_type, labels, classes, seed=seed)
