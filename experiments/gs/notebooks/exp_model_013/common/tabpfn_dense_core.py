"""Fold-safe dense H0 evidence features for a fixed TabPFN screen.

This module never reads test data.  It only converts matrices already built
from an outer-fold training frame and its validation frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from scipy import sparse


WT = "WT"


def normalise_event_cell(value: object) -> tuple[str, ...]:
    """Return unique events; NaN, blank and WT always produce no event."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    text = str(value).strip().upper()
    if not text or text == WT:
        return ()
    return tuple(dict.fromkeys(token.removeprefix("P.") for token in re.sub(r"[;,|]+", " ", text).split() if token))


@dataclass
class FitOnlyStandardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, fit_matrix: np.ndarray) -> "FitOnlyStandardizer":
        values = np.asarray(fit_matrix, dtype=np.float32)
        if values.ndim != 2 or not len(values):
            raise ValueError("fit_matrix must be a non-empty 2D array")
        self.mean_ = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale_ = values.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")
        values = np.asarray(matrix, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.mean_):
            raise ValueError("matrix does not match fitted feature count")
        output = (values - self.mean_) / self.scale_
        if not np.isfinite(output).all():
            raise ValueError("non-finite dense feature encountered")
        return output.astype(np.float32, copy=False)


def _to_dense(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    return matrix.toarray().astype(np.float32, copy=False) if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float32)


def build_dense_h0_view(
    x_fit: sparse.csr_matrix,
    x_valid: sparse.csr_matrix,
    feature_names: list[str],
    eb_fit: np.ndarray,
    eb_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Keep compact dense H0 blocks plus outer-train-only EB scores.

    Raw gene and exact-event one-hot blocks are intentionally excluded.
    """
    prefixes = ("B__", "V__", "T__", "A_pair__", "S__", "E__")
    keep = np.asarray([name.startswith(prefixes) for name in feature_names], dtype=bool)
    if not keep.any():
        raise ValueError("no dense H0 blocks found")
    selected_names = [name for name, include in zip(feature_names, keep) if include]
    fit_base, valid_base = _to_dense(x_fit[:, keep]), _to_dense(x_valid[:, keep])
    eb_fit, eb_valid = np.asarray(eb_fit, dtype=np.float32), np.asarray(eb_valid, dtype=np.float32)
    if eb_fit.ndim != 2 or eb_fit.shape[1] == 0 or eb_valid.shape[1] != eb_fit.shape[1]:
        raise ValueError("EB score matrices must be aligned non-empty 2D arrays")

    def evidence_shape(values: np.ndarray) -> np.ndarray:
        absolute = np.abs(values)
        positive = np.maximum(values, 0.0)
        negative = np.minimum(values, 0.0)
        return np.column_stack([
            positive.sum(axis=1), negative.sum(axis=1), absolute.sum(axis=1),
            positive.max(axis=1), negative.min(axis=1), absolute.max(axis=1),
        ]).astype(np.float32)

    fit_shape, valid_shape = evidence_shape(eb_fit), evidence_shape(eb_valid)
    output_fit = np.hstack([fit_base, eb_fit, fit_shape]).astype(np.float32, copy=False)
    output_valid = np.hstack([valid_base, eb_valid, valid_shape]).astype(np.float32, copy=False)
    names = selected_names + [f"EB__class_score_{index}" for index in range(eb_fit.shape[1])] + [
        "EB_shape__positive_sum", "EB_shape__negative_sum", "EB_shape__absolute_sum",
        "EB_shape__positive_max", "EB_shape__negative_min", "EB_shape__absolute_max",
    ]
    if not np.isfinite(output_fit).all() or not np.isfinite(output_valid).all():
        raise ValueError("dense H0 view contains non-finite values")
    return output_fit, output_valid, names


def package_contract() -> dict[str, object]:
    """Non-fitting dependency metadata used by the notebook and audit."""
    try:
        import tabpfn  # type: ignore
    except ModuleNotFoundError:
        return {"tabpfn_installed": False, "tabpfn_version": None}
    return {"tabpfn_installed": True, "tabpfn_version": getattr(tabpfn, "__version__", "unknown")}
