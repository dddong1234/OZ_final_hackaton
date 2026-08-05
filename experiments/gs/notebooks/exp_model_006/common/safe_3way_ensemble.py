"""Probability helpers for the fixed train-only 3-way screen."""
from __future__ import annotations

import numpy as np


WEIGHTS = {"multinomial": .55, "ovr": .30, "lightgbm": .15}


def align_probability(model, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Align a classifier's probability columns to the shared class order."""
    probability = np.asarray(probability, dtype=np.float64)
    lookup = {str(label): index for index, label in enumerate(model.classes_)}
    output = probability[:, [lookup[str(label)] for label in classes]]
    output /= output.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)
    return output


def fixed_three_way_probability(
    multinomial: np.ndarray, ovr: np.ndarray, lightgbm: np.ndarray
) -> np.ndarray:
    """Apply the pre-declared 0.55/0.30/0.15 blend without tuning."""
    multi = np.asarray(multinomial, dtype=np.float64)
    ovr_probability = np.asarray(ovr, dtype=np.float64)
    lgbm = np.asarray(lightgbm, dtype=np.float64)
    if multi.shape != ovr_probability.shape or multi.shape != lgbm.shape:
        raise ValueError("three probability matrices must have equal shape")
    output = WEIGHTS["multinomial"] * multi + WEIGHTS["ovr"] * ovr_probability + WEIGHTS["lightgbm"] * lgbm
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)
    return output
