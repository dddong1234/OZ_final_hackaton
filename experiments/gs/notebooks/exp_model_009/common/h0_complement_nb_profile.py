"""Fixed, train-only Complement NB profile branch for the faithful H0 ensemble."""
from __future__ import annotations

import numpy as np

H0_WEIGHT = 0.80
NB_WEIGHT = 0.20
NB_ALPHA = 1.0


def profile_blend(h0_probability: np.ndarray, nb_probability: np.ndarray) -> np.ndarray:
    """Return the predeclared H0/NB probability blend without tuning a weight."""
    h0_probability = np.asarray(h0_probability, dtype=np.float64)
    nb_probability = np.asarray(nb_probability, dtype=np.float64)
    if h0_probability.ndim != 2 or h0_probability.shape != nb_probability.shape:
        raise ValueError("H0 and Complement NB probability matrices must share shape")
    if not np.allclose(h0_probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("H0 probability rows must sum to one")
    if not np.allclose(nb_probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Complement NB probability rows must sum to one")
    blended = H0_WEIGHT * h0_probability + NB_WEIGHT * nb_probability
    np.testing.assert_allclose(blended.sum(axis=1), 1.0, atol=1e-6)
    return blended.astype(np.float32)
