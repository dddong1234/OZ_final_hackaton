"""Fold-safe utilities for automatically validated binary pair specialists."""
from __future__ import annotations

import numpy as np


def top_two_pair_mask(probability: np.ndarray, classes: np.ndarray, pair: tuple[str, str]) -> np.ndarray:
    """Route only rows whose base-model top two labels equal the pair."""
    probability = np.asarray(probability, dtype=np.float64)
    classes = np.asarray(classes, dtype=object)
    indices = np.argpartition(probability, -2, axis=1)[:, -2:]
    labels = np.sort(classes[indices].astype(str), axis=1)
    target = np.asarray(sorted(pair), dtype=str)
    return np.all(labels == target, axis=1)


def apply_pair_probability(
    base_probability: np.ndarray,
    specialist_probability: np.ndarray,
    classes: np.ndarray,
    pair: tuple[str, str],
    *,
    route: np.ndarray | None = None,
) -> np.ndarray:
    """Replace only pair-internal proportions while preserving pair mass."""
    base = np.asarray(base_probability, dtype=np.float64)
    specialist = np.asarray(specialist_probability, dtype=np.float64)
    classes = np.asarray(classes, dtype=object)
    if specialist.shape != (len(base), 2):
        raise ValueError("specialist probability must have shape (n_rows, 2)")
    lookup = {str(label): index for index, label in enumerate(classes)}
    columns = [lookup[pair[0]], lookup[pair[1]]]
    route = top_two_pair_mask(base, classes, pair) if route is None else np.asarray(route, dtype=bool)
    output = base.copy()
    mass = base[:, columns].sum(axis=1)
    output[np.ix_(route, columns)] = mass[route, None] * specialist[route]
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(output[:, columns].sum(axis=1), mass, atol=1e-6)
    return output


def select_non_overlapping_pairs(candidates: list[dict], *, maximum: int = 2) -> list[tuple[str, str]]:
    """Greedily retain evidence-positive pairs with no shared label."""
    selected: list[tuple[str, str]] = []
    used: set[str] = set()
    ordered = sorted(candidates, key=lambda row: (-float(row["pair_f1_delta"]), -(int(row["recovered"]) - int(row["broken"])), tuple(row["pair"])))
    for row in ordered:
        pair = tuple(row["pair"])
        if float(row["pair_f1_delta"]) <= 0 or int(row["recovered"]) <= int(row["broken"]):
            continue
        if set(pair) & used:
            continue
        selected.append(pair)
        used.update(pair)
        if len(selected) == maximum:
            break
    return selected
