"""Train-only exact mutation-profile posterior retrieval."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from h0_faithful_pipeline import normalise_cell


PRIOR_STRENGTH = 1.0


@dataclass(frozen=True)
class ProfileLookup:
    classes: np.ndarray
    prior: np.ndarray
    counts: dict[str, np.ndarray]


def profile_key(row: pd.Series, genes: list[str]) -> str:
    """Canonical row-local key; WT, blank and NaN contribute no event."""
    events = []
    for gene in genes:
        events.extend(f"{gene}__{event}" for event in normalise_cell(row[gene]))
    return "\x1f".join(sorted(set(events)))


def build_profile_lookup(
    fit_frame: pd.DataFrame, labels: np.ndarray, genes: list[str], classes: np.ndarray
) -> ProfileLookup:
    class_index = {str(label): position for position, label in enumerate(classes)}
    counts: dict[str, np.ndarray] = {}
    for row_index in range(len(fit_frame)):
        key = profile_key(fit_frame.iloc[row_index], genes)
        if not key:
            continue
        if key not in counts:
            counts[key] = np.zeros(len(classes), dtype=np.float64)
        counts[key][class_index[str(labels[row_index])]] += 1.0
    prior = np.bincount([class_index[str(label)] for label in labels], minlength=len(classes)).astype(np.float64)
    prior /= prior.sum()
    return ProfileLookup(classes=np.asarray(classes), prior=prior, counts=counts)


def query_profile_posteriors(
    apply_frame: pd.DataFrame, genes: list[str], lookup: ProfileLookup
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    posterior = np.tile(lookup.prior, (len(apply_frame), 1))
    matched = np.zeros(len(apply_frame), dtype=bool)
    support = np.zeros(len(apply_frame), dtype=np.int32)
    purity = np.zeros(len(apply_frame), dtype=np.float64)
    for row_index in range(len(apply_frame)):
        count = lookup.counts.get(profile_key(apply_frame.iloc[row_index], genes))
        if count is None:
            continue
        total = float(count.sum())
        matched[row_index] = True
        support[row_index] = int(total)
        purity[row_index] = float(count.max() / total) if total else 0.0
        posterior[row_index] = (count + PRIOR_STRENGTH * lookup.prior) / (total + PRIOR_STRENGTH)
    return posterior, matched, support, purity


def fixed_profile_blend(
    h0_probability: np.ndarray, posterior: np.ndarray, matched: np.ndarray, *, weight: float = .20
) -> np.ndarray:
    output = h0_probability.copy()
    output[matched] = (1.0 - weight) * h0_probability[matched] + weight * posterior[matched]
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)
    return output
