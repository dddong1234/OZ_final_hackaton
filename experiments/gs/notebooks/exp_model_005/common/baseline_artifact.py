"""Validated optional OOF baseline artifacts for bounded experiment runners."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TEAM_REFERENCE_MACRO_F1 = 0.54202


@dataclass(frozen=True)
class BaselineReference:
    probabilities: np.ndarray | None
    reference_macro_f1: float
    comparison_mode: str


def reference_only_baseline() -> BaselineReference:
    """Use the recorded team score without rerunning its costly base models."""
    return BaselineReference(None, TEAM_REFERENCE_MACRO_F1, "unpaired_reference")


def validate_baseline_oof(frame: pd.DataFrame, classes: np.ndarray, row_count: int) -> BaselineReference:
    """Validate row-aligned probabilities produced by an already fold-safe runner."""
    expected = [f"prob__{label}" for label in classes]
    if len(frame) != row_count:
        raise ValueError(f"baseline OOF row count must be {row_count}, got {len(frame)}")
    observed = [column for column in frame.columns if column.startswith("prob__")]
    if observed != expected:
        raise ValueError("baseline OOF probability columns must exactly follow the train class order")
    probability = frame[expected].to_numpy(dtype=np.float32)
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("baseline OOF probabilities must be finite and non-negative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("baseline OOF probability rows must sum to one")
    return BaselineReference(probability, TEAM_REFERENCE_MACRO_F1, "paired_artifact")
