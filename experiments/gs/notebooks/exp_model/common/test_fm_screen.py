"""Contracts for the seed-42 FM candidate screening runner."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


RUNNER = Path(__file__).with_name("fm_screen_runner.py")
SPEC = importlib.util.spec_from_file_location("fm_screen_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sqrt_balanced_weight_is_less_extreme_than_balanced() -> None:
    encoded = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    balanced = MODULE.class_weight_vector(encoded, 2, "balanced")
    sqrt_balanced = MODULE.class_weight_vector(encoded, 2, "sqrt_balanced")
    none = MODULE.class_weight_vector(encoded, 2, "none")
    assert balanced[1] / balanced[0] > sqrt_balanced[1] / sqrt_balanced[0] > 1.0
    assert np.allclose(none, [1.0, 1.0])


def test_candidate_stages_follow_the_fixed_screening_grid() -> None:
    stages = MODULE.candidate_stages()
    assert [stage for stage, _ in stages] == ["O1_learning_rate", "O2_rank", "O3_class_weight"]
    assert [candidate.learning_rate for candidate in stages[0][1]] == [3e-4, 1e-3, 3e-3]
    assert [candidate.rank for candidate in stages[1][1]] == [4, 8, 16]


def test_select_winner_reads_rank_column_not_series_rank_method() -> None:
    rows = [
        {"candidate_id": "low", "kind": "fm", "rank": 4, "learning_rate": 3e-4,
         "class_weight": "balanced", "paired_delta_0p25_oof": 0.001, "paired_delta_0p25_fold_min": 0.0},
        {"candidate_id": "high", "kind": "fm", "rank": 8, "learning_rate": 1e-3,
         "class_weight": "none", "paired_delta_0p25_oof": 0.002, "paired_delta_0p25_fold_min": 0.0},
    ]
    winner = MODULE.select_winner(rows)
    assert winner.candidate_id == "high"
    assert winner.rank == 8


if __name__ == "__main__":
    test_sqrt_balanced_weight_is_less_extreme_than_balanced()
    test_candidate_stages_follow_the_fixed_screening_grid()
    test_select_winner_reads_rank_column_not_series_rank_method()
    print("FM screening contracts passed")
