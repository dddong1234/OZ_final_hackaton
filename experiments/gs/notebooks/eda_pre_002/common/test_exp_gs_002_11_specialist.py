import importlib.util
import sys
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).with_name("exp-gs-002-memory-safe.py")
SPEC = importlib.util.spec_from_file_location("exp_gs_002_runner_test", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_soft_pair_blend_preserves_pair_mass_and_other_classes():
    primary = np.array([[0.10, 0.20, 0.70], [0.45, 0.05, 0.50]], dtype=np.float64)
    expert_kirc_probability = np.array([0.80, 0.20], dtype=np.float64)

    blended = MODULE.soft_blend_pair_probabilities(
        primary_probability=primary,
        left_index=0,
        right_index=1,
        expert_left_probability=expert_kirc_probability,
        alpha=0.30,
    )

    np.testing.assert_allclose(blended.sum(axis=1), 1.0)
    np.testing.assert_allclose(blended[:, 0] + blended[:, 1], primary[:, 0] + primary[:, 1])
    np.testing.assert_allclose(blended[:, 2], primary[:, 2])
    assert blended[0, 0] > primary[0, 0]
