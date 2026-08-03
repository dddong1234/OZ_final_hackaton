"""Contracts for the train-only class prototype probability model."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix


RUNNER = Path(__file__).with_name("exp_profile_model_runner.py")
SPEC = importlib.util.spec_from_file_location("exp_profile_model_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_prototype_probability_is_normalized_and_prefers_matching_profile() -> None:
    matrix = csr_matrix([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    target = np.array(["A", "A", "B", "B"])
    model = MODULE.PrototypeCosine(temperature=0.10).fit(matrix, target)
    probability = model.predict_proba(csr_matrix([[1, 0], [0, 1]], dtype=float))
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert list(model.classes_[probability.argmax(axis=1)]) == ["A", "B"]


if __name__ == "__main__":
    test_prototype_probability_is_normalized_and_prefers_matching_profile()
    print("Profile-model contracts passed")
