"""Minimal regression check for the fixed One-vs-Rest LR configuration."""

from pathlib import Path
import importlib.util
import sys

import numpy as np


RUNNER = Path(__file__).with_name("exp-gs-002-memory-safe.py")
SPEC = importlib.util.spec_from_file_location("exp_gs_002_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_ovr_logistic_configuration() -> None:
    model = MODULE.make_model("logistic", seed=42, max_iter=2000, multi_class="ovr")
    assert type(model).__name__ == "OneVsRestClassifier"
    assert model.estimator.solver == "lbfgs"
    assert model.estimator.C == 0.07
    assert model.estimator.max_iter == 2000
    assert model.estimator.class_weight == "balanced"


def test_fixed_three_way_blend_uses_declared_weights_and_preserves_probability_mass() -> None:
    primary = np.array([[0.70, 0.20, 0.10], [0.10, 0.80, 0.10]])
    multinomial = np.array([[0.20, 0.30, 0.50], [0.40, 0.20, 0.40]])
    ovr = np.array([[0.10, 0.50, 0.40], [0.20, 0.30, 0.50]])
    actual = MODULE.fixed_three_way_probability_blend(primary, multinomial, ovr)
    expected = 0.50 * primary + 0.25 * multinomial + 0.25 * ovr
    assert np.allclose(actual, expected)
    assert np.allclose(actual.sum(axis=1), 1.0)


def test_event_tfidf_ovr_submission_helper_exists() -> None:
    assert callable(MODULE.make_event_tfidf_ovr_submission)


if __name__ == "__main__":
    test_ovr_logistic_configuration()
    test_fixed_three_way_blend_uses_declared_weights_and_preserves_probability_mass()
    test_event_tfidf_ovr_submission_helper_exists()
    print("event TF-IDF OVR model configuration test passed")
