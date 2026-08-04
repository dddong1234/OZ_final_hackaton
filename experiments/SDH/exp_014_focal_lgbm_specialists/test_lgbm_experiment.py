from __future__ import annotations

import numpy as np
import pandas as pd

import lgbm_experiment as exp


def test_focal_objective_is_finite_and_positive_hessian() -> None:
    objective = exp.make_focal_objective(3, gamma=1.0, alpha=0.25)
    y_true = np.array([0, 1, 2, 1])
    raw = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.1, 0.2, 0.3],
            [-0.5, 0.2, 1.1],
            [2.0, -1.0, 0.0],
        ]
    )
    gradient, hessian = objective(y_true, raw)
    assert gradient.shape == raw.shape
    assert hessian.shape == raw.shape
    assert np.isfinite(gradient).all()
    assert np.isfinite(hessian).all()
    assert (hessian > 0).all()


def test_softmax_rows_sum_to_one() -> None:
    raw = np.array([[1000.0, 999.0], [-1000.0, -999.0]])
    probability = exp._softmax(raw)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert np.isfinite(probability).all()


def test_fixed_blend_preserves_probability_contract() -> None:
    labels = np.array(["A", "B", "A"])
    classes = np.array(["A", "B"])
    fold_metrics = pd.DataFrame(
        [{"fold": 1, "f1_macro": 0.5, "accuracy": 0.5, "feature_count": 2, "elapsed_seconds": 0.0}]
    )
    lr_probability = np.array([[0.8, 0.2], [0.4, 0.6], [0.6, 0.4]])
    model_probability = np.array([[0.6, 0.4], [0.2, 0.8], [0.3, 0.7]])
    lr = exp.OOFResult(
        "lr",
        42,
        classes,
        lr_probability,
        classes[lr_probability.argmax(axis=1)],
        fold_metrics,
        {"oof_f1_macro": 1.0},
    )
    model = exp.OOFResult(
        "model",
        42,
        classes,
        model_probability,
        classes[model_probability.argmax(axis=1)],
        fold_metrics,
        {"oof_f1_macro": 0.6},
    )
    result = exp.fixed_blends(lr, model, labels)
    assert list(result["model_weight"]) == list(exp.BLEND_MODEL_WEIGHTS)
    assert np.isfinite(result["f1_macro"]).all()
