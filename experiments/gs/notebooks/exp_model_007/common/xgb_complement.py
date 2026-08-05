"""Fixed XGBoost complement contract for the faithful H0 experiment."""
from __future__ import annotations

import numpy as np

H0_WEIGHT = 0.80
XGB_WEIGHT = 0.20


def fixed_blend(h0_probability: np.ndarray, xgb_probability: np.ndarray) -> np.ndarray:
    """Apply the predeclared H0/XGB probability blend without tuning."""
    blended = H0_WEIGHT * np.asarray(h0_probability, dtype=np.float64) + XGB_WEIGHT * np.asarray(xgb_probability, dtype=np.float64)
    return (blended / blended.sum(axis=1, keepdims=True)).astype(np.float32)


def xgb_config(*, seed: int, class_count: int) -> dict:
    """One regularized, shallow multiclass XGB configuration; no search grid."""
    return {
        "objective": "multi:softprob",
        "num_class": int(class_count),
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 5.0,
        "subsample": 0.80,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.25,
        "reg_lambda": 5.0,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": -1,
        "verbosity": 0,
    }
