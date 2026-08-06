from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

import exact_event_pipeline as pipeline
import oof_validation as validation


def test_exact_crossfit_is_finite_and_row_aligned() -> None:
    matrix = sparse.csr_matrix(np.asarray([
        [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0],
        [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0],
    ], dtype=np.float32))
    labels = np.asarray(["A"] * 5 + ["B"] * 5)
    classes = np.asarray(["A", "B"])
    train, apply = pipeline.cross_fitted_exact_eb(
        matrix, matrix[:3], labels, classes, seed=42
    )
    assert train.shape == (10, 2)
    assert apply.shape == (3, 2)
    assert np.isfinite(train).all()
    assert np.isfinite(apply).all()


def test_frozen_model_and_decision_contract() -> None:
    assert validation.SEEDS == (42, 777, 2024)
    assert validation.OUTER_SPLITS == 5
    assert pipeline.SELECTIVE_MARGIN == 0.05
    assert pipeline.SELECTIVE_LR_WEIGHT == 0.80
    assert pipeline.H0_SPECIALIST_WEIGHT == 0.20
    model = validation._make_lgbm(seed=42, class_count=26)
    params = model.get_params()
    assert params["n_estimators"] == 400
    assert params["learning_rate"] == 0.05
    assert params["num_leaves"] == 25


def test_oof_runner_never_reads_test_or_uses_fixed_domain_identifiers() -> None:
    source = Path(validation.__file__).read_text(encoding="utf-8")
    assert "pd.read_csv" not in source
    assert "test.csv" not in source
    assert "concat([train" not in source
    for forbidden in ("V600E", "R132H", "KIPAN", "GBMLGG"):
        assert forbidden not in source
