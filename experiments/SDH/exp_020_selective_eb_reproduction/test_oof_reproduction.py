from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import oof_reproduction as exp
import provided_pipeline as provided


def test_frozen_selective_gate_uses_non_eb_only_below_margin() -> None:
    non_eb = np.asarray([[.8, .2], [.2, .8]])
    eb = np.asarray([[.51, .49], [.1, .9]])
    selected, use_non_eb = provided.selective_probability(non_eb, eb)
    assert use_non_eb.tolist() == [True, False]
    np.testing.assert_allclose(selected[0], non_eb[0])
    np.testing.assert_allclose(selected[1], eb[1])


def test_frozen_branch_replacement_is_80_20() -> None:
    lr = np.asarray([[.8, .2], [.3, .7]])
    specialist = np.asarray([[.4, .6], [.9, .1]])
    actual = provided.fixed_branch_replacement(lr, specialist)
    np.testing.assert_allclose(actual, .8 * lr + .2 * specialist, atol=1e-7)


def test_empirical_bayes_crossfit_is_finite_and_row_aligned() -> None:
    rows = []
    for index in range(20):
        rows.append([1, 0, 1] if index < 10 else [0, 1, 1])
    matrix = sparse.csr_matrix(np.asarray(rows, dtype=np.float32))
    labels = np.asarray(["A"] * 10 + ["B"] * 10)
    classes = np.asarray(["A", "B"])
    fit, apply = provided.cross_fitted_eb_scores(
        matrix, matrix[:4], labels, classes, seed=42
    )
    assert fit.shape == (20, 2)
    assert apply.shape == (4, 2)
    assert np.isfinite(fit).all() and np.isfinite(apply).all()


def _dummy_result(seed: int) -> exp.SeedResult:
    labels = np.asarray(["A", "B", "A", "B"])
    classes = np.asarray(["A", "B"])
    probability = np.asarray([
        [.8, .2], [.2, .8], [.7, .3], [.3, .7]
    ])
    return exp.SeedResult(
        seed=seed,
        ids=np.asarray(["I0", "I1", "I2", "I3"]),
        classes=classes,
        labels=labels,
        probabilities={name: probability.copy() for name in exp.VARIANTS},
        fold_metrics=pd.DataFrame(),
        class_metrics=pd.DataFrame(),
        audit=pd.DataFrame({"leakage_check": [True]}),
        convergence_warning_count=0,
        runtime_minutes=0.0,
    )


def test_aggregate_reports_mean_and_probability_average_separately() -> None:
    per_seed, summary = exp.aggregate_results([
        _dummy_result(42), _dummy_result(777), _dummy_result(2024)
    ])
    assert per_seed.seed.nunique() == 3
    final = summary[
        summary.variant == "final_selective_eb_specialist"
    ].iloc[0]
    assert final.per_seed_f1_mean == 1.0
    assert final.probability_averaged_oof_f1 == 1.0


def test_model_contract_is_frozen() -> None:
    model = exp._make_lgbm(seed=42, class_count=26)
    parameters = model.get_params()
    assert parameters["n_estimators"] == 400
    assert parameters["learning_rate"] == 0.05
    assert parameters["num_leaves"] == 25
    assert parameters["class_weight"] == "balanced"
    assert provided.SELECTIVE_MARGIN == 0.05
    assert provided.SELECTIVE_LR_WEIGHT == 0.80
    assert provided.H0_SPECIALIST_WEIGHT == 0.20


def test_sources_have_no_test_fit_or_fixed_domain_constants() -> None:
    evaluator = Path(exp.__file__).read_text(encoding="utf-8")
    supplied = Path(provided.__file__).read_text(encoding="utf-8")
    assert "test.csv" not in evaluator
    assert "concat([train" not in supplied
    for forbidden in ("V600E", "R132H", "KIPAN", "GBMLGG"):
        assert forbidden not in supplied

