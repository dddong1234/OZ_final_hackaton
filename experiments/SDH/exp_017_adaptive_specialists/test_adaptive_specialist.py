from __future__ import annotations

import numpy as np
import pandas as pd

import adaptive_specialist as adaptive


def test_catalog_is_unique_and_contains_no_fixed_class_names() -> None:
    cases = adaptive.case_catalog()
    assert len(cases) == 144
    assert len(cases) == len(set(cases))
    text = repr(cases)
    for forbidden in ("KIRC", "KIPAN", "LGG", "GBMLGG"):
        assert forbidden not in text


def test_capacity_selection() -> None:
    assert adaptive.select_preset("fixed_sl", 0, 999) == "small"
    assert adaptive.select_preset("fixed_sl", 1, 1) == "large"
    assert adaptive.select_preset("support_100", 0, 100) == "small"
    assert adaptive.select_preset("support_100", 0, 101) == "large"


def test_pair_margin_routes_only_uncertain_pair_predictions() -> None:
    probability = np.array(
        [[0.45, 0.40, 0.15], [0.80, 0.10, 0.10], [0.30, 0.25, 0.45]]
    )
    classes = np.array(["A", "B", "C"])
    prediction = classes[probability.argmax(axis=1)]
    mask = adaptive._routing_mask(
        probability, prediction, ("A", "B"), [0, 1], "pair_margin", 0.20
    )
    assert mask.tolist() == [True, False, False]


def test_global_margin_routes_only_low_margin_pair_predictions() -> None:
    probability = np.array(
        [[0.45, 0.40, 0.15], [0.80, 0.10, 0.10], [0.30, 0.25, 0.45]]
    )
    classes = np.array(["A", "B", "C"])
    prediction = classes[probability.argmax(axis=1)]
    mask = adaptive._routing_mask(
        probability, prediction, ("A", "B"), [0, 1], "global_margin", 0.10
    )
    assert mask.tolist() == [True, False, False]


def test_apply_case_preserves_probability_and_uses_cached_specialist() -> None:
    classes = np.array(["A", "B"])
    labels = np.array(["A", "B"])
    main_probability = np.array([[0.8, 0.2], [0.2, 0.8]])
    fold_metrics = pd.DataFrame(
        [{"fold": 1, "f1_macro": 1.0, "accuracy": 1.0, "feature_count": 2, "elapsed_seconds": 0.0}]
    )
    main = adaptive.exp14.OOFResult(
        "main", 42, classes, main_probability, classes[main_probability.argmax(axis=1)],
        fold_metrics, {"feature_count_mean": 2.0},
    )
    specialist_probability = np.array([[0.1, 0.9], [0.9, 0.1]])
    bank = [
        adaptive.SpecialistBankFold(
            1, np.array([0, 1]), (("A", "B"),), (2,),
            {name: (specialist_probability,) for name in ("small", "medium", "large")},
        )
    ]
    case = adaptive.AdaptiveCase("test", "fixed_ll", "hard")
    result = adaptive.apply_case(main, bank, labels, case)
    np.testing.assert_allclose(result.probability.sum(axis=1), 1.0)
    assert result.prediction.tolist() == ["B", "A"]
