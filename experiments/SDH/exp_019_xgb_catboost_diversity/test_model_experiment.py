from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import model_experiment as exp


def test_catalog_has_two_balanced_families_and_unique_names() -> None:
    cases = exp.case_catalog()
    assert len(cases) == 24
    assert len(cases) == len(set(cases))
    assert sum(case.family == "xgb" for case in cases.values()) == 12
    assert sum(case.family == "cat" for case in cases.values()) == 12
    assert {case.view for case in cases.values()} == {
        "full", "compact", "hybrid512", "hybrid1024"
    }


def test_feature_views_use_train_support_and_keep_signed_values() -> None:
    names = [
        "G__A", "G__B", "G__C", "B__event_count", "V__missense_event_count",
        "A_pair__0", "S__0", "C__A_vs_B_contrast", "E__gene_type__A",
    ]
    dense = np.asarray([
        [1, 0, 0, 1, 2, 0, 1, -1, -0.5],
        [1, 1, 0, 2, 1, 1, 0, 1, 0.5],
        [1, 0, 1, 1, 1, 0, 1, -1, 1.5],
    ], dtype=np.float32)
    x_fit = sparse.csr_matrix(dense)
    x_valid = sparse.csr_matrix(dense[:1])
    fit, valid, view_names = exp._feature_views(x_fit, x_valid, names)
    assert fit["full"].shape[1] == len(names)
    assert "G__A" not in view_names["compact"]
    assert "C__A_vs_B_contrast" in view_names["compact"]
    assert fit["compact"].data.min() < 0
    assert valid["compact"].shape[1] == fit["compact"].shape[1]


def test_auto_pair_discovery_uses_only_supplied_labels() -> None:
    mutation = sparse.csr_matrix(np.vstack([
        np.tile([1, 0, 0], (9, 1)),
        np.tile([0, 1, 0], (9, 1)),
        np.tile([0, 0, 1], (9, 1)),
    ]).astype(np.float32))
    labels = np.asarray(["A"] * 9 + ["B"] * 9 + ["C"] * 9)
    pairs = exp._discover_auto_pairs(mutation, labels, seed=42)
    assert pairs
    assert all(left in {"A", "B", "C"} and right in {"A", "B", "C"}
               for left, right, _ in pairs)
    assert all(top_k == exp.AUTO_GENES_PER_PAIR for _, _, top_k in pairs)


def test_balanced_sample_weight_equalises_class_mass() -> None:
    labels = np.asarray([0, 0, 0, 1])
    weights = exp._balanced_sample_weight(labels, 2)
    assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())


def test_source_has_no_raw_train_test_concat_or_fixed_domain_constants() -> None:
    source = open(exp.__file__, encoding="utf-8").read()
    assert "concat([train" not in source
    assert "concat([fit" not in source
    for forbidden in ("V600E", "R132H", "KIPAN", "GBMLGG"):
        assert forbidden not in source


def test_incremental_grid_can_reject_candidate_with_zero_weight() -> None:
    truth = np.asarray([0, 1, 2, 0, 1, 2])
    good = np.eye(3)[truth] * 0.8 + 0.2 / 3
    bad = np.roll(good, 1, axis=1)
    choice = exp._best_incremental_weights(good, good, bad, truth)
    assert choice["candidate_weight"] == 0.0


def test_fixed_incremental_metrics_counts_recovery_and_damage() -> None:
    classes = np.asarray(["A", "B"])
    labels = np.asarray(["A", "B", "A", "B"])

    def result(name, probability):
        probability = np.asarray(probability, dtype=float)
        return exp.OOFResult(
            name, name, 42, probability, probability.argmax(axis=1),
            classes, [0.0], [2], 0,
        )

    lr = result("lr", [[.8, .2], [.2, .8], [.4, .6], [.2, .8]])
    lgbm = result("lgbm", [[.8, .2], [.2, .8], [.4, .6], [.2, .8]])
    candidate = result("candidate", [[.8, .2], [.2, .8], [.9, .1], [.9, .1]])
    metrics = exp.fixed_incremental_metrics(
        lr, lgbm, candidate, labels, candidate_weight=0.5
    )
    assert metrics["base_recovered_count"] == 1
    assert metrics["base_damaged_count"] == 1
    assert metrics["base_net_correct_count"] == 0
