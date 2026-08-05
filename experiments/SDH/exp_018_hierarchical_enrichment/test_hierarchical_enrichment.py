from __future__ import annotations

import numpy as np
import pandas as pd

import hierarchical_enrichment as exp


GENES = ["IDH1", "TP53"]


def _frame(rows: list[tuple[object, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=GENES)


def test_case_catalog_is_fixed_and_contains_safe_baseline() -> None:
    cases = exp.case_catalog()
    assert len(cases) == 17
    assert len(cases) == len(set(cases))
    assert exp.SAFE_BASELINE_CASE in cases
    assert all(case.min_support > 0 for case in cases.values())


def test_apply_only_token_cannot_change_train_vocabulary() -> None:
    train = _frame([("R132H", "WT"), ("R132H", "R175H"), ("WT", "R248Q")])
    apply_a = _frame([("R140Q", "WT")])
    apply_b = _frame([("R999W", "R999W")])
    parent_names = ("IDH1__MISSENSE", "TP53__MISSENSE")

    pair_a = exp.fit_transform_fine_tokens(
        train, apply_a, GENES, parent_names, kind="amino"
    )
    pair_b = exp.fit_transform_fine_tokens(
        train, apply_b, GENES, parent_names, kind="amino"
    )

    assert pair_a.names == pair_b.names
    assert (pair_a.train != pair_b.train).nnz == 0
    assert "IDH1__R>Q" not in pair_a.names
    assert "IDH1__R>W" not in pair_a.names


def test_position_tokens_have_train_parent_mapping() -> None:
    train = _frame([("R132H", "WT"), ("R175H", "R248Q")])
    apply = _frame([("R140Q", "R273H")])
    parent_names = ("IDH1__MISSENSE", "TP53__MISSENSE")
    pair = exp.fit_transform_fine_tokens(
        train,
        apply,
        GENES,
        parent_names,
        kind="position",
        position_scheme="p50",
    )
    assert pair.train.shape[1] == len(pair.names)
    assert pair.apply.shape[1] == len(pair.names)
    assert np.array_equal(np.unique(pair.parent_columns), np.array([0, 1]))


def test_hierarchical_residual_scores_are_finite() -> None:
    fine = exp.sparse.csr_matrix(
        np.asarray(
            [
                [1, 0], [1, 0], [0, 1], [0, 1],
                [1, 0], [0, 1], [1, 0], [0, 1],
            ],
            dtype=np.float32,
        )
    )
    parent = exp.sparse.csr_matrix(np.ones((8, 1), dtype=np.float32))
    labels = np.asarray(["A", "A", "B", "B", "A", "B", "A", "B"])
    selected, weights = exp._fit_residual(
        fine,
        parent,
        np.asarray([0, 0]),
        labels,
        ["A", "B"],
        min_support=2,
        backoff=10.0,
    )
    assert len(selected) == 2
    assert np.isfinite(weights).all()
    assert np.abs(weights).max() <= exp.WEIGHT_CLIP


def test_source_has_no_raw_train_apply_concat_path() -> None:
    source = open(exp.__file__, encoding="utf-8").read()
    assert "concat([train" not in source
    assert "concat([fit" not in source
    assert "raw_train_apply_concat\": False" in source

