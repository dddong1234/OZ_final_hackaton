from __future__ import annotations

import numpy as np
import pandas as pd

import standalone_pipeline as pipe


GENES = ["BRAF", "IDH1", "TP53"]


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BRAF": ["V600E", "WT", "A10V", "WT"],
            "IDH1": ["WT", "R132H", "WT", "R132H"],
            "TP53": ["R175H", "WT", "R175H", "WT"],
        }
    )


def test_test_only_tokens_do_not_change_training_state() -> None:
    train = _train_frame()
    test_a = pd.DataFrame(
        {"BRAF": ["WT"], "IDH1": ["WT"], "TP53": ["WT"]}
    )
    test_b = pd.DataFrame(
        {
            "BRAF": ["TEST_ONLY_A999Z"],
            "IDH1": ["WT"],
            "TP53": ["WT"],
        }
    )

    train_a, _, vocabulary_a = pipe.fit_transform_pair(train, test_a, GENES)
    train_b, _, vocabulary_b = pipe.fit_transform_pair(train, test_b, GENES)

    assert vocabulary_a == vocabulary_b
    assert "BRAF__TEST_ONLY_A999Z" not in vocabulary_b.exact_events
    assert (train_a.exact != train_b.exact).nnz == 0
    assert (train_a.gene_type != train_b.gene_type).nnz == 0


def test_row_order_does_not_change_training_state() -> None:
    train = _train_frame()
    test = train.iloc[[3, 1, 0]].reset_index(drop=True)
    reversed_test = test.iloc[::-1].reset_index(drop=True)

    train_a, _, vocabulary_a = pipe.fit_transform_pair(train, test, GENES)
    train_b, _, vocabulary_b = pipe.fit_transform_pair(
        train, reversed_test, GENES
    )

    assert vocabulary_a == vocabulary_b
    np.testing.assert_array_equal(train_a.burden, train_b.burden)
    assert (train_a.mutation != train_b.mutation).nnz == 0


def test_normalisation_and_type_classification() -> None:
    assert pipe.normalise_cell(np.nan) == ()
    assert pipe.normalise_cell("WT") == ()
    assert pipe.normalise_cell("p.V600E; V600E") == ("V600E",)
    assert pipe.classify_event("V600E") == "MISSENSE"
    assert pipe.classify_event("R100*") == "NONSENSE"
    assert pipe.classify_event("Q10FS") == "FRAMESHIFT"


def test_shared_baseline_has_no_fixed_domain_columns() -> None:
    train = pd.concat([_train_frame()] * 5, ignore_index=True)
    labels = np.tile(np.array(["A", "B", "A", "B"]), 5)
    x_train, x_apply, names, audit = pipe.build_design_matrices(
        train, train.iloc[:2].copy(), labels, GENES, seed=42
    )
    assert x_train.shape[1] == len(names)
    assert x_apply.shape[1] == len(names)
    assert not any(name.startswith(("C__", "D__exact_")) for name in names)
    assert audit["fixed_contrast_enabled"] is False
    assert audit["fixed_exact_event_enabled"] is False


def test_fixed_contrast_cannot_be_reenabled() -> None:
    train = pd.concat([_train_frame()] * 5, ignore_index=True)
    labels = np.tile(np.array(["A", "B", "A", "B"]), 5)
    try:
        pipe.build_design_matrices(
            train,
            train.iloc[:2].copy(),
            labels,
            GENES,
            seed=42,
            use_fixed_contrast=True,
        )
    except ValueError as error:
        assert "removed" in str(error)
    else:
        raise AssertionError("fixed contrast must not be available")
