"""Contracts for fold-safe sparse boosting feature variants."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy import sparse


RUNNER = Path(__file__).with_name("exp_sparse_boosting_runner.py")
SPEC = importlib.util.spec_from_file_location("exp_sparse_boosting_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_gene_selection_does_not_use_validation_rows() -> None:
    mutation = sparse.csr_matrix(np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],  # validation-only gene 2 must not affect selection
    ], dtype=np.float32))
    labels = np.array(["A", "A", "B", "B"])

    selected = MODULE.select_top_genes(mutation, labels, np.array([0, 1, 2]), top_k=1)

    assert selected.tolist() == [0]


def test_reduced_mask_uses_original_gene_indices_when_some_raw_genes_are_inactive() -> None:
    names = ["G__B", "G__C", "B__events", "A_pair__0", "C__KIRC_KIPAN_count"]
    genes = ["A", "B", "C"]
    keep = MODULE.reduced_feature_mask(names, genes, np.array([1]))

    assert keep.tolist() == [True, False, True, True, True]


def test_runner_declares_train_only_oof_input() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"train.csv"' in source
    assert '"test.csv"' not in source


if __name__ == "__main__":
    test_gene_selection_does_not_use_validation_rows()
    test_reduced_mask_uses_original_gene_indices_when_some_raw_genes_are_inactive()
    test_runner_declares_train_only_oof_input()
    print("Sparse boosting contracts passed")
