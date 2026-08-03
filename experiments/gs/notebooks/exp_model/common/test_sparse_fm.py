"""Minimal sparse FM contract: zero-valued padded entries must not affect logits."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


RUNNER = Path(__file__).with_name("sparse_fm_runner.py")
SPEC = importlib.util.spec_from_file_location("sparse_fm_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sparse_fm_ignores_padding_values() -> None:
    torch.manual_seed(7)
    model = MODULE.SparseMulticlassFM(n_features=8, n_classes=3, rank=4)
    indices = torch.tensor([[1, 3, 0], [2, 0, 0]], dtype=torch.long)
    values = torch.tensor([[1.0, 2.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    baseline = model(indices, values)
    padded_indices = torch.tensor([[1, 3, 7], [2, 6, 5]], dtype=torch.long)
    assert torch.allclose(baseline, model(padded_indices, values))


def test_cache_build_groups_row_and_gene_columns() -> None:
    """Multiple events in one gene must produce row/gene topology aggregates."""
    frame = pd.DataFrame(
        {"TP53": ["R175H, R248Q", "WT"], "IDH1": ["WT", "R132H"]}
    )
    cache = MODULE.Cache.build(frame, ["TP53", "IDH1"])
    assert cache.burden.shape == (2, 3)
    assert cache.topology.shape == (2, 8)
    assert cache.burden[0, 2] == 1


def test_probability_alignment_writes_validation_by_class_block() -> None:
    output = np.zeros((4, 3), dtype=np.float32)
    rows = np.array([1, 3])
    columns = [2, 0]
    probability = np.array([[0.8, 0.2], [0.7, 0.3]], dtype=np.float32)
    MODULE.assign_probability(output, rows, columns, probability)
    assert np.allclose(output[1], [0.2, 0.0, 0.8])
    assert np.allclose(output[3], [0.3, 0.0, 0.7])


if __name__ == "__main__":
    test_sparse_fm_ignores_padding_values()
    test_cache_build_groups_row_and_gene_columns()
    test_probability_alignment_writes_validation_by_class_block()
    print("sparse FM padding contract passed")
