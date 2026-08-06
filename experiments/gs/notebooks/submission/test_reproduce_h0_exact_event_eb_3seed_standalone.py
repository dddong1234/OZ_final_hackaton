"""Fast contract tests for the portable Exact-event EB submission script."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("reproduce_h0_exact_event_eb_3seed_standalone.py")
SPEC = spec_from_file_location("exact_event_standalone", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_missing_and_wt_are_not_events() -> None:
    assert MODULE.normalise_cell(np.nan) == ()
    assert MODULE.normalise_cell("") == ()
    assert MODULE.normalise_cell("WT") == ()
    assert MODULE.normalise_cell("p.R132H; R132H") == ("R132H",)


def test_exact_vocabulary_is_fit_only() -> None:
    genes = ["G1"]
    fit = pd.DataFrame({"G1": ["R1H", "WT"]})
    apply = pd.DataFrame({"G1": ["V2E"]})
    vocabulary = MODULE.fit_vocabulary(fit, genes)
    projected = MODULE.transform_rows(apply, genes, vocabulary)
    assert vocabulary.exact_events == ("G1__R1H",)
    assert projected.exact.shape == (1, 1)
    assert projected.exact.nnz == 0


def test_seed_average_preserves_probability_contract() -> None:
    first = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32)
    second = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
    averaged = MODULE.average_seed_probabilities([first, second])
    assert np.allclose(averaged.sum(axis=1), 1.0)
    assert np.allclose(averaged[0], [0.6, 0.4])
