import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).parent))

from h0_faithful_pipeline import (  # noqa: E402
    ENRICHMENT_MIN_SUPPORT,
    RECURRENT_MIN_COUNT,
    fit_vocabulary,
    make_h0_fold_matrices,
    normalise_cell,
)


class H0FaithfulContractTest(unittest.TestCase):
    def test_safe_parser_never_turns_nan_or_wt_into_events(self):
        self.assertEqual(normalise_cell(np.nan), ())
        self.assertEqual(normalise_cell("WT"), ())
        self.assertEqual(normalise_cell("  "), ())

    def test_reference_constants_are_locked(self):
        self.assertEqual(RECURRENT_MIN_COUNT, 5)
        self.assertEqual(ENRICHMENT_MIN_SUPPORT, 10)

    def test_validation_only_event_cannot_create_fit_vocabulary_column(self):
        fit = pd.DataFrame({"G1": ["R1H"] * 10 + ["R2H"] * 10})
        valid = pd.DataFrame({"G1": ["Q2W", "Q2W"]})
        labels = np.asarray(["A"] * 10 + ["B"] * 10)
        vocabulary = fit_vocabulary(fit, ["G1"])
        self.assertNotIn("G1__Q2W", vocabulary.exact_events)
        x_fit, x_valid, names, audit = make_h0_fold_matrices(fit, valid, labels, ["G1"], seed=42)
        self.assertEqual(x_fit.shape[1], x_valid.shape[1])
        self.assertTrue(audit["vocabulary_source_fit_only"])
        self.assertFalse(any("Q2W" in name for name in names))


if __name__ == "__main__":
    unittest.main()
