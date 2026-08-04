import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from h0_faithful_pipeline import (  # noqa: E402
    build_design_matrices,
    fit_vocabulary,
    normalise_cell,
)


class H0FaithfulPipelineTest(unittest.TestCase):
    def test_nan_wt_and_blank_are_not_events(self):
        self.assertEqual(normalise_cell(np.nan), ())
        self.assertEqual(normalise_cell("WT"), ())
        self.assertEqual(normalise_cell("  "), ())
        self.assertEqual(normalise_cell("p.R1H R2*"), ("R1H", "R2*"))

    def test_vocabulary_is_fit_only_and_recurrent_is_missense_only(self):
        fit = pd.DataFrame({"G1": ["R1H"] * 5 + ["R2*"] * 5, "G2": ["WT"] * 10})
        apply = pd.DataFrame({"G1": ["Q3W"], "G2": ["WT"]})
        vocabulary = fit_vocabulary(fit, ["G1", "G2"])
        self.assertNotIn("G1__Q3W", vocabulary.exact_events)
        x_fit, x_apply, names, audit = build_design_matrices(fit, apply, np.asarray(["A"] * 5 + ["B"] * 5), ["G1", "G2"], seed=42)
        self.assertTrue(any(name == "R__G1__R1H" for name in names))
        self.assertFalse(any(name == "R__G1__R2*" for name in names))
        self.assertEqual(x_apply.shape[1], x_fit.shape[1])
        self.assertEqual(audit["vocabulary_source"], "fit_frame_only")

    def test_feature_contract_contains_burden_variant_apair_topology_and_standardized_enrichment(self):
        labels = np.asarray(["A", "B", "C"] * 20)
        frame = pd.DataFrame({"G1": [{"A": "R1H", "B": "R2*", "C": "Q3W"}[label] for label in labels], "G2": [{"A": "WT", "B": "R4H", "C": "Q6W"}[label] for label in labels]})
        x_fit, _, names, audit = build_design_matrices(frame, frame.iloc[:2], labels, ["G1", "G2"], seed=7)
        self.assertTrue(any(name.startswith("B__") for name in names))
        self.assertTrue(any(name.startswith("V__") for name in names))
        self.assertEqual(audit["pre_filter_block_counts"]["amino_pair"], 380)
        self.assertEqual(audit["pre_filter_block_counts"]["topology"], 8)
        self.assertEqual(audit["pre_filter_block_counts"]["enrichment"], 3)
        self.assertEqual(audit["enrichment_inner_splits"], 5)
        self.assertEqual(x_fit.shape[1], len(names))


if __name__ == "__main__":
    unittest.main()
