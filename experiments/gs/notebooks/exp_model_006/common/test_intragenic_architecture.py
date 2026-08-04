import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from intragenic_architecture import architecture_token_sets, cross_fitted_architecture_scores  # noqa: E402


class IntragenicArchitectureTest(unittest.TestCase):
    def test_multi_event_tokens_are_gene_specific_and_nan_safe(self):
        frame = pd.DataFrame({
            "G1": ["R10H R11H", np.nan, "WT"],
            "G2": ["R20*", "R21H R21*", ""],
        })
        token_sets = architecture_token_sets(frame, ["G1", "G2"])
        self.assertIn("G1__EVENT_COUNT_2PLUS", token_sets[0])
        self.assertIn("G1__MULTI_MISSENSE", token_sets[0])
        self.assertIn("G2__MISSENSE_PLUS_TRUNCATING", token_sets[1])
        self.assertEqual(token_sets[2], set())

    def test_same_position_requires_distinct_events(self):
        frame = pd.DataFrame({"G1": ["R10H R10K", "R10H R11H"]})
        token_sets = architecture_token_sets(frame, ["G1"])
        self.assertIn("G1__SAME_POSITION_MULTI_EVENT", token_sets[0])
        self.assertNotIn("G1__SAME_POSITION_MULTI_EVENT", token_sets[1])

    def test_cross_fitted_scores_have_no_validation_labels_in_fit(self):
        token_sets = [{"A"}, {"B"}] * 6
        labels = np.array(["X", "Y"] * 6)
        scores, applied, names, vocabulary_size = cross_fitted_architecture_scores(token_sets, labels, np.array(["X", "Y"]), np.arange(10), np.array([10, 11]), seed=42)
        self.assertEqual(scores.shape[0], 10)
        self.assertEqual(applied.shape[0], 2)
        self.assertGreater(vocabulary_size, 0)
        self.assertEqual(len(names), scores.shape[1])

    def test_runner_does_not_read_test_or_use_fixed_biological_rules(self):
        runner = (Path(__file__).parent / "run_intragenic_architecture_eb.py").read_text(encoding="utf-8")
        self.assertNotIn("test.csv", runner)
        self.assertNotIn("pd.concat([train", runner)
        self.assertNotIn("LR_EXACT", runner)
        self.assertNotIn("CONTRAST_PAIRS", runner)


if __name__ == "__main__":
    unittest.main()
