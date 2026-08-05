import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from baseline_artifact import reference_only_baseline, validate_baseline_oof  # noqa: E402


class BaselineArtifactTest(unittest.TestCase):
    def test_reference_only_baseline_does_not_require_probabilities(self):
        baseline = reference_only_baseline()
        self.assertIsNone(baseline.probabilities)
        self.assertEqual(baseline.reference_macro_f1, 0.54202)
        self.assertEqual(baseline.comparison_mode, "unpaired_reference")

    def test_artifact_requires_exact_class_column_order(self):
        frame = pd.DataFrame({"true_class": ["A", "B"], "prob__B": [.1, .9], "prob__A": [.9, .1]})
        with self.assertRaises(ValueError):
            validate_baseline_oof(frame, np.asarray(["A", "B"], dtype=object), 2)

    def test_runner_does_not_rebuild_team_baseline(self):
        source = (Path(__file__).parent / "run_evidence_set_network.py").read_text(encoding="utf-8")
        self.assertIn("--baseline-oof", source)
        self.assertNotIn("run_team_baseline_oof(", source)


if __name__ == "__main__":
    unittest.main()
