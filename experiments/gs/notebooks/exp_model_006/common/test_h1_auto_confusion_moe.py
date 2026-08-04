import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from h1_auto_confusion_moe import (  # noqa: E402
    discover_confusion_groups,
    redistribute_group_probability_mass,
)
from run_h1_auto_confusion_moe import decision  # noqa: E402


class H1AutoConfusionMoETest(unittest.TestCase):
    def test_groups_are_discovered_without_fixed_class_names(self):
        labels = np.array(["A", "B", "C", "D"])
        truth = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
        predicted = np.array(["B", "B", "A", "A", "C", "C", "D", "D"])
        groups = discover_confusion_groups(truth, predicted, labels, n_groups=3)
        self.assertEqual(len(groups), 3)
        self.assertTrue(any(set(group) == {"A", "B"} for group in groups))
        self.assertEqual(set().union(*(set(group) for group in groups)), set(labels))

    def test_group_specialist_preserves_base_probability_mass(self):
        classes = np.array(["A", "B", "C"])
        base = np.array([[0.20, 0.30, 0.50], [0.70, 0.10, 0.20]])
        specialist = np.array([[0.90, 0.10], [0.25, 0.75]])
        output = redistribute_group_probability_mass(base, specialist, classes, ("A", "B"))
        np.testing.assert_allclose(output.sum(axis=1), 1.0)
        np.testing.assert_allclose(output[:, :2].sum(axis=1), base[:, :2].sum(axis=1))
        np.testing.assert_allclose(output[:, 2], base[:, 2])

    def test_production_source_does_not_read_or_join_test(self):
        source = (Path(__file__).parent / "h1_auto_confusion_moe.py").read_text()
        source += (Path(__file__).parent / "run_h1_auto_confusion_moe.py").read_text()
        self.assertNotIn("test.csv", source)
        self.assertNotIn("pd.concat", source)
        self.assertNotIn("FINAL_EXACT", source)

    def test_decision_requires_four_positive_folds_for_strong_candidate(self):
        verdict = decision(.544744, .561000, [.004, .004, -.001, .004, .004], .0)
        self.assertEqual(verdict["decision"], "strong_validation_candidate")


if __name__ == "__main__":
    unittest.main()
