import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from faithful_h0_ranker import apply_correction, make_pairwise_rows  # noqa: E402


class FaithfulH0RankerTest(unittest.TestCase):
    def test_zero_strength_keeps_h0_probability_exactly(self):
        probability = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float32)
        residual = np.asarray([[1.0, -2.0, 3.0], [0.4, 0.3, -0.1]], dtype=np.float32)
        corrected = apply_correction(probability, residual, 0.0)
        np.testing.assert_allclose(corrected, probability, rtol=0, atol=1e-7)

    def test_pairwise_rows_are_symmetric_for_all_non_truth_classes(self):
        features = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        rows, target = make_pairwise_rows(features, np.asarray(["A", "B"]), np.asarray(["A", "B", "C"]))
        self.assertEqual(rows.shape, (8, 2))
        self.assertEqual(target.tolist(), [1, 0, 1, 0, 1, 0, 1, 0])
        np.testing.assert_allclose(rows[0], -rows[1])

    def test_runner_source_has_no_test_read_or_fixed_biological_rules(self):
        source = (Path(__file__).parent / "run_faithful_h0_allclass_ranker.py").read_text(encoding="utf-8")
        self.assertNotIn("test.csv", source)
        self.assertNotIn("LR_EXACT", source)
        self.assertNotIn("CONTRAST_PAIRS", source)


if __name__ == "__main__":
    unittest.main()
