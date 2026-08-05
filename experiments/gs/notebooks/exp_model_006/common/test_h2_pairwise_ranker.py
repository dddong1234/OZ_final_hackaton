import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from h2_pairwise_ranker import apply_residual_correction, make_symmetric_pairs  # noqa: E402


class H2PairwiseRankerTest(unittest.TestCase):
    def test_pair_rows_are_symmetric(self):
        features = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
        x, y = make_symmetric_pairs(features, np.asarray([0, 1, 0]), np.asarray([0, 1]))
        self.assertEqual(x.shape, (6, 2))
        self.assertTrue(np.array_equal(y, np.asarray([1, 0, 1, 0, 1, 0])))
        self.assertTrue(np.allclose(x[0], -x[1]))

    def test_residual_correction_preserves_probability_rows(self):
        probability = np.asarray([[0.6, 0.4]], dtype=np.float64)
        corrected = apply_residual_correction(probability, np.asarray([[2.0, -2.0]]), 0.2)
        self.assertTrue(np.allclose(corrected.sum(axis=1), 1.0))
        self.assertGreater(corrected[0, 0], probability[0, 0])


if __name__ == "__main__":
    unittest.main()
