from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from faithful_h0_ranker_core import apply_residual, build_evidence_shape, make_symmetric_pairs


class FaithfulH0RankerCoreTest(unittest.TestCase):
    def test_evidence_shape_has_finite_candidate_features(self) -> None:
        matrix = sparse.csr_matrix([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        weights = np.asarray([[1.0, -0.5, 0.2], [-1.0, 0.6, -0.1]], dtype=np.float32)
        priors = np.asarray([0.6, 0.4], dtype=np.float32)
        probability = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
        output = build_evidence_shape(matrix, weights, priors, probability)
        self.assertEqual(output.shape, (2, 2, 19))
        self.assertTrue(np.isfinite(output).all())
        self.assertGreater(output[0, 0, 0], 0.0)
        self.assertLess(output[0, 1, 0], 0.0)

    def test_pair_rows_are_directionally_symmetric(self) -> None:
        features = np.asarray([[[2.0], [1.0]], [[1.0], [3.0]]], dtype=np.float32)
        x, y = make_symmetric_pairs(features, np.asarray(["A", "B"], dtype=object), np.asarray(["A", "B"], dtype=object))
        self.assertEqual(x.shape, (4, 1))
        self.assertEqual(y.tolist(), [1, 0, 1, 0])
        np.testing.assert_allclose(x[0], -x[1])
        np.testing.assert_allclose(x[2], -x[3])

    def test_log_probability_residual_is_normalized(self) -> None:
        probability = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
        residual = np.asarray([[1.0, -1.0], [-2.0, 2.0]], dtype=np.float32)
        output = apply_residual(probability, residual, 0.20)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
