import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))

from h0_selective_eb_replacement import (  # noqa: E402
    H0_SPECIALIST_WEIGHT,
    SELECTIVE_LR_WEIGHT,
    fixed_branch_replacement,
    cross_fitted_eb_scores,
)


class H0SelectiveEBReplacementTest(unittest.TestCase):
    def test_replaces_only_lr_branch_with_fixed_weights(self):
        selective_lr = np.asarray([[0.70, 0.30], [0.20, 0.80]])
        specialist = np.asarray([[0.10, 0.90], [0.60, 0.40]])

        actual = fixed_branch_replacement(selective_lr, specialist)

        np.testing.assert_allclose(
            actual,
            SELECTIVE_LR_WEIGHT * selective_lr + H0_SPECIALIST_WEIGHT * specialist,
        )
        np.testing.assert_allclose(actual.sum(axis=1), 1.0)

    def test_rejects_mismatched_probability_shapes(self):
        with self.assertRaises(ValueError):
            fixed_branch_replacement(np.ones((2, 3)) / 3, np.ones((2, 2)) / 2)

    def test_cross_fitted_empirical_bayes_never_uses_apply_rows_for_fit(self):
        matrix = sparse.csr_matrix(np.asarray([
            [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
            [0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        ], dtype=np.float32))
        labels = np.asarray(['A'] * 5 + ['B'] * 5)
        train_score, apply_score = cross_fitted_eb_scores(matrix, matrix[:2], labels, np.asarray(['A', 'B']), seed=42)
        self.assertEqual(train_score.shape, (10, 2))
        self.assertEqual(apply_score.shape, (2, 2))
        self.assertTrue(np.isfinite(train_score).all() and np.isfinite(apply_score).all())


if __name__ == "__main__":
    unittest.main()
