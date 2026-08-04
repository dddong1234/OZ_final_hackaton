import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent))

from eb_offset_residual import (
    fit_offset_residual,
    hashed_event_matrix,
    offset_audit,
    offset_probability,
)


class EbOffsetResidualTest(unittest.TestCase):
    def test_zero_residual_returns_offset_probability(self):
        offset = np.log(np.array([[0.7, 0.3], [0.2, 0.8]]))
        result = offset_probability(offset, np.zeros((3, 2)), np.zeros(2), csr_matrix((2, 3)))
        self.assertTrue(np.allclose(result, np.exp(offset)))

    def test_hash_is_reproducible_without_vocabulary_fit(self):
        tokens = [{"TP53__MISSENSE"}, {"BRAF__NONSENSE"}]
        left = hashed_event_matrix(tokens, np.array([0, 1]))
        right = hashed_event_matrix(tokens, np.array([0, 1]))
        self.assertEqual((left != right).nnz, 0)
        self.assertEqual(left.shape[1], 16384)

    def test_offset_training_audit_rejects_outer_validation_rows(self):
        audit = offset_audit(
            np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([3])
        )
        self.assertTrue(audit["offset_train_rows_are_inner_oof"])
        self.assertFalse(audit["outer_validation_used_for_residual_fit"])

    def test_training_produces_finite_epoch_losses(self):
        features = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]))
        labels = np.array([0, 1, 0, 1])
        offset = np.log(np.full((4, 2), 0.5))
        weight, bias, history = fit_offset_residual(
            features,
            labels,
            offset,
            np.ones(2),
            epochs=2,
            batch_size=2,
            seed=42,
        )
        self.assertEqual(weight.shape, (2, 2))
        self.assertEqual(bias.shape, (2,))
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(history).all())


if __name__ == "__main__":
    unittest.main()
