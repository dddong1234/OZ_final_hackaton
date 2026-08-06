from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))

from run_exact_evidence_confidence_screen import (  # noqa: E402
    ExactEvidenceState,
    apply_confidence_features,
)


class ExactEvidenceConfidenceTest(unittest.TestCase):
    def test_contribution_shape_features_are_finite_and_generic(self) -> None:
        matrix = sparse.csr_matrix(np.asarray([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=np.float32))
        state = ExactEvidenceState(
            selected=np.asarray([0, 1]),
            weights=np.asarray([[2.0, -1.0], [-0.5, 0.5]], dtype=np.float32),
            reliability=np.asarray([0.5, 1.0], dtype=np.float32),
        )
        result = apply_confidence_features(matrix, state, class_count=2)
        self.assertEqual(result.shape, (4, 18))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[0, 0], 0.0)  # class 0 positive sum
        self.assertLess(result[0, 1], 0.0)     # class 0 negative sum
        self.assertEqual(float(result[3].sum()), 0.0)  # no event is all-zero


if __name__ == "__main__":
    unittest.main()
