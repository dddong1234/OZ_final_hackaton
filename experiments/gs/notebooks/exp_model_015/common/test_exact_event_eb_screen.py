import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
from run_exact_event_eb_screen import apply_exact_eb, fit_exact_eb


class ExactEventEBTest(unittest.TestCase):
    def test_uses_observed_tokens_without_support_threshold(self):
        matrix = sparse.csr_matrix(np.asarray([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float32))
        labels = np.asarray(["A", "B", "C", "A", "B", "C"])
        state = fit_exact_eb(matrix, labels, np.asarray(["A", "B", "C"]))
        self.assertEqual(len(state.selected), 3)
        self.assertEqual(apply_exact_eb(matrix, state, 3).shape, (6, 3))


if __name__ == "__main__":
    unittest.main()
