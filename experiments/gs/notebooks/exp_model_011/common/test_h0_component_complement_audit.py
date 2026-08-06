import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_h0_component_complement_audit import equal_blend, recovery_breakage  # noqa: E402


class ComponentComplementAuditTest(unittest.TestCase):
    def test_equal_blend_normalizes_each_probability_row(self):
        left = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32)
        right = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        blended = equal_blend(left, right)
        np.testing.assert_allclose(blended.sum(axis=1), np.ones(2))
        np.testing.assert_allclose(blended[0], np.array([0.4, 0.6]))

    def test_recovery_breakage_counts_compare_to_h0(self):
        truth = np.array(["A", "B", "A"])
        h0 = np.array(["B", "B", "A"])
        candidate = np.array(["A", "A", "B"])
        recovered, broken = recovery_breakage(truth, h0, candidate)
        self.assertEqual((recovered, broken), (1, 2))


if __name__ == "__main__":
    unittest.main()
