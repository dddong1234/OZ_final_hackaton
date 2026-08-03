import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from selective_eb_gate import SELECTIVE_MARGIN, selective_probability


class SelectiveEbGateTest(unittest.TestCase):
    def test_low_eb_margin_uses_p1_non_eb_probability(self):
        p1 = np.asarray([[.8, .2], [.3, .7]], dtype=float)
        eb = np.asarray([[.51, .49], [.1, .9]], dtype=float)
        result, selected = selective_probability(p1, eb)
        np.testing.assert_allclose(result[0], p1[0])
        np.testing.assert_allclose(result[1], eb[1])
        self.assertEqual(selected.tolist(), [True, False])

    def test_threshold_is_fixed_before_new_seed_validation(self):
        self.assertEqual(SELECTIVE_MARGIN, .05)


if __name__ == "__main__":
    unittest.main()
