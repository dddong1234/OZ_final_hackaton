import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from xgb_complement import fixed_blend, xgb_config  # noqa: E402


class XGBComplementTest(unittest.TestCase):
    def test_fixed_blend_uses_predeclared_weights_and_normalizes(self):
        h0 = np.asarray([[0.70, 0.30], [0.10, 0.90]])
        xgb = np.asarray([[0.20, 0.80], [0.60, 0.40]])
        actual = fixed_blend(h0, xgb)
        np.testing.assert_allclose(actual, .80 * h0 + .20 * xgb)
        np.testing.assert_allclose(actual.sum(axis=1), 1.0)

    def test_xgb_config_is_multiclass_hist_and_regularized(self):
        config = xgb_config(seed=42, class_count=26)
        self.assertEqual(config["objective"], "multi:softprob")
        self.assertEqual(config["num_class"], 26)
        self.assertEqual(config["tree_method"], "hist")
        self.assertGreater(config["reg_lambda"], 0)


if __name__ == "__main__":
    unittest.main()
