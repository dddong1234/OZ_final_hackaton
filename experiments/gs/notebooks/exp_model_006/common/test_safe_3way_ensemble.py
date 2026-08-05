import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from safe_3way_ensemble import fixed_three_way_probability  # noqa: E402


class SafeThreeWayTest(unittest.TestCase):
    def test_fixed_weights_preserve_probability_simplex(self):
        multi = np.array([[.7, .3], [.4, .6]])
        ovr = np.array([[.5, .5], [.8, .2]])
        lgbm = np.array([[.9, .1], [.1, .9]])
        blended = fixed_three_way_probability(multi, ovr, lgbm)
        np.testing.assert_allclose(blended.sum(axis=1), 1.0)
        np.testing.assert_allclose(blended[0], [.67, .33])

    def test_production_source_has_no_test_or_fixed_event_rules(self):
        source = "\n".join(
            (Path(__file__).parent / name).read_text()
            for name in ("safe_3way_ensemble.py", "run_safe_3way_ensemble.py")
            if (Path(__file__).parent / name).exists()
        )
        for forbidden in ("test.csv", "pd.concat", "FINAL_EXACT", "CONTRAST_PAIRS"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
