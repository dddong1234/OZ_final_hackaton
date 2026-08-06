import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tabpfn_dense_core import FitOnlyStandardizer, normalise_event_cell


class TabPFNDenseCoreTest(unittest.TestCase):
    def test_standardizer_uses_fit_rows_only(self):
        fit = np.asarray([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
        valid = np.asarray([[101.0, 105.0]], dtype=np.float32)
        scaler = FitOnlyStandardizer().fit(fit)
        transformed_fit = scaler.transform(fit)
        transformed_valid = scaler.transform(valid)
        np.testing.assert_allclose(transformed_fit.mean(axis=0), 0.0, atol=1e-6)
        self.assertTrue(np.all(transformed_valid > 90.0))
        self.assertEqual(transformed_valid.dtype, np.float32)

    def test_nan_wt_and_blank_never_become_events(self):
        self.assertEqual(normalise_event_cell(np.nan), ())
        self.assertEqual(normalise_event_cell("WT"), ())
        self.assertEqual(normalise_event_cell("  "), ())
        self.assertEqual(normalise_event_cell("R132H; R132H"), ("R132H",))


if __name__ == "__main__":
    unittest.main()
