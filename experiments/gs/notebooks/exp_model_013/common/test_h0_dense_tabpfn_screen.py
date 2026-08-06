import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_h0_dense_tabpfn_screen import SCREEN_DELTA, decide_screen, smoke_contract


class H0DenseTabPFNScreenTest(unittest.TestCase):
    def test_decision_requires_score_and_fold_direction(self):
        accepted = decide_screen(delta=SCREEN_DELTA, positive_folds=4, h0_reference_match=True)
        self.assertEqual(accepted, "screen_candidate")
        self.assertEqual(decide_screen(delta=0.0149, positive_folds=5, h0_reference_match=True), "not_detected")
        self.assertEqual(decide_screen(delta=0.020, positive_folds=3, h0_reference_match=True), "not_detected")
        self.assertEqual(decide_screen(delta=0.020, positive_folds=5, h0_reference_match=False), "baseline_not_reproduced")

    def test_smoke_contract_never_reads_test_or_creates_nan_events(self):
        audit = smoke_contract()
        self.assertFalse(audit["test_read"])
        self.assertFalse(audit["train_test_concat"])
        self.assertTrue(audit["leakage_check"])
        self.assertEqual(audit["nan_as_mutation_count"], 0)
        self.assertTrue(np.isfinite(audit["example_dense_feature_count"]))


if __name__ == "__main__":
    unittest.main()
