import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_h0_prediction_stability_audit import audit_frame


class PredictionStabilityAuditTest(unittest.TestCase):
    def test_tracks_recovered_and_broken_predictions(self):
        labels = ["A", "B"]
        frame = pd.DataFrame({"row_index": [0, 1, 2], "truth": ["A", "B", "A"]})
        values = {
            "seed_42": [[.2, .8], [.1, .9], [.9, .1]],
            "seed_777": [[.8, .2], [.9, .1], [.8, .2]],
            "seed_2024": [[.8, .2], [.9, .1], [.8, .2]],
            "fold_aligned_bagged": [[.6, .4], [.367, .633], [.833, .167]],
        }
        for variant, rows in values.items():
            for index, label in enumerate(labels):
                frame[f"{variant}__{label}"] = np.asarray(rows)[:, index]
        summary, _, rows, _, audit = audit_frame(frame)
        self.assertEqual(len(summary), 4)
        self.assertEqual(int((rows.bagging_transition == "recovered").sum()), 1)
        self.assertEqual(int((rows.bagging_transition == "broken").sum()), 0)
        self.assertTrue(audit["leakage_check"])


if __name__ == "__main__":
    unittest.main()
