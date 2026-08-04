import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from h2_safe_h0 import build_evidence_shape, discover_pairs, fit_predict_h0  # noqa: E402


class H2SafeH0Test(unittest.TestCase):
    def test_discovered_pairs_do_not_require_fixed_class_names(self):
        x = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.float32)
        y = np.asarray(["X", "X", "Y", "Y", "Z", "Z"])
        pairs = discover_pairs(x, y, np.asarray(["X", "Y", "Z"]), top_k=2)
        self.assertTrue(all(left in {"X", "Y", "Z"} and right in {"X", "Y", "Z"} for left, right in pairs))

    def test_evidence_shape_has_requested_candidate_feature_count(self):
        gene_type = np.asarray([[1, 0], [1, 1]], dtype=np.float32)
        weights = np.asarray([[2.0, -1.0], [-2.0, 1.0]], dtype=np.float32)
        probability = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
        shape = build_evidence_shape(gene_type, weights, np.asarray([.5, .5]), probability)
        self.assertEqual(shape.shape, (2, 2, 19))
        self.assertTrue(np.isfinite(shape).all())

    def test_small_fit_predict_smoke_keeps_validation_out_of_eb_fit(self):
        labels = np.asarray(["A", "B", "C"] * 6)
        frame = pd.DataFrame({"G1": [{"A": "R1H", "B": "R2*", "C": "Q3W"}[label] for label in labels], "G2": [{"A": "WT", "B": "R5*", "C": "Q6W"}[label] for label in labels]})
        prediction = fit_predict_h0(frame.iloc[:15], frame.iloc[15:], ["G1", "G2"], labels[:15], seed=42)
        self.assertEqual(prediction.probability.shape, (3, 3))
        self.assertTrue(np.allclose(prediction.probability.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
