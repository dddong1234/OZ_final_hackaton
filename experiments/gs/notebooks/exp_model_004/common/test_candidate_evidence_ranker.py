import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from candidate_evidence_ranker import (
    build_ranker_audit,
    candidate_matrix,
    candidate_scores_to_probability,
    topk_metrics,
)


class CandidateEvidenceRankerTest(unittest.TestCase):
    def test_candidate_matrix_has_one_row_per_patient_candidate(self):
        p_non_eb = np.array([[0.7, 0.3], [0.1, 0.9]])
        p_eb = np.array([[0.6, 0.4], [0.2, 0.8]])
        evidence = np.array([[1.0, -0.5], [0.2, 0.7]])
        features, patient_index, candidate_index = candidate_matrix(
            p_non_eb, p_eb, evidence, np.array([2.0, 4.0]), class_count=2
        )
        self.assertEqual(features.shape[0], 4)
        self.assertEqual(patient_index.tolist(), [0, 0, 1, 1])
        self.assertEqual(candidate_index.tolist(), [0, 1, 0, 1])
        self.assertTrue(np.isfinite(features).all())

    def test_candidate_scores_are_softmax_normalized_per_patient(self):
        probability = candidate_scores_to_probability(
            np.array([2.0, 0.0, 1.0, 1.0]), n_samples=2, n_classes=2
        )
        self.assertTrue(np.allclose(probability.sum(axis=1), 1.0))
        self.assertEqual(probability.shape, (2, 2))
        self.assertGreater(probability[0, 0], probability[0, 1])

    def test_ranker_training_rows_are_inner_oof_only(self):
        audit = build_ranker_audit(
            outer_train=np.array([0, 1, 2, 3]),
            inner_prediction_rows=np.array([0, 1, 2, 3]),
            outer_valid=np.array([4, 5]),
        )
        self.assertTrue(audit["ranker_training_rows_are_inner_oof"])
        self.assertFalse(audit["outer_validation_used_for_ranker_fit"])

    def test_topk_metrics_recognizes_true_class_in_second_candidate(self):
        classes = np.array(["A", "B", "C"])
        probability = np.array([[0.7, 0.2, 0.1], [0.4, 0.5, 0.1]])
        metrics = topk_metrics(np.array(["B", "A"]), probability, classes)
        self.assertEqual(metrics["top1_recall"], 0.0)
        self.assertEqual(metrics["top2_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
