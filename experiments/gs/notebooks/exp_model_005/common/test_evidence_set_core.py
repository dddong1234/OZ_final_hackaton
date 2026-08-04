import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from evidence_set_core import (  # noqa: E402
    EvidenceSetNetwork,
    build_event_evidence,
    listwise_loss,
    nested_audit,
)


class EvidenceSetCoreTest(unittest.TestCase):
    def test_event_evidence_keeps_positive_and_negative_contributions(self):
        evidence = build_event_evidence(
            [("TP53", "MISSENSE", "R175H")],
            {"TP53__MISSENSE": np.asarray([1.5, -0.5], dtype=np.float32)},
            {"TP53__MISSENSE": 20},
            np.asarray([1.0]),
            class_count=2,
        )
        self.assertEqual(evidence.shape, (2, 1, 16))
        self.assertGreater(evidence[0, 0, 0], 0)
        self.assertLess(evidence[1, 0, 0], 0)

    def test_listwise_network_returns_one_score_per_class(self):
        import torch

        features = torch.zeros((2, 26, 3, 16), dtype=torch.float32)
        mask = torch.ones((2, 26, 3), dtype=torch.bool)
        logits = EvidenceSetNetwork(input_dim=16)(features, mask)
        self.assertEqual(tuple(logits.shape), (2, 26))
        self.assertTrue(torch.isfinite(listwise_loss(logits, torch.tensor([0, 1]), torch.ones(26))).item())

    def test_nested_audit_rejects_validation_rows_in_eb_fit(self):
        audit = nested_audit(
            outer_train=np.asarray([0, 1]),
            inner_oof_rows=np.asarray([0, 1]),
            outer_validation=np.asarray([2]),
        )
        self.assertTrue(audit["ranker_training_rows_are_inner_oof"])
        self.assertFalse(audit["outer_validation_used_for_eb_fit"])


if __name__ == "__main__":
    unittest.main()
