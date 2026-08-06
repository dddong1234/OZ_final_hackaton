from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generate_submission_equal_h0_non_eb_3seed import equal_probability, run_seed_bagged, smoke


class EqualProbabilityTest(unittest.TestCase):
    def test_uses_fixed_half_weights_and_normalizes_rows(self) -> None:
        left = np.array([[0.8, 0.2]], dtype=np.float32)
        right = np.array([[0.2, 0.8]], dtype=np.float32)

        actual = equal_probability(left, right)

        np.testing.assert_allclose(actual, [[0.5, 0.5]])

    def test_rejects_any_seed_contract_other_than_validated_three_seed_tuple(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            run_seed_bagged(seeds=(42,))

    def test_smoke_reports_train_only_contract_without_generating_submission(self) -> None:
        audit = smoke()

        self.assertEqual(audit["test_role"], "not_read")
        self.assertEqual(audit["nan_as_mutation_count"], 0)
        self.assertEqual(audit["seed_contract"], [42, 777, 2024])
