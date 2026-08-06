from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generate_submission_h0_exact_event_eb_3seed import (  # noqa: E402
    VALIDATED_SEEDS, average_seed_probabilities, run_seed_bagged, smoke,
)


class ExactEventSubmissionTest(unittest.TestCase):
    def test_equal_seed_average_preserves_probability_mass(self) -> None:
        probability = average_seed_probabilities([
            np.asarray([[0.7, 0.3]], dtype=np.float32),
            np.asarray([[0.4, 0.6]], dtype=np.float32),
            np.asarray([[0.5, 0.5]], dtype=np.float32),
        ])
        np.testing.assert_allclose(probability, [[0.53333336, 0.46666667]], rtol=1e-5)

    def test_rejects_any_seed_contract_other_than_validated_three_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            run_seed_bagged(seeds=(42,))

    def test_smoke_is_train_only(self) -> None:
        audit = smoke()
        self.assertEqual(audit["test_role"], "not_read")
        self.assertEqual(audit["nan_as_mutation_count"], 0)
        self.assertEqual(audit["seed_contract"], list(VALIDATED_SEEDS))
