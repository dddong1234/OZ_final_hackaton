"""Empirical-Bayes 3-seed 확정 검증 집계 회귀 테스트."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eb_confirmation import summarize_three_seed


class EmpiricalBayesConfirmationTest(unittest.TestCase):
    def test_uses_paired_seed_delta_and_keeps_safety_contract(self):
        rows = pd.DataFrame(
            [
                {"seed": 42, "variant": "P1 multinomial LR", "oof_macro_f1": 0.52,
                 "feature_count": 8201.0, "convergence_warning_count": 0,
                 "leakage_check": True, "nan_as_mutation_count": 0},
                {"seed": 42, "variant": "eb", "oof_macro_f1": 0.53,
                 "feature_count": 8201.0, "convergence_warning_count": 0,
                 "leakage_check": True, "nan_as_mutation_count": 0},
                {"seed": 777, "variant": "P1 multinomial LR", "oof_macro_f1": 0.51,
                 "feature_count": 8201.0, "convergence_warning_count": 0,
                 "leakage_check": True, "nan_as_mutation_count": 0},
                {"seed": 777, "variant": "eb", "oof_macro_f1": 0.525,
                 "feature_count": 8201.0, "convergence_warning_count": 0,
                 "leakage_check": True, "nan_as_mutation_count": 0},
            ]
        )

        per_seed, aggregate = summarize_three_seed(rows)

        self.assertEqual(
            per_seed.groupby("seed")["paired_delta_vs_p1"].first().round(6).tolist(),
            [0.01, 0.015],
        )
        eb = aggregate.loc[aggregate.variant.eq("eb")].iloc[0]
        self.assertAlmostEqual(eb.oof_macro_f1_mean, 0.5275)
        self.assertAlmostEqual(eb.paired_delta_mean, 0.0125)
        self.assertTrue(bool(eb.leakage_check_all))
        self.assertEqual(int(eb.nan_as_mutation_count_max), 0)


if __name__ == "__main__":
    unittest.main()
