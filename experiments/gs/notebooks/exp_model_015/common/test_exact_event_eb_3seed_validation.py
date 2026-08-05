import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_exact_event_eb_3seed_validation import aggregate


class ExactEventEBThreeSeedTest(unittest.TestCase):
    def test_requires_all_three_positive_seed_deltas(self):
        summary = pd.DataFrame([
            {"seed": seed, "variant": variant, "oof_macro_f1": score, "oof_accuracy": score, "feature_count_mean": 1, "convergence_warning_count": 0, "leakage_check": True, "nan_as_mutation_count": 0}
            for seed, base, candidate in ((42, .50, .52), (777, .50, .51), (2024, .50, .49))
            for variant, score in (("H0_selective_EB", base), ("exact_event_EB", candidate))
        ])
        folds = pd.DataFrame([
            {"seed": seed, "fold": fold, "variant": variant, "macro_f1": score}
            for seed, base in ((42, .50), (777, .50), (2024, .50))
            for fold in range(1, 6)
            for variant, score in (("H0_selective_EB", base), ("exact_event_EB", base + .01))
        ])
        classes = pd.DataFrame([
            {"seed": seed, "class": label, "variant": variant, "f1": .5 + (variant == "exact_event_EB") * .01}
            for seed in (42, 777, 2024) for label in ("A", "B") for variant in ("H0_selective_EB", "exact_event_EB")
        ])
        _, decision = aggregate(summary, folds, classes)
        self.assertFalse(decision["accepted_3seed"])
        self.assertFalse(decision["all_seed_delta_positive"])


if __name__ == "__main__":
    unittest.main()
