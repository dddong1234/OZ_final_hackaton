from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_faithful_h0_candidate_ranker import result_directory, run_contract, summary_columns


class FaithfulH0CandidateRankerRunnerTest(unittest.TestCase):
    def test_result_directory_is_the_experiment_result_directory(self) -> None:
        path = result_directory()
        self.assertEqual(path.name, "result")
        self.assertEqual(path.parent.name, "exp_model_008")

    def test_contract_explicitly_excludes_test_and_outer_validation_fit(self) -> None:
        contract = run_contract()
        self.assertFalse(contract["test_read"])
        self.assertFalse(contract["outer_validation_used_for_eb_fit"])
        self.assertFalse(contract["outer_validation_used_for_ranker_fit"])
        self.assertFalse(contract["fixed_class_gene_exact_mutation_rules"])

    def test_summary_schema_has_safety_fields(self) -> None:
        columns = set(summary_columns())
        self.assertTrue({"oof_macro_f1", "leakage_check", "nan_as_mutation_count", "delta_vs_h0"}.issubset(columns))


if __name__ == "__main__":
    unittest.main()
