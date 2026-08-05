import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from h0_selective_eb_replacement_runner import result_directory, split_checkpoint_oof, summary_columns  # noqa: E402


class H0SelectiveEBReplacementRunnerTest(unittest.TestCase):
    def test_result_directory_is_experiment_result_not_common_result(self):
        result = result_directory(Path(__file__))
        self.assertEqual(result.name, "result")
        self.assertEqual(result.parent.name, "exp_model_007")

    def test_summary_schema_contains_contract_audits(self):
        required = {
            "seed", "variant", "oof_macro_f1", "delta_vs_h0",
            "convergence_warning_count", "leakage_check", "nan_as_mutation_count",
        }
        self.assertTrue(required.issubset(set(summary_columns())))

    def test_checkpoint_gate_usage_is_removed_from_probability_mapping(self):
        payload = {"h0": [[0.5, 0.5]], "candidate": [[0.4, 0.6]], "gate_usage": [True]}
        probabilities, gate_usage = split_checkpoint_oof(payload)
        self.assertEqual(set(probabilities), {"h0", "candidate"})
        self.assertTrue(bool(gate_usage[0]))


if __name__ == "__main__":
    unittest.main()
