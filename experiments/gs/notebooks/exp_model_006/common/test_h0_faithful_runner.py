import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_h0_faithful_reproduction import baseline_gate, has_required_columns  # noqa: E402


class H0FaithfulRunnerTest(unittest.TestCase):
    def test_production_code_never_reads_or_joins_test(self):
        root = Path(__file__).parent
        text = (root / "h0_faithful_pipeline.py").read_text() + (root / "run_h0_faithful_reproduction.py").read_text()
        self.assertNotIn("test.csv", text)
        self.assertNotIn("pd.concat", text)
        self.assertNotIn("FINAL_EXACT", text)
        self.assertNotIn("from experiments", text)

    def test_baseline_gate_blocks_downstream_when_reference_is_not_reproduced(self):
        audit = baseline_gate(0.523717)
        self.assertFalse(audit["baseline_reproduced"])
        self.assertTrue(audit["block_downstream_experiments"])

    def test_baseline_gate_accepts_reference_tolerance(self):
        audit = baseline_gate(0.5438)
        self.assertTrue(audit["baseline_reproduced"])
        self.assertFalse(audit["block_downstream_experiments"])

    def test_summary_schema_check_returns_boolean_without_iterating_over_boolean(self):
        summary = pd.DataFrame({
            "variant": ["H0_blend_80_20"],
            "oof_macro_f1": [0.543679],
            "oof_accuracy": [0.5],
            "feature_count_mean": [8200.0],
            "convergence_warning_count": [0],
            "leakage_check": [True],
            "nan_as_mutation_count": [0],
            "reference_oof_macro_f1": [0.543679],
            "reference_delta": [0.0],
        })
        self.assertTrue(has_required_columns("summary", summary))


if __name__ == "__main__":
    unittest.main()
