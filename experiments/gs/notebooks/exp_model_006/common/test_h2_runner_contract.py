import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


class H2RunnerContractTest(unittest.TestCase):
    def test_production_code_does_not_read_or_join_test(self):
        text = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.glob("h2_*.py"))
        self.assertNotIn("test.csv", text)
        self.assertNotIn("pd.concat", text)
        self.assertNotIn("FINAL_EXACT", text)
        self.assertNotIn("from experiments", text)

    def test_runner_uses_only_declared_strengths_and_shared_pairwise_model(self):
        text = (ROOT / "run_h2_evidence_shape_pairwise.py").read_text(encoding="utf-8")
        self.assertIn("ALPHAS = (.10, .20)", text)
        self.assertIn("PAIRWISE_C = .035", text)
        self.assertNotIn(".0175", text)
        self.assertIn("make_symmetric_pairs", text)


if __name__ == "__main__":
    unittest.main()
