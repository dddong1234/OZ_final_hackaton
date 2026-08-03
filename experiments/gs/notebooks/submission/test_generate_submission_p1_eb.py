from pathlib import Path
import unittest


class P1EmpiricalBayesSubmissionContractTest(unittest.TestCase):
    def test_submission_runner_has_fixed_safe_contract(self):
        text = (Path(__file__).parent / "generate_submission_p1_empirical_bayes.py").read_text(encoding="utf-8")
        self.assertIn("TRAIN_SEEDS = (42, 777, 2024)", text)
        self.assertIn("empirical_bayes=True", text)
        self.assertIn("nan_as_mutation_count", text)
        self.assertIn("sample_submission.csv", text)
        self.assertIn("test_read_after_fit_only", text)


if __name__ == "__main__":
    unittest.main()
