import json
from pathlib import Path
import subprocess
import sys
import unittest

import nbformat


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
RUNNER = HERE / "run_h0_prototype_similarity_screen.py"
NOTEBOOK = HERE.parent / "exp" / "exp-h0-prototype-similarity-01.ipynb"


class PrototypeSimilarityRunnerContractTest(unittest.TestCase):
    def test_runner_has_no_test_file_reference_or_team_import(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("test.csv", source)
        self.assertNotIn("experiments/SDH", source)
        self.assertNotIn("experiments/iljun", source)

    def test_smoke_contract_reports_train_only_and_nan_zero(self):
        output = subprocess.check_output([sys.executable, str(RUNNER), "--smoke"], text=True)
        payload = json.loads(output.strip().splitlines()[-1])
        self.assertFalse(payload["test_read"])
        self.assertTrue(payload["leakage_check"])
        self.assertEqual(payload["nan_as_mutation_count"], 0)
        self.assertEqual(payload["prototype_probability_row_sum"], 1.0)

    def test_notebook_uses_runner_tqdm_and_explicit_execution_switch(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        self.assertIn("run_h0_prototype_similarity_screen.py", source)
        self.assertIn("tqdm", source)
        self.assertIn("RUN_EXPERIMENT", source)
