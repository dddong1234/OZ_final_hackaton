"""exp_model_003 노트북의 정적 실행 계약 검사."""
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"
EXPECTED = {
    "exp-eb-topk-parser-audit-01.ipynb",
    "exp-point-process-eb-01.ipynb",
    "exp-multivariate-eb-01.ipynb",
    "exp-pretrained-mutation-encoder-01.ipynb",
    "exp-macro-f1-decoder-01.ipynb",
    "exp-intragenic-architecture-01.ipynb",
    "exp-4state-dependency-01.ipynb",
}


class NotebookContractTest(unittest.TestCase):
    def test_all_planned_notebooks_exist_and_are_valid_json(self):
        self.assertTrue(EXPECTED.issubset({path.name for path in EXP.glob("*.ipynb")}))
        for name in EXPECTED:
            payload = json.loads((EXP / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            self.assertTrue(payload["cells"])

    def test_notebooks_do_not_read_test_csv(self):
        for name in EXPECTED:
            text = (EXP / name).read_text(encoding="utf-8")
            self.assertNotIn("test.csv", text)

    def test_code_cells_compile(self):
        for name in EXPECTED:
            payload = json.loads((EXP / name).read_text(encoding="utf-8"))
            for number, cell in enumerate(payload["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{name}:cell{number}", "exec")

    def test_result_file_names_interpolate_seed(self):
        for name in ("exp-eb-topk-parser-audit-01.ipynb", "exp-point-process-eb-01.ipynb", "exp-multivariate-eb-01.ipynb"):
            payload = json.loads((EXP / name).read_text(encoding="utf-8"))
            source = "\n".join("".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code")
            self.assertIn('RESULT / f"', source)
            self.assertIn("seed{SEED}", source)


if __name__ == "__main__":
    unittest.main()
