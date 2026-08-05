import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "exp" / "exp-class-conditional-evidence-set-network-01.ipynb"


class EvidenceNotebookContractTest(unittest.TestCase):
    def test_notebook_has_train_only_runner_and_compilable_code_cells(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("test.csv", text)
        data = json.loads(text)
        for index, cell in enumerate(data["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(index), "exec")


if __name__ == "__main__":
    unittest.main()
