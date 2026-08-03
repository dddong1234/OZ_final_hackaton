import json
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "exp"
FILES = {"exp-parser-grammar-audit-01.ipynb", "exp-parser-recovery-g1-01.ipynb", "exp-parser-recovery-g2-01.ipynb", "exp-p1-eb-vulnerability-audit-01.ipynb", "exp-selective-eb-gate-01.ipynb", "exp-all-class-evidence-ranker-01.ipynb"}
class NotebookContract(unittest.TestCase):
    def test_notebooks_are_safe_json(self):
        self.assertTrue(FILES.issubset({p.name for p in EXP.glob("*.ipynb")}))
        for name in FILES:
            text=(EXP/name).read_text(); self.assertNotIn("test.csv",text)
            data=json.loads(text)
            for i,c in enumerate(data["cells"]):
                if c["cell_type"]=="code": compile("".join(c["source"]),f"{name}:{i}","exec")
if __name__ == "__main__": unittest.main()
