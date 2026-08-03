import json, unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"exp/exp-frozen-biomedical-encoder-01.ipynb"
class Contract(unittest.TestCase):
 def test_notebook_contract(self):
  text=P.read_text(); self.assertNotIn("test.csv",text); data=json.loads(text)
  for i,c in enumerate(data["cells"]):
   if c["cell_type"]=="code": compile("".join(c["source"]),str(i),"exec")
if __name__=="__main__": unittest.main()
