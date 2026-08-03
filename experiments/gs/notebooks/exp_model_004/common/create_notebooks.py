from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; EXP = ROOT / "exp"

def cell(kind, source): return {"cell_type": kind, "metadata": {}, "source": [line + "\n" for line in source.strip().splitlines()], **({"execution_count": None, "outputs": []} if kind == "code" else {})}
def write(name, cells): (EXP / name).write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1), encoding="utf-8")
SETUP = '''from pathlib import Path
import subprocess, sys, json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "experiments/gs/notebooks/exp_model_004/common/run_parser_recovery.py").exists())
RUNNER = ROOT / "experiments/gs/notebooks/exp_model_004/common/run_parser_recovery.py"
RESULT = ROOT / "experiments/gs/notebooks/exp_model_004/result"'''
def runner(axis, runid, default=True): return f'''SEED=42; RUN_EXPERIMENT={default}; RUN_ID="{runid}"
if RUN_EXPERIMENT:
    process=subprocess.Popen([sys.executable,str(RUNNER),"--axis","{axis}","--seed",str(SEED),"--run-id",RUN_ID],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    tail=[]
    for line in tqdm(process.stdout,desc="parser runner",unit="line"):
        print(line,end=""); tail=(tail+[line])[-120:]
    if process.wait(): raise RuntimeError("runner failed:\\n"+"".join(tail))
else: print("RUN_EXPERIMENT=False")'''
def main():
    EXP.mkdir(exist_ok=True)
    write("exp-parser-grammar-audit-01.ipynb", [cell("markdown", "# Parser recovery: 원문 문법 감사\n\ntrain 원문만 읽어 segment 보존·UNKNOWN·canonical type을 확인합니다. test는 읽지 않습니다."), cell("code", SETUP), cell("code", runner("audit", "exp-parser-grammar-audit-01")), cell("code", '''types=pd.read_csv(RESULT/"exp-parser-grammar-audit-01_canonical_types.csv")
unknown=pd.read_csv(RESULT/"exp-parser-grammar-audit-01_unknown_patterns.csv")
contract=json.loads((RESULT/"exp-parser-grammar-audit-01_contract.json").read_text())
display(types); display(unknown.head(30)); display(contract)
assert contract["segment_conservation"] and contract["nan_cell_count"] >= 0
types.set_index("canonical_type")["count"].plot.bar(figsize=(9,4),title="Canonical event coverage"); plt.tight_layout(); plt.show()''')])
    write("exp-parser-recovery-g1-01.ipynb", [cell("markdown", "# G1: parser recovery only\n\n복수 이벤트 분리·문법 복원만 반영하고 canonical type은 기존 parent type으로 매핑합니다."), cell("code", SETUP), cell("code", runner("g1", "exp-parser-recovery-g1-01")), cell("code", '''summary=pd.read_csv(RESULT/f"exp-parser-recovery-g1-01_seed{SEED}_summary.csv")
folds=pd.read_csv(RESULT/f"exp-parser-recovery-g1-01_seed{SEED}_fold_metrics.csv")
display(summary); display(folds)
assert summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()
summary.set_index("variant")["oof_macro_f1"].plot.bar(figsize=(7,4),title="G1 parser recovery"); plt.tight_layout(); plt.show()''')])
    write("exp-parser-recovery-g2-01.ipynb", [cell("markdown", "# G2: canonical event-type EB\n\nG1 screen 통과 후에만 실행합니다. canonical taxonomy가 만드는 추가 효과를 확인합니다."), cell("code", SETUP), cell("code", runner("g2", "exp-parser-recovery-g2-01", "False")), cell("code", '''summary=pd.read_csv(RESULT/f"exp-parser-recovery-g2-01_seed{SEED}_summary.csv")
folds=pd.read_csv(RESULT/f"exp-parser-recovery-g2-01_seed{SEED}_fold_metrics.csv")
display(summary); display(folds)
assert summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()''')])
if __name__ == "__main__": main()
