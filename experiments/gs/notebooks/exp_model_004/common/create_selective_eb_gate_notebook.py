from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def cell(kind: str, source: str) -> dict:
    payload = {"cell_type": kind, "metadata": {}, "source": [line + "\n" for line in source.strip().splitlines()]}
    if kind == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


def main() -> None:
    cells = [
        cell("markdown", """# Selective EB gate — 새 seed 확정 검증

기존 취약구간 감사에서 고정한 규칙을 새 seed `31415 / 52 / 62`에만 적용한다.
P1+EB 확률의 Top-1/Top-2 margin이 `0.05` 미만이면 P1 non-EB 확률을, 나머지는 P1+EB 확률을 사용한다. threshold·피처·blend 비율은 탐색하지 않으며 평가 데이터는 읽지 않고 제출파일도 만들지 않는다."""),
        cell("code", """from pathlib import Path
import json, subprocess, sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
ROOT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'experiments/gs/notebooks/exp_model_004/common/run_selective_eb_gate.py').exists())
RUNNER=ROOT/'experiments/gs/notebooks/exp_model_004/common/run_selective_eb_gate.py'
RESULT=ROOT/'experiments/gs/notebooks/exp_model_004/result'
RUN_ID='exp-selective-eb-gate-01'
SEEDS=(31415,52,62)
RUN_EXPERIMENT=False"""),
        cell("code", """if RUN_EXPERIMENT:
    command=[sys.executable,str(RUNNER),'--run-id',RUN_ID,'--seeds',*[str(seed) for seed in SEEDS]]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); tail=[]
    for line in tqdm(process.stdout,desc='selective EB gate',unit='line'):
        print(line,end=''); tail=(tail+[line])[-120:]
    if process.wait(): raise RuntimeError('runner failed:\\n'+''.join(tail))
else: print('RUN_EXPERIMENT=False: existing results only')"""),
        cell("code", """summary=pd.read_csv(RESULT/f'{RUN_ID}_3seed_summary.csv')
per_seed=pd.read_csv(RESULT/f'{RUN_ID}_per_seed.csv')
folds=pd.read_csv(RESULT/f'{RUN_ID}_fold_metrics.csv')
classes=pd.read_csv(RESULT/f'{RUN_ID}_class_metrics.csv')
usage=pd.read_csv(RESULT/f'{RUN_ID}_gate_usage.csv')
audit=json.loads((RESULT/f'{RUN_ID}_leakage_audit.json').read_text())
assert audit['test_read'] is False and audit['threshold']==0.05 and audit['threshold_retuned'] is False
assert per_seed.leakage_check.all() and per_seed.nan_as_mutation_count.eq(0).all()
display(summary); display(per_seed.sort_values(['seed','variant'])); display(usage); display(classes.sort_values('delta_gate_vs_eb').head(10))"""),
        cell("code", """pivot=per_seed.pivot(index='seed',columns='variant',values='oof_macro_f1')
ax=pivot.plot(marker='o',figsize=(8,4),title='Selective EB gate: new-seed OOF Macro F1')
ax.set_ylabel('Macro F1'); plt.tight_layout(); plt.show()
delta=per_seed[per_seed.variant.eq('selective_EB_gate')].set_index('seed').delta_vs_eb
ax=delta.plot.bar(figsize=(7,4),title='Gate delta vs P1+EB')
ax.axhline(0,color='black',linewidth=1); ax.set_ylabel('Macro F1 delta'); plt.tight_layout(); plt.show()"""),
    ]
    payload = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (EXP / "exp-selective-eb-gate-01.ipynb").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
