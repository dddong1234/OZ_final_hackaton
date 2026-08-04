from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def cell(kind: str, source: str) -> dict:
    output = {"cell_type": kind, "metadata": {}, "source": [line + "\n" for line in source.strip().splitlines()]}
    if kind == "code":
        output.update({"execution_count": None, "outputs": []})
    return output


def main() -> None:
    cells = [
        cell("markdown", """# All-class candidate evidence ranker — seed 42 screen

각 환자를 26개 암종 후보 행으로 펼쳐 P1 non-EB/P1+EB 확률과 fold-train EB evidence를 다시 비교한다. ranker 학습 행은 outer-train 내부 inner OOF에서만 만들며, evaluation data는 읽지 않고 제출파일도 생성하지 않는다.

승격 조건은 gate 대비 Macro F1 `+0.015`, 5 folds 중 4개 상승, low-margin F1 `+0.03`, Top-k 회복 개선, 수렴·누수·NaN 계약 통과다."""),
        cell("code", """from pathlib import Path
import json, subprocess, sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
ROOT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'experiments/gs/notebooks/exp_model_004/common/run_all_class_evidence_ranker.py').exists())
RUNNER=ROOT/'experiments/gs/notebooks/exp_model_004/common/run_all_class_evidence_ranker.py'
RESULT=ROOT/'experiments/gs/notebooks/exp_model_004/result'
RUN_ID='exp-all-class-evidence-ranker-01'
SEED=42
RUN_EXPERIMENT=False"""),
        cell("code", """if RUN_EXPERIMENT:
    command=[sys.executable,str(RUNNER),'--seed',str(SEED),'--run-id',RUN_ID]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); tail=[]
    for line in tqdm(process.stdout,desc='candidate evidence ranker',unit='line'):
        print(line,end=''); tail=(tail+[line])[-160:]
    if process.wait(): raise RuntimeError('runner failed:\\n'+''.join(tail))
else: print('RUN_EXPERIMENT=False: existing results only')"""),
        cell("code", """summary=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_summary.csv')
folds=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_fold_metrics.csv')
classes=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_class_metrics.csv')
low=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_low_margin_metrics.csv')
ranker_audit=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_ranker_audit.csv')
audit=json.loads((RESULT/f'{RUN_ID}_seed{SEED}_leakage_audit.json').read_text())
assert audit['test_read'] is False and audit['ranker_training_inner_oof_only'] is True
assert summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()
assert ranker_audit.ranker_training_rows_are_inner_oof.all()
assert (~ranker_audit.outer_validation_used_for_ranker_fit).all()
display(summary); display(ranker_audit); display(low); display(classes.sort_values('delta_ranker_vs_gate').head(10))"""),
        cell("code", """fig,axes=plt.subplots(1,3,figsize=(16,4))
summary.set_index('variant').oof_macro_f1.plot.bar(ax=axes[0],title='OOF Macro F1')
axes[0].set_ylabel('Macro F1')
fold_pivot=folds.pivot(index='fold',columns='variant',values='macro_f1')
(fold_pivot['all_class_ranker']-fold_pivot['selective_EB_gate']).plot.bar(ax=axes[1],title='Ranker − gate by fold')
axes[1].axhline(0,color='black',linewidth=1); axes[1].set_ylabel('Macro F1 delta')
topk=summary.set_index('variant')[['top1_recall','top2_recall','top3_recall']].T
topk.plot.bar(ax=axes[2],title='Candidate inclusion recall')
axes[2].set_ylabel('Recall')
plt.tight_layout(); plt.show()
low.set_index('variant').macro_f1.plot.bar(figsize=(7,4),title='Low-margin Macro F1')
plt.ylabel('Macro F1'); plt.tight_layout(); plt.show()"""),
    ]
    payload = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (EXP / "exp-all-class-evidence-ranker-01.ipynb").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
