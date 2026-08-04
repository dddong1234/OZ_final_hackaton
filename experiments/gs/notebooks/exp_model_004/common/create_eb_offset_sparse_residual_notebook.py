from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def cell(kind: str, source: str) -> dict:
    payload = {
        "cell_type": kind,
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }
    if kind == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


def main() -> None:
    cells = [
        cell(
            "markdown",
            """# EB-offset sparse residual — seed 42 screen

P1+EB의 암종별 확률을 고정 log-probability offset으로 두고, 원본 gene binary와 고정 BLAKE2b gene×event-type hash로 학습한 희소 선형 residual만 더한다. residual weight와 bias는 0에서 시작한다.

각 outer fold에서 residual 학습용 offset은 outer-train 내부 5-fold OOF 확률로만 생성한다. evaluation data는 읽지 않으며 제출파일도 생성하지 않는다. 사전 고정 screen 통과 조건은 gate 대비 `+0.015`, 5 folds 중 4개 상승, low-margin F1 `+0.03`, 클래스 붕괴 없음이다.""",
        ),
        cell(
            "code",
            """from pathlib import Path
import json, subprocess, sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
ROOT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'experiments/gs/notebooks/exp_model_004/common/run_eb_offset_sparse_residual.py').exists())
RUNNER=ROOT/'experiments/gs/notebooks/exp_model_004/common/run_eb_offset_sparse_residual.py'
RESULT=ROOT/'experiments/gs/notebooks/exp_model_004/result'
RUN_ID='exp-eb-offset-sparse-residual-01'
SEED=42
RUN_EXPERIMENT=False""",
        ),
        cell(
            "code",
            """if RUN_EXPERIMENT:
    command=[sys.executable,str(RUNNER),'--seed',str(SEED),'--run-id',RUN_ID]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); tail=[]
    for line in tqdm(process.stdout,desc='EB-offset residual',unit='line'):
        print(line,end=''); tail=(tail+[line])[-160:]
    if process.wait(): raise RuntimeError('runner failed:\\n'+''.join(tail))
else: print('RUN_EXPERIMENT=False: existing results only')""",
        ),
        cell(
            "code",
            """summary=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_summary.csv')
folds=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_fold_metrics.csv')
classes=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_class_metrics.csv')
low=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_low_margin_metrics.csv')
loss=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_epoch_loss.csv')
offset_audit=pd.read_csv(RESULT/f'{RUN_ID}_seed{SEED}_offset_audit.csv')
audit=json.loads((RESULT/f'{RUN_ID}_seed{SEED}_leakage_audit.json').read_text())
assert audit['test_read'] is False and audit['offset_train_inner_oof_only'] is True
assert audit['offset_zero_initialized'] is True and audit['threshold_retuned'] is False
assert summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()
assert offset_audit.offset_train_rows_are_inner_oof.all()
assert (~offset_audit.outer_validation_used_for_residual_fit).all()
display(summary); display(offset_audit); display(low)
display(classes.sort_values('delta_residual_vs_gate').head(10))""",
        ),
        cell(
            "code",
            """fig,axes=plt.subplots(1,3,figsize=(16,4))
summary.set_index('variant').oof_macro_f1.plot.bar(ax=axes[0],title='OOF Macro F1')
axes[0].set_ylabel('Macro F1')
fold_pivot=folds.pivot(index='fold',columns='variant',values='macro_f1')
(fold_pivot['eb_offset_residual']-fold_pivot['selective_EB_gate']).plot.bar(ax=axes[1],title='Residual − gate by fold')
axes[1].axhline(0,color='black',linewidth=1); axes[1].set_ylabel('Macro F1 delta')
low.set_index('variant').macro_f1.plot.bar(ax=axes[2],title='Low-margin Macro F1')
axes[2].set_ylabel('Macro F1')
plt.tight_layout(); plt.show()
loss.pivot_table(index='epoch',columns='fold',values='weighted_loss').plot(figsize=(8,4),title='Residual weighted loss by fold')
plt.ylabel('Weighted loss'); plt.tight_layout(); plt.show()""",
        ),
    ]
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (EXP / "exp-eb-offset-sparse-residual-01.ipynb").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
