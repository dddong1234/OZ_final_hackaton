"""Create the train-only raw profile audit notebook."""
from __future__ import annotations

import json
from pathlib import Path


TARGET = Path(__file__).resolve().parents[1] / "exp" / "exp-raw-profile-purity-audit-01.ipynb"


def cell(kind: str, source: str) -> dict:
    return {"cell_type": kind, "metadata": {}, "execution_count": None if kind == "code" else None, "outputs": [] if kind == "code" else [], "source": [line + "\n" for line in source.strip().splitlines()]}


def main() -> None:
    notebook = {
        "cells": [
            cell("markdown", """# Raw vs normalized mutation profile purity audit

모델을 학습하지 않고 train 변이 문자열에서만 raw 표기와 normalized 표기의 profile purity를 비교합니다. raw purity가 높더라도 표기/배치 artifact일 수 있으므로, 이 결과만으로 raw formatting feature를 채택하지 않습니다.
"""),
            cell("code", """from pathlib import Path
import subprocess, sys
from tqdm.auto import tqdm

ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / 'experiments/gs/notebooks/exp_model_005/common/run_raw_profile_audit.py').exists())
RUNNER = ROOT / 'experiments/gs/notebooks/exp_model_005/common/run_raw_profile_audit.py'
RESULT = RUNNER.parent.parent / 'result'
RUN_ID = 'exp-raw-profile-purity-audit-01'
RUN_EXPERIMENT = False
{'runner': RUNNER, 'result_dir': RESULT, 'train_only_contract': True}"""),
            cell("code", """if RUN_EXPERIMENT:
    process = subprocess.Popen([sys.executable, str(RUNNER), '--run-id', RUN_ID], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in tqdm(process.stdout, desc='raw-profile audit', unit='line'):
        print(line, end='')
        tail = (tail + [line])[-120:]
    if process.wait():
        raise RuntimeError('Raw profile audit failed:\\n' + ''.join(tail))
else:
    print('RUN_EXPERIMENT=False: 결과 파일만 읽습니다.')"""),
            cell("code", """import json
import pandas as pd
import matplotlib.pyplot as plt

summary = pd.read_csv(RESULT / f'{RUN_ID}_summary.csv')
raw = pd.read_csv(RESULT / f'{RUN_ID}_raw_profile_purity.csv')
normalized = pd.read_csv(RESULT / f'{RUN_ID}_normalized_profile_purity.csv')
audit = json.loads((RESULT / f'{RUN_ID}_audit.json').read_text())
assert audit['test_read'] is False
display(summary)
summary.set_index('profile_kind').weighted_purity.plot.bar(figsize=(5, 3), title='Weighted profile purity')
plt.ylim(0, 1); plt.tight_layout(); plt.show()"""),
        ],
        "metadata": {"kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
