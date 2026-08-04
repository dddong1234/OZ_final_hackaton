"""Create the safe, user-run Evidence Set Network notebook."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
TARGET = HERE / "exp" / "exp-class-conditional-evidence-set-network-01.ipynb"


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.strip().splitlines()]}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.strip().splitlines()]}


def main() -> None:
    cells = [
        markdown("""# Class-conditional Evidence Set Network — seed 42 screen

목적: team 3-way ensemble의 암종별 예측과 비교해, fold-train Empirical-Bayes event evidence를 후보 암종별 event set으로 유지하는 listwise network를 검증합니다.

- 데이터는 train만 읽습니다.
- event vocabulary·support·EB weight·standardization은 fit partition에서만 생성합니다.
- team baseline 재현이 `0.54202 ± 0.003` 범위를 벗어나면 결과를 저장하되 승격 판정은 차단합니다.
- seed 42 screen 통과 기준: +0.030, 4/5 fold 상승, 저마진 +0.040, 15개 이상 클래스 개선.
"""),
        code("""from pathlib import Path
import subprocess, sys
from tqdm.auto import tqdm

ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / 'experiments/gs/notebooks/exp_model_005/common/run_evidence_set_network.py').exists())
RUNNER = ROOT / 'experiments/gs/notebooks/exp_model_005/common/run_evidence_set_network.py'
RESULT = RUNNER.parent.parent / 'result'
RUN_ID = 'exp-class-conditional-evidence-set-network-01'
SEED = 42
RUN_EXPERIMENT = False

{'runner': RUNNER, 'result_dir': RESULT, 'seed': SEED, 'train_only_contract': True}"""),
        code("""if RUN_EXPERIMENT:
    command = [sys.executable, str(RUNNER), '--seed', str(SEED), '--run-id', RUN_ID]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in tqdm(process.stdout, desc='evidence-set runner', unit='line'):
        print(line, end='')
        tail = (tail + [line])[-120:]
    if process.wait():
        raise RuntimeError('Evidence Set runner failed:\\n' + ''.join(tail))
else:
    print('RUN_EXPERIMENT=False: 결과 파일만 읽습니다.')"""),
        code("""import json
import matplotlib.pyplot as plt
import pandas as pd

summary = pd.read_csv(RESULT / f'{RUN_ID}_seed{SEED}_summary.csv')
folds = pd.read_csv(RESULT / f'{RUN_ID}_seed{SEED}_fold_metrics.csv')
classes = pd.read_csv(RESULT / f'{RUN_ID}_seed{SEED}_class_metrics.csv')
low_margin = pd.read_csv(RESULT / f'{RUN_ID}_seed{SEED}_low_margin_metrics.csv')
audit = pd.read_csv(RESULT / f'{RUN_ID}_seed{SEED}_nested_audit.csv')
contract = json.loads((RESULT / f'{RUN_ID}_seed{SEED}_feature_contract.json').read_text())
leakage = json.loads((RESULT / f'{RUN_ID}_seed{SEED}_leakage_audit.json').read_text())

assert summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()
assert audit.outer_validation_used_for_eb_fit.eq(False).all()
assert contract['train_only_vocabulary'] and not contract['test_read']
display(summary)
display(audit)
display(low_margin)
display(classes.sort_values('delta_network_vs_team'))
leakage['promotion']"""),
        code("""ax = summary.set_index('variant').oof_macro_f1.plot.bar(figsize=(7, 4), ylim=(0.35, 0.65), title='Evidence Set screen: OOF Macro F1')
ax.set_ylabel('OOF Macro F1')
plt.tight_layout(); plt.show()

folds.pivot(index='fold', columns='variant', values='macro_f1').plot(marker='o', figsize=(8, 4), title='Fold Macro F1')
plt.tight_layout(); plt.show()

classes.sort_values('delta_network_vs_team').plot.barh(x='class', y='delta_network_vs_team', figsize=(7, 7), title='Class F1 delta: network − team')
plt.axvline(0, color='black', linewidth=1); plt.tight_layout(); plt.show()"""),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == '__main__':
    main()
