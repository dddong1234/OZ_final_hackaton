"""Create the runnable, reader-facing notebook for the fixed branch-replacement screen."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parent.parent
NOTEBOOK = EXPERIMENT / "exp" / "exp-h0-selective-eb-branch-replacement-01.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        markdown("""# H0 Selective-EB LR Branch Replacement\n\n## 목표\n\n기준 H0의 `0.80 × 구조화 LR + 0.20 × fold-local LGBM hard specialist`에서 **LGBM 및 specialist를 전혀 바꾸지 않고**, 80% LR 분기만 fold-train Empirical-Bayes LR과 고정 margin gate로 대체한다.\n\n- H0: `0.80 × H0 LR + 0.20 × 기존 specialist LGBM`\n- 후보: `0.80 × Selective-EB LR + 0.20 × 동일 specialist LGBM`\n\n`0.05` margin, `0.80/0.20` 가중치, LR/LGBM 파라미터는 사전에 고정한다. 이 노트북은 기본값에서 결과만 읽으며, 전체 CV는 `RUN_EXPERIMENT=True`일 때만 실행한다."""),
        markdown("""## 규정·누수 계약\n\n- OOF screen은 `train.csv`만 읽고 `test.csv`를 읽지 않는다.\n- train/test 결합, test 기반 vocabulary·통계·스케일링·피처 선택을 하지 않는다.\n- gene×event-type vocabulary와 EB 통계는 outer-fold train에서만 fit한다.\n- validation은 transform/evaluation만 수행한다.\n- 고정 암종명·유전자명·exact mutation 목록을 사용하지 않는다.\n- WT/빈 문자열/NaN은 event가 아니며 `nan_as_mutation_count == 0`을 확인한다.\n- fold별 atomic checkpoint를 `result/`에 저장하므로 중단 후 같은 명령으로 재개한다."""),
        code("""from pathlib import Path\nimport subprocess, sys\nfrom tqdm.auto import tqdm\n\nROOT = Path('/Users/admin/Documents/FinalProject/OZ_fianl_hackaton')\nEXPERIMENT = ROOT / 'experiments' / 'gs' / 'notebooks' / 'exp_model_007'\nRUNNER = EXPERIMENT / 'common' / 'h0_selective_eb_replacement_runner.py'\nRESULT = EXPERIMENT / 'result'\nRUN_ID = 'exp-h0-selective-eb-branch-replacement-01'\nSEEDS = (42,)  # screen 통과 시 (42, 777, 2024)로 한 번만 확장\nRUN_EXPERIMENT = False\nassert RUNNER.exists() and (ROOT / 'data' / 'raw' / 'train.csv').exists()\n{'runner': RUNNER, 'result_dir': RESULT, 'seeds': SEEDS}"""),
        markdown("""### 1. 실행 전 smoke test\n\nparser/확률 결합/경로 계약만 확인한다. 전체 CV는 실행하지 않는다."""),
        code("""subprocess.run([sys.executable, str(RUNNER), '--smoke'], check=True)"""),
        markdown("""### 2. OOF 실행 또는 재개\n\n`RUN_EXPERIMENT=True`로 바꾸면 fold별 checkpoint를 저장한다. 이미 끝난 fold는 건너뛴다."""),
        code("""if RUN_EXPERIMENT:\n    command = [sys.executable, str(RUNNER), '--run-id', RUN_ID, '--seeds', *map(str, SEEDS)]\n    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n    tail = []\n    for line in tqdm(process.stdout, desc='H0 selective-EB folds', unit='line'):\n        print(line, end='')\n        tail = (tail + [line])[-120:]\n    if process.wait():\n        raise RuntimeError('runner failed; checkpoint remains available:\\n' + ''.join(tail))\nelse:\n    print('RUN_EXPERIMENT=False: 기존 결과만 읽습니다.')"""),
        markdown("""## 결과·감사\n\n실행 뒤 아래 셀을 순서대로 실행한다. summary/fold/class/audit이 없으면 먼저 위의 실행 셀을 완료한다."""),
        code("""import json\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\nsummary = pd.read_csv(RESULT / f'{RUN_ID}_seed_summary.csv')\naggregate = pd.read_csv(RESULT / f'{RUN_ID}_aggregate_summary.csv')\nfolds = pd.read_csv(RESULT / f'{RUN_ID}_fold_metrics.csv')\nclasses = pd.read_csv(RESULT / f'{RUN_ID}_class_metrics.csv')\naudit = json.loads((RESULT / f'{RUN_ID}_leakage_audit.json').read_text())\nassert summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()\ndisplay(summary.sort_values(['seed', 'variant']))\ndisplay(aggregate)\naudit"""),
        code("""pivot = folds.pivot_table(index=['seed', 'fold'], columns='variant', values='macro_f1')\nfig, axes = plt.subplots(1, 2, figsize=(13, 4))\npivot.plot(marker='o', ax=axes[0], title='Fold Macro F1')\n( pivot['H0_selective_EB'] - pivot['H0'] ).plot.bar(ax=axes[1], title='Paired delta by fold')\naxes[0].set_ylabel('Macro F1'); axes[1].axhline(0, color='black', lw=1)\nplt.tight_layout(); plt.show()"""),
        code("""f1 = classes.pivot_table(index=['seed', 'class'], columns='variant', values='f1')\ndelta = (f1['H0_selective_EB'] - f1['H0']).groupby('class').mean().sort_values()\nax = delta.plot.barh(figsize=(8, 7), title='Class F1: Selective-EB branch replacement − H0')\nax.axvline(0, color='black', lw=1); plt.tight_layout(); plt.show()\ndisplay(delta.to_frame('f1_delta'))"""),
        markdown("""## 자동 판정\n\nscreen: seed42에서 평균 delta `≥ +0.003` 및 5 folds 중 4개 이상 양수면 3-seed 확정 검증 후보.\n\n3-seed: 새 피처·가중치·threshold를 추가하지 않고 `42/777/2024`에서 모두 양수, 평균 `≥ +0.003`, 15 folds 중 11개 이상 양수여야 채택한다."""),
        code("""decision = audit['decision']\nprint(f'판정: {decision}')\nprint(f"positive folds: {audit['positive_fold_count']}")\nprint(f"fixed threshold: {audit['selective_margin']}, re-tuned: {audit['threshold_retuned']}")"""),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == '__main__':
    build()
