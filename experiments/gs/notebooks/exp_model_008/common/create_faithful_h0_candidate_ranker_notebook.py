"""Create the user-run H3 faithful H0 candidate-ranker notebook."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve()
EXP_DIR = HERE.parent.parent
OUTPUT = EXP_DIR / "exp" / "exp-faithful-h0-candidate-ranker-01.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python (.venv)", "language": "python", "name": "python3"}
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# H3 — Faithful H0 Candidate Evidence Ranker\n\n"
            "목표는 기존 H0 Selective-EB를 유지한 채 모든 26개 암종 후보의 evidence 분포를 shared pairwise ranker로 재정렬해 **큰 로컬 점프**를 확인하는 것입니다."
        ),
        nbf.v4.new_markdown_cell(
            "## 변경하지 않는 계약\n\n"
            "- 기준 H0: `0.80 × Selective-EB LR + 0.20 × 자동 LGBM specialist`\n"
            "- Selective-EB margin: 사전 고정 `0.05`\n"
            "- outer CV: seed42 Stratified 5-fold / outer train 안쪽 inner 3-fold OOF\n"
            "- 모든 event vocabulary, EB 가중치, ranking meta-feature, alpha 선택은 outer-train 안에서만 fit\n"
            "- test.csv 미열람, train/test concat 금지, 고정 암종·유전자·exact mutation 규칙 금지\n"
            "- correction strength는 inner OOF에서만 `{0.10, 0.20}` 중 하나를 고정 선택\n"
            "- NaN/WT/blank는 이벤트가 아니며 `nan_as_mutation_count=0`을 검증"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import subprocess, sys\n"
            "from tqdm.auto import tqdm\n\n"
            "ROOT = Path('/Users/admin/Documents/FinalProject/OZ_fianl_hackaton')\n"
            "EXP_DIR = ROOT / 'experiments' / 'gs' / 'notebooks' / 'exp_model_008'\n"
            "RUNNER = EXP_DIR / 'common' / 'run_faithful_h0_candidate_ranker.py'\n"
            "RESULT = EXP_DIR / 'result'\n"
            "RUN_ID = 'exp-faithful-h0-candidate-ranker-01'\n"
            "SEED = 42\n"
            "RUN_EXPERIMENT = True\n"
            "assert RUNNER.exists() and (ROOT / 'data/raw/train.csv').exists()\n"
            "print({'runner': RUNNER, 'result': RESULT, 'seed': SEED, 'run': RUN_EXPERIMENT})"
        ),
        nbf.v4.new_code_cell(
            "if RUN_EXPERIMENT:\n"
            "    process = subprocess.Popen(\n"
            "        [sys.executable, str(RUNNER), '--seed', str(SEED), '--run-id', RUN_ID],\n"
            "        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,\n"
            "    )\n"
            "    tail = []\n"
            "    for line in tqdm(process.stdout, desc='H3 faithful ranker folds', unit='line'):\n"
            "        print(line, end='')\n"
            "        tail = (tail + [line])[-120:]\n"
            "    if process.wait():\n"
            "        raise RuntimeError('H3 runner failed:\\n' + ''.join(tail))\n"
            "else:\n"
            "    print('RUN_EXPERIMENT=False: 기존 결과만 읽습니다.')"
        ),
        nbf.v4.new_markdown_cell("## 결과와 자동 판정\n\n승격은 `+0.015`, 4/5 fold 상승, low-margin 보호, fold/class 편중 방지를 모두 만족해야 합니다."),
        nbf.v4.new_code_cell(
            "import json\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "prefix = RESULT / f'{RUN_ID}_seed{SEED}'\n"
            "summary = pd.read_csv(prefix.with_name(prefix.name + '_summary.csv'))\n"
            "folds = pd.read_csv(prefix.with_name(prefix.name + '_fold_metrics.csv'))\n"
            "classes = pd.read_csv(prefix.with_name(prefix.name + '_class_metrics.csv'))\n"
            "topk = pd.read_csv(prefix.with_name(prefix.name + '_topk.csv'))\n"
            "low = pd.read_csv(prefix.with_name(prefix.name + '_low_margin.csv'))\n"
            "audit = json.loads(prefix.with_name(prefix.name + '_leakage_audit.json').read_text(encoding='utf-8'))\n\n"
            "assert summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()\n"
            "display(summary)\n"
            "display(folds.pivot(index='fold', columns='variant', values='macro_f1'))\n"
            "display(low)\n"
            "audit['selection']"
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(17, 4))\n"
            "folds.pivot(index='fold', columns='variant', values='macro_f1').plot(marker='o', ax=axes[0], title='Fold Macro F1')\n"
            "classes.sort_values('delta_f1').plot.barh(x='class', y='delta_f1', ax=axes[1], legend=False, title='Class F1 delta')\n"
            "axes[1].axvline(0, color='black', linewidth=1)\n"
            "topk.pivot(index='k', columns='variant', values='recall').plot.bar(ax=axes[2], title='Top-k recall')\n"
            "plt.tight_layout(); plt.show()"
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    nbf.validate(nb)
    print(OUTPUT)


if __name__ == "__main__":
    main()
