"""Create the runnable final-submission notebook without executing model training."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve()
EXP_DIR = HERE.parent.parent
OUTPUT = EXP_DIR / "exp" / "exp-h0-selective-eb-submission-01.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python (.venv)", "language": "python", "name": "python3"}
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# H0 Selective-EB 최종 제출 생성\n\n"
            "3-seed에서 통과한 고정 구성으로 최종 train 재학습 및 test 추론을 한 번 수행합니다. "
            "출력은 `experiments/gs/notebooks/submission/`에 저장됩니다."
        ),
        nbf.v4.new_markdown_cell(
            "## 고정 계약\n\n"
            "- 최종 확률: `0.80 × Selective-EB LR + 0.20 × 자동 LGBM specialist`\n"
            "- Selective gate: EB LR의 Top-1−Top-2 margin `< 0.05`이면 non-EB LR, 나머지는 EB LR\n"
            "- margin과 blend 비율은 재탐색하지 않습니다.\n"
            "- vocabulary, recurrent event, EB 통계, 표준화, specialist 암종쌍은 full train에서만 fit합니다.\n"
            "- test는 이미 학습한 변환 적용과 예측에만 사용하며 train/test를 결합하지 않습니다.\n"
            "- 고정 암종명·유전자명·exact mutation 목록은 사용하지 않습니다."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import subprocess, sys\n"
            "from tqdm.auto import tqdm\n\n"
            "ROOT = Path('/Users/admin/Documents/FinalProject/OZ_fianl_hackaton')\n"
            "EXP_DIR = ROOT / 'experiments' / 'gs' / 'notebooks' / 'exp_model_007'\n"
            "RUNNER = EXP_DIR / 'common' / 'h0_selective_eb_submission.py'\n"
            "SUBMISSION_DIR = ROOT / 'experiments' / 'gs' / 'notebooks' / 'submission'\n"
            "RUN_SUBMISSION = True\n\n"
            "assert RUNNER.exists()\n"
            "assert (ROOT / 'data/raw/train.csv').exists()\n"
            "assert (ROOT / 'data/raw/test.csv').exists()\n"
            "assert (ROOT / 'data/raw/sample_submission.csv').exists()\n"
            "print({'runner': RUNNER, 'output_dir': SUBMISSION_DIR, 'run': RUN_SUBMISSION})"
        ),
        nbf.v4.new_code_cell(
            "if RUN_SUBMISSION:\n"
            "    process = subprocess.Popen(\n"
            "        [sys.executable, str(RUNNER)],\n"
            "        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,\n"
            "    )\n"
            "    tail = []\n"
            "    for line in tqdm(process.stdout, desc='final submission training', unit='line'):\n"
            "        print(line, end='')\n"
            "        tail = (tail + [line])[-120:]\n"
            "    if process.wait():\n"
            "        raise RuntimeError('submission runner failed:\\n' + ''.join(tail))\n"
            "else:\n"
            "    print('RUN_SUBMISSION=False: 생성하지 않았습니다.')"
        ),
        nbf.v4.new_markdown_cell("## 결과 검증\n\nCSV의 ID 순서·행 수·클래스와 audit 계약을 확인합니다."),
        nbf.v4.new_code_cell(
            "import json\n"
            "import pandas as pd\n\n"
            "submission_path = SUBMISSION_DIR / 'submission_h0_selective_eb_lr_lgbm_specialist_seed42.csv'\n"
            "audit_path = submission_path.with_suffix('.audit.json')\n"
            "submission = pd.read_csv(submission_path)\n"
            "audit = json.loads(audit_path.read_text(encoding='utf-8'))\n"
            "sample = pd.read_csv(ROOT / 'data/raw/sample_submission.csv')\n\n"
            "assert submission.columns.tolist() == sample.columns.tolist() == ['ID', 'SUBCLASS']\n"
            "assert submission.ID.equals(sample.ID)\n"
            "assert len(submission) == len(sample) and submission.SUBCLASS.notna().all()\n"
            "assert audit['leakage_check'] is True\n"
            "assert audit['nan_as_mutation_count'] == 0\n"
            "assert audit['test_read_for_fit_statistics_selection_or_scaling'] is False\n"
            "display(submission.head())\n"
            "audit"
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    nbf.validate(notebook)
    print(OUTPUT)


if __name__ == '__main__':
    main()
