"""Create the reader-facing TabPFN screen notebook with nbformat."""
from pathlib import Path
import nbformat as nbf


HERE = Path(__file__).resolve()
NOTEBOOK = HERE.parent.parent / "exp" / "exp-h0-dense-tabpfn-screen-01.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        markdown("# H0 dense TabPFN screen\n\n## tl;dr\nH0의 규정 안전 EB/구조 증거를 저차원 dense 입력으로 고정 TabPFN에 전달해 오류 다양성을 확인합니다. 이 노트북은 seed42 screen만 실행하며 test.csv를 읽지 않습니다."),
        markdown("## Context & Methods\n\n### Key Assumptions\n- H0는 seed42 OOF 0.547915 ± 0.001을 먼저 재현해야 합니다.\n- vocabulary, EB, scaling은 outer-fold train only입니다.\n- fixed `0.80 H0 + 0.20 TabPFN`만 사용하며 HPO·AutoTabPFN·가중치 탐색은 하지 않습니다.\n- 특정 class/gene/exact mutation 규칙과 test 통계는 사용하지 않습니다."),
        code("from pathlib import Path\nimport importlib.util, json, subprocess, sys\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image\n\nROOT = Path('/Users/admin/Documents/FinalProject/OZ_fianl_hackaton')\nRUNNER = ROOT / 'experiments/gs/notebooks/exp_model_013/common/run_h0_dense_tabpfn_screen.py'\nRESULT = ROOT / 'experiments/gs/notebooks/exp_model_013/result'\nRUN_ID = 'exp-h0-dense-tabpfn-screen-01'\nRUN_EXPERIMENT = False  # 준비를 확인한 뒤 True로 변경\nDEVICE = 'cuda'  # CUDA가 없으면 'cpu'; CPU 전체 CV는 매우 느릴 수 있습니다.\nassert RUNNER.exists()\nprint({'runner': RUNNER, 'result': RESULT, 'tabpfn_installed': importlib.util.find_spec('tabpfn') is not None})"),
        markdown("### 1. Dependency check\n\n`tabpfn`이 없으면 먼저 아래 주석 명령을 별도 셀/터미널에서 실행하세요. 첫 fit에서 공식 checkpoint 다운로드 및 라이선스 인증이 필요할 수 있습니다.\n\n`/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m pip install tabpfn`"),
        code("smoke = subprocess.run([sys.executable, str(RUNNER), '--smoke'], text=True, capture_output=True, check=True)\nprint(smoke.stdout)\nassert 'test_read' in smoke.stdout and 'nan_as_mutation_count' in smoke.stdout"),
        markdown("### 2. Run seed42 screen\n\nH0 재현이 기준에서 벗어나면 TabPFN fit을 시작하지 않습니다. fold별로 진행 상황을 출력하며, H0-only checkpoint를 남겨 dependency 오류 뒤에도 불필요한 재학습을 피합니다."),
        code("if RUN_EXPERIMENT:\n    command = [sys.executable, str(RUNNER), '--run-id', RUN_ID, '--device', DEVICE]\n    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n    tail = []\n    for line in process.stdout:\n        print(line, end='')\n        tail = (tail + [line])[-120:]\n    if process.wait():\n        raise RuntimeError('TabPFN screen failed:\\n' + ''.join(tail))\nelse:\n    print('RUN_EXPERIMENT=False: result files are read only.')"),
        markdown("## Results"),
        code("prefix = RESULT / f'{RUN_ID}_seed42'\nsummary_path = prefix.with_name(prefix.name + '_summary.csv')\nif summary_path.exists():\n    summary = pd.read_csv(summary_path)\n    folds = pd.read_csv(prefix.with_name(prefix.name + '_fold_metrics.csv'))\n    topk = pd.read_csv(prefix.with_name(prefix.name + '_topk_metrics.csv'))\n    decision = json.loads(prefix.with_name(prefix.name + '_leakage_audit.json').read_text())\n    assert summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()\n    display(summary)\n    display(topk)\n    display(pd.DataFrame([decision]))\nelse:\n    print('No result yet. Set RUN_EXPERIMENT=True after TabPFN installation and checkpoint access are ready.')"),
        code("for suffix in ('_fold_macro_f1.png', '_class_f1_delta.png', '_topk_recall.png'):\n    image = prefix.with_name(prefix.name + suffix)\n    if image.exists():\n        display(Image(filename=str(image)))"),
        markdown("## Takeaways\n\n- `screen_candidate`는 H0 대비 +0.015 이상, 4/5 fold 양수일 때만 부여됩니다.\n- 미통과이면 TabPFN의 재튜닝이나 blend 비율 탐색은 하지 않고 축을 종료합니다.\n- 통과하면 동일 설정 그대로 42/777/2024 3-seed 검증으로만 확장합니다."),
    ]
    notebook.metadata.kernelspec = {"display_name": "Python (.venv)", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == '__main__':
    build()
