"""Create reproducible notebooks for the exp_model_003 plan.

The generator is intentionally local and deterministic: it does not read data,
train models, or inspect result files while creating notebooks.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


def payload(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = '''from pathlib import Path
import subprocess, sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents)
            if (path / "experiments/gs/notebooks/exp_model_003/common/run_p1_eb_axis.py").exists())
COMMON = ROOT / "experiments/gs/notebooks/exp_model_003/common"
RESULT = ROOT / "experiments/gs/notebooks/exp_model_003/result"
RUNNER = COMMON / "run_p1_eb_axis.py"
assert RUNNER.exists(), RUNNER
'''


def run_cell(axis: str, run_id: str) -> str:
    return f'''SEED = 42
RUN_EXPERIMENT = True  # 전체 OOF 실행 전에는 False로 둘 수 있습니다.
RUN_ID = "{run_id}"

if RUN_EXPERIMENT:
    command = [sys.executable, str(RUNNER), "--axis", "{axis}", "--seed", str(SEED), "--run-id", RUN_ID]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in tqdm(process.stdout, desc="OOF runner", unit="line"):
        print(line, end="")
        tail = (tail + [line])[-120:]
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"runner failed (exit={{return_code}}). Last runner output:\\n" + "".join(tail))
else:
    print("RUN_EXPERIMENT=False: 기존 결과만 조회합니다.")
'''


def audit_notebook() -> list[dict]:
    run_id = "exp-eb-topk-parser-audit-01"
    return [
        md("""# EB 기준선: Top-k·파서·지원량 감사

P1+Empirical-Bayes 기준선을 변경하지 않고, top-k oracle·margin·파서 coverage와 후속 구조 피처 지원량을 train OOF로 확인합니다. 실행기는 train만 읽으며 test 데이터는 참조하지 않습니다."""),
        code(SETUP), code(run_cell("audit", run_id)),
        code(f'''from pandas.errors import EmptyDataError

def read_csv_or_empty(path, columns):
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)

summary = pd.read_csv(RESULT / f"{run_id}_seed{{SEED}}_summary.csv")
rank_rows = pd.read_csv(RESULT / f"{run_id}_seed{{SEED}}_rank_rows.csv")
coverage = pd.read_csv(RESULT / f"{run_id}_seed{{SEED}}_parser_coverage.csv")
support = read_csv_or_empty(
    RESULT / f"{run_id}_seed{{SEED}}_structure_support.csv",
    ["gene", "event_count", "same_codon", "position_span"],
)
display(summary); display(coverage); display(support.head(20))
assert int(summary.nan_as_mutation_count.iloc[0]) == 0
assert bool(summary.leakage_check.iloc[0])
if "correct" not in rank_rows:
    rank_rows["correct"] = rank_rows["true_class"].eq(rank_rows["top1_class"])
rank_rows.groupby("correct")[["margin", "entropy"]].mean().plot.bar(figsize=(7, 4), title="P1+EB residual confidence")
plt.tight_layout(); plt.show()'''),
    ]


def axis_notebook(title: str, axis: str, run_id: str, explanation: str) -> list[dict]:
    return [
        md(f"""# {title}

{explanation}

seed 42 screen만 수행합니다. 사전 승격 조건(+0.010, 4/5 fold 양수, 안전성 통과)을 만족한 하나의 구성만 3-seed로 확장합니다."""),
        code(SETUP), code(run_cell(axis, run_id)),
        code(f'''summary = pd.read_csv(RESULT / f"{run_id}_seed{{SEED}}_summary.csv")
folds = pd.read_csv(RESULT / f"{run_id}_seed{{SEED}}_fold_metrics.csv")
display(summary); display(folds)
assert summary.leakage_check.all()
assert summary.nan_as_mutation_count.eq(0).all()
summary.set_index("variant")["oof_macro_f1"].plot.bar(figsize=(8, 4), title="{title}")
plt.tight_layout(); plt.show()'''),
    ]


def gated_notebook(title: str, reason: str, gate_code: str) -> list[dict]:
    return [
        md(f"""# {title}

{reason}

이 파일은 사전 감사 결과가 통과했을 때만 구현·실행하는 조건부 실험입니다. 기본값은 실행하지 않음이며, test 데이터는 읽거나 사용하지 않습니다."""),
        code(SETUP),
        code(gate_code),
        code('''RUN_EXPERIMENT = False
if RUN_EXPERIMENT:
    raise RuntimeError("사전 gate를 통과한 뒤에만 이 노트북의 전용 실행기를 추가·실행합니다. 현재는 계획 고정 단계입니다.")
print("Gate-only notebook: no model was trained.")'''),
    ]


def write(name: str, cells: list[dict]) -> None:
    (EXP / name).write_text(json.dumps(payload(cells), ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    EXP.mkdir(exist_ok=True)
    write("exp-eb-topk-parser-audit-01.ipynb", audit_notebook())
    write("exp-point-process-eb-01.ipynb", axis_notebook(
        "연속 위치·allele Point-process EB", "point", "exp-point-process-eb-01",
        "exact allele → codon → 20aa Gaussian local density → gene×event-type EB 순으로 backoff하여 26×(sum/max/top2) 점수를 만듭니다. 위치·통계·정규화는 outer fold-train과 inner cross-fit에 한정합니다."))
    write("exp-multivariate-eb-01.ipynb", axis_notebook(
        "다변량 Empirical-Bayes", "multieb", "exp-multivariate-eb-01",
        "gene×event-type의 class weight 행렬을 rank 4/8 저랭크 공유 구조로 수축합니다. P1+EB의 토큰 정의는 유지하고, 희귀 암종 간 정보 공유만 검증합니다."))
    write("exp-pretrained-mutation-encoder-01.ipynb", gated_notebook(
        "사전학습 mutation-string encoder (의존성 gate)",
        "외부 환자 데이터·annotation·sequence를 사용하지 않고 literal mutation 문자열만 frozen encoder에 넣는 별도 축입니다. 현재는 패키지/모델 가용성과 대회 허용 범위를 확인하는 단계입니다.",
        '''import importlib.util
status = {{name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "sentence_transformers")}}
display(pd.DataFrame([status]))
assert status["torch"], "torch가 필요합니다"
print("transformers 설치·모델 다운로드는 대회 규정과 팀 승인 후에만 진행합니다.")'''))
    write("exp-macro-f1-decoder-01.ipynb", gated_notebook(
        "저차원 Macro F1 decoder", "Top-k oracle/margin 감사에서 정답이 상위 후보에 자주 남아 있을 때만 global temperature·강한 class bias를 nested OOF로 학습합니다.",
        '''audit = RESULT / "exp-eb-topk-parser-audit-01_seed42_summary.csv"
if audit.exists():
    display(pd.read_csv(audit))
else:
    print("먼저 exp-eb-topk-parser-audit-01을 실행하세요.")'''))
    write("exp-intragenic-architecture-01.ipynb", gated_notebook(
        "동일 유전자 내 복합 변이 구조", "감사에서 다중 이벤트·same-codon·position span 지원량이 충분할 때만 gene×architecture token을 EB 26차원 점수로 압축합니다.",
        '''support_path = RESULT / "exp-eb-topk-parser-audit-01_seed42_structure_support.csv"
if support_path.exists():
    display(pd.read_csv(support_path).head(30))
else:
    print("먼저 parser/support 감사를 실행하세요.")'''))
    write("exp-4state-dependency-01.ipynb", gated_notebook(
        "4-state dependency energy", "co-mutation의 mutation/mutation만이 아니라 mutation/WT, WT/mutation, WT/WT를 fold-train에서 비교하는 후순위 구조 축입니다.",
        '''support_path = RESULT / "exp-eb-topk-parser-audit-01_seed42_structure_support.csv"
if support_path.exists():
    display(pd.read_csv(support_path).head(30))
else:
    print("먼저 support 감사를 실행하세요.")'''))


if __name__ == "__main__":
    main()
