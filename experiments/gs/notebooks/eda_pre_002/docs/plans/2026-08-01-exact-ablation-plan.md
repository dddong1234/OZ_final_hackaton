# exp-gs-002-04 Exact Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** H-AS exact mutation 4개 ablation을 3-seed OOF로 재현 가능하게 실행하는 노트북을 만든다.

**Architecture:** 공통 실행기의 `CANDIDATES`에 exact 하나씩 제외한 네 후보를 추가한다. 노트북은 기준 후보와 네 제거 후보를 같은 Logistic/CV 조건에서 실행하고, 이미 생성된 CSV를 읽어 요약·비교·시각화를 만든다.

**Tech Stack:** Python, pandas, scikit-learn, matplotlib, tqdm, Jupyter Notebook.

## Global Constraints

- 프로젝트 변경은 `experiments/gs` 안에서만 수행한다.
- LogisticRegression은 `lbfgs`, `C=0.07`, `max_iter=2000`, `class_weight=balanced`로 고정한다.
- StratifiedKFold 5-fold와 seeds `42`, `2024`, `777`을 고정한다.
- fold-train only, `leakage_check=True`, `nan_as_mutation_count=0` 검증을 유지한다.
- 외부 파일 import를 사용하지 않는다.
- OOF 결과는 `eda_pre_002/result/`에, 향후 제출 파일은 `experiments/gs/notebooks/submission/`에 저장한다.

---

### Task 1: 공통 실행기 ablation 후보 정의

**Files:**
- Modify: `experiments/gs/notebooks/eda_pre_002/common/exp-gs-002-memory-safe.py`
- Test: AST 기반 후보 레지스트리 확인

**Interfaces:**
- Consumes: `LR_EXACT: tuple[tuple[str, str], ...]`
- Produces: `CANDIDATES`의 `H-AS-LR-exact-minus-*` 4개 `Candidate`

- [ ] **Step 1: 후보 키가 없는 정적 검증을 작성한다**

```python
required = {
    "H-AS-LR-exact-minus-BRAF-V600E",
    "H-AS-LR-exact-minus-IDH1-R132H",
    "H-AS-LR-exact-minus-PIK3CA-H1047R",
    "H-AS-LR-exact-minus-PIK3CA-E545K",
}
assert required.issubset(CANDIDATES)
```

- [ ] **Step 2: 구현 전 검증이 실패함을 확인한다**

Run: `python -c "...required.issubset(CANDIDATES)..."`

Expected: `AssertionError`

- [ ] **Step 3: 각 exact 하나를 제외한 후보를 구현한다**

```python
def without_exact(event: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(item for item in LR_EXACT if item != event)

CANDIDATES["H-AS-LR-exact-minus-BRAF-V600E"] = Candidate(
    "H-AS-LR-exact-minus-BRAF-V600E", H_AS.backbone, without_exact(("BRAF", "V600E"))
)
```

동일한 패턴으로 나머지 세 변이를 정의한다.

- [ ] **Step 4: 후보 수와 제외 대상의 정적 검증을 통과시킨다**

```python
assert len(CANDIDATES["H-AS-LR-exact-minus-BRAF-V600E"].exact_events) == 3
assert ("BRAF", "V600E") not in CANDIDATES["H-AS-LR-exact-minus-BRAF-V600E"].exact_events
assert len(CANDIDATES["H-AS-LR-exact"].exact_events) == 4
```

- [ ] **Step 5: 변경 범위를 확인한다**

Run: `git diff -- experiments/gs/notebooks/eda_pre_002/common/exp-gs-002-memory-safe.py`

Expected: exact ablation 후보 정의만 추가됨.

### Task 2: exp-gs-002-04 노트북 생성

**Files:**
- Create: `experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-04.ipynb`
- Consumes: 공통 실행기의 `--candidate`, `--model`, `--seed`, `--run-id`
- Produces: seed별 OOF/class F1 CSV와 3-seed summary CSV

- [ ] **Step 1: 실행 전 노트북 구조 검증을 작성한다**

```python
assert notebook["cells"]
assert any("RUN_EXPERIMENT" in "".join(cell.get("source", [])) for cell in notebook["cells"])
assert any("tqdm" in "".join(cell.get("source", [])) for cell in notebook["cells"])
```

- [ ] **Step 2: 노트북이 없는 상태를 확인한다**

Run: `test -f experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-04.ipynb`

Expected: non-zero exit code.

- [ ] **Step 3: 노트북을 생성한다**

포함할 셀:

1. 목적·고정 조건·후보 목록 Markdown
2. 프로젝트 루트, common runner, result 경로, seeds와 후보 목록 설정
3. `RUN_EXPERIMENT=False` 기본 실행 셀: 후보×seed tqdm, subprocess 실패 시 즉시 오류
4. 결과 파일 존재 확인, seed별 OOF 병합, 3-seed mean/std·feature count·warning·누수/NaN 요약 저장
5. 기준 `H-AS-LR-exact` 대비 delta 표와 후보별 Macro F1 bar chart
6. 판정 기준 Markdown

- [ ] **Step 4: JSON·실행 가드·후보 목록을 검증한다**

```python
import json
nb = json.load(open("experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-04.ipynb"))
source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
assert "RUN_EXPERIMENT = False" in source
assert "tqdm" in source
assert "H-AS-LR-exact-minus-BRAF-V600E" in source
assert "nan_as_mutation_count" in source and "leakage_check" in source
```

- [ ] **Step 5: 변경 범위를 확인한다**

Run: `git diff -- experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-04.ipynb`

Expected: 실행 기본값 False와 5개 후보 3-seed 비교 셀 포함.

### Task 3: 실행기 및 노트북 정적 검증

**Files:**
- Test: `experiments/gs/notebooks/eda_pre_002/common/exp-gs-002-memory-safe.py`
- Test: `experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-04.ipynb`

**Interfaces:**
- Consumes: 후보 레지스트리 및 노트북 JSON
- Produces: 실행 전 정적 검증 결과

- [ ] **Step 1: Python 구문을 확인한다**

Run: `python -m py_compile experiments/gs/notebooks/eda_pre_002/common/exp-gs-002-memory-safe.py`

Expected: exit code 0.

- [ ] **Step 2: NaN 파서 자체 점검을 실행한다**

Run: `.venv/bin/python experiments/gs/notebooks/eda_pre_002/common/exp-gs-002-memory-safe.py --self-check`

Expected: `self-check: parser/NaN contract passed`.

- [ ] **Step 3: 노트북 JSON 및 후보 구성을 재검증한다**

Run: `python -c "import json; ..."`

Expected: 5개 후보와 `RUN_EXPERIMENT=False`가 확인됨.

- [ ] **Step 4: 실험 미실행 상태를 확인한다**

Run: `find experiments/gs/notebooks/eda_pre_002/result -name 'exp-gs-002-04_*' -print`

Expected: 노트북 생성 단계에서는 OOF 실험 결과가 생성되지 않음.
