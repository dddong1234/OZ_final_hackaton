# Final Preprocessing Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 팀원이 최종 전처리를 이해하고 외부 로컬 모듈 없이 동일 제출 파일을 재현할 수 있게 한다.

**Architecture:** 노션용 Markdown은 전처리 구성·근거·검증·인사이트를 설명한다. 단일 Python 실행기는 `data/raw`만 읽고 최종 후보의 피처 생성, 전체 train 학습, test 예측, 제출 형식 검증을 하나의 파일에서 수행한다.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy sparse, scikit-learn, tqdm.

## Global Constraints

- 모든 신규 파일은 `experiments/gs` 아래에만 생성한다.
- 실행기는 다른 프로젝트 Python 파일을 import하지 않는다.
- test는 최종 행 단위 변환·예측에만 사용하며 fit·통계·피처 선택에는 사용하지 않는다.
- test NaN은 mutation event로 취급하지 않는다.
- Logistic Regression은 `lbfgs`, `C=0.07`, `max_iter=2000`, `class_weight='balanced'`, `seed=42`를 사용한다.

---

### Task 1: 팀 공유용 전처리 문서

**Files:**
- Create: `experiments/gs/notebooks/eda_pre_002/docs/exp-gs-002-final_preprocessing_notion.md`

**Interfaces:**
- Consumes: 확정 후보 H-AS + exact 4 + confusion-pair contrast + A_pair-only + log1p의 검증 결과.
- Produces: 팀원이 읽는 Markdown 문서.

- [x] 전처리의 출발점과 홍주님 baseline 구성을 약어 대신 의미로 설명한다.
- [x] 추가한 전처리마다 무엇·왜·검증 결과를 명시한다.
- [x] 08의 독립 seed 안정성, 누수·NaN·수렴 점검을 기록한다.

### Task 2: 외부 import 없는 최종 실행기

**Files:**
- Create: `experiments/gs/notebooks/submission/exp-gs-002-final_single_run.py`

**Interfaces:**
- Consumes: `data/raw/train.csv`, `data/raw/test.csv`, 선택적으로 `data/raw/sample_submission.csv`.
- Produces: `experiments/gs/notebooks/submission/submission_exp-gs-002-final_single_run_seed42.csv` 및 메타데이터 CSV.

- [x] row-local mutation parser가 `WT`와 NaN을 빈 이벤트로 처리하는 assert를 둔다.
- [x] train-derived recurrent missense 및 contrast selection을 전체 train에서만 계산한다.
- [x] 최종 피처와 Logistic Regression을 하나의 파일에서 정의한다.
- [x] 제출 전 ID·컬럼·행 수·결측·NaN 처리·수렴 경고를 검증한다.

### Task 3: 정적 검증

**Files:**
- Test: `experiments/gs/notebooks/submission/exp-gs-002-final_single_run.py`

- [x] `py_compile`로 문법을 검사한다.
- [x] `--self-check`으로 NaN parser 계약을 검사한다.
- [x] 결과 파일을 생성하지 않는 `--help` 호출이 가능한지 확인한다.
