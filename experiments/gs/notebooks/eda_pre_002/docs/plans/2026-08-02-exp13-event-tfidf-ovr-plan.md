# exp-gs-002-13 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 동일 event-token TF-IDF 전처리에서 multinomial과 OVR Logistic Regression 및 08 primary 앙상블을 OOF로 공정 비교하는 13번 노트북을 만든다.

**Architecture:** 공통 실행기에 OVR 전용 Logistic Regression 생성과 동일-fold 비교 runner를 추가한다. 노트북은 3개 고정 seed 실행·결과 집계·시각화만 담당한다.

**Tech Stack:** Python, pandas, scipy sparse, scikit-learn, matplotlib, tqdm, Jupyter.

## Global Constraints

- 작업 범위는 `experiments/gs` 내부로 한정한다.
- OOF 단계에서는 test.csv를 읽지 않는다.
- fold-train only; WT/NaN은 mutation token으로 만들지 않는다.
- `C=0.07`, `max_iter=2000`, 5-fold, seeds `42/2024/777`을 고정한다.
- 실제 무거운 OOF 실행은 사용자가 노트북에서 수행한다.

---

### Task 1: OVR 모델 생성 인터페이스 검증

**Files:**
- Create: `experiments/gs/notebooks/eda_pre_002/common/test_exp_gs_002_13_ovr.py`
- Modify: `experiments/gs/notebooks/eda_pre_002/common/exp-gs-002-memory-safe.py`

- [ ] 기존 `make_model`이 OVR 옵션을 받지 못함을 확인하는 테스트를 작성·실행한다.
- [ ] `multi_class='ovr'`인 표준 LR을 생성하도록 최소 확장한다.
- [ ] 테스트와 기존 self-check를 실행한다.

### Task 2: 동일-fold TF-IDF 대조 runner

**Files:**
- Modify: `experiments/gs/notebooks/eda_pre_002/common/exp-gs-002-memory-safe.py`

- [ ] primary·multinomial token·OVR token probability를 동일 outer fold에서 생성한다.
- [ ] 각 단독/0.5 앙상블 Macro F1·Accuracy·클래스 F1·불일치율·안전성 메타데이터를 반환한다.
- [ ] `--event-tfidf-ovr` CLI로 CSV를 저장한다.

### Task 3: 재실행 가능한 13번 노트북

**Files:**
- Create: `experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-13.ipynb`

- [ ] 고정 파라미터와 누수 방지 규칙을 첫 셀에 명시한다.
- [ ] tqdm으로 seed별 runner를 실행한다.
- [ ] per-seed/summary/class-F1 CSV와 OOF plot을 저장한다.

### Task 4: 정적 검증

- [ ] `py_compile`, OVR 단위 테스트, `--self-check`, notebook JSON 검증을 실행한다.
- [ ] 실제 OOF는 실행하지 않았음을 명시한다.
