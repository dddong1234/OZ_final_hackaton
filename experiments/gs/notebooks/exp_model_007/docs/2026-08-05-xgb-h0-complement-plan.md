# H0 + XGBoost Complement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement task-by-task.

**Goal:** 규정 안전 H0가 놓치는 비선형 경계를 XGBoost로 보완하는 단일 screen을 제공한다.

**Architecture:** H0의 fold-train 구조화·EB 행렬을 그대로 XGBoost에 전달하고, H0 확률과 XGB 확률을 고정 0.80/0.20으로 결합한다. 모든 vocabulary·enrichment·specialist는 H0 내부에서 fold-train으로 fit되며, XGB는 test를 읽지 않는다.

**Tech Stack:** Python, scipy sparse, XGBoost 3.3, scikit-learn, matplotlib.

## Global Constraints

- 작업과 생성물은 `experiments/gs/notebooks/exp_model_007` 아래에만 둔다.
- 고정 암종명·유전자명·exact mutation 목록을 사용하지 않는다.
- seed42 screen에서는 `test.csv`를 읽지 않는다.
- 후보 selection과 supervised 통계는 outer-fold train에서만 수행한다.
- H0와 XGB의 blend는 `0.80/0.20`으로 고정하고 탐색하지 않는다.

---

### Task 1: XGB 결합 계약

**Files:**
- Create: `common/xgb_complement.py`
- Test: `common/test_xgb_complement.py`

**Interfaces:**
- Produces: `fixed_blend(h0_probability, xgb_probability) -> np.ndarray`
- Produces: `xgb_config(seed, class_count) -> dict`

- [ ] 검증 데이터가 아닌 확률 행렬만으로 0.80/0.20 결합과 정규화를 검사한다.
- [ ] `multi:softprob`, `tree_method='hist'`, 고정 규제 설정을 반환한다.

### Task 2: Fold-safe screen 실행기

**Files:**
- Create: `common/run_h0_xgb_complement.py`

**Interfaces:**
- Consumes: `fit_h0_fold`의 train-only sparse design matrix와 H0 probability.
- Produces: seed summary, fold/class/audit CSV, leakage JSON, matplotlib PNG.

- [ ] 각 outer fold에서 H0와 XGB를 같은 train/validation split으로 fit한다.
- [ ] H0/XGB/fixed blend를 paired 평가한다.
- [ ] seed42 screen과 임의의 `--seeds` 확장 실행을 지원한다.
- [ ] test 미열람, NaN 비변이, fold-train 전용 audit를 저장한다.

### Task 3: 실행 노트북

**Files:**
- Create: `exp/exp-h0-xgb-complement-01.ipynb`

- [ ] 목적·고정 조건·규정 계약을 명시한다.
- [ ] smoke cell, tqdm streaming 실행 cell, CSV/JSON 표시 cell, fold/class 시각화 cell을 둔다.
- [ ] full CV는 사용자가 실행한다.
