# H1 Auto-Confusion MoE Implementation Plan

**Goal:** Train-only H0의 자동 혼동 구조를 이용해 암종 그룹 내부 확률만 보정하는 seed42 OOF 실험을 만든다.

**Architecture:** 각 outer fold에서 H0를 한 번 학습해 기준 확률을 만들고, outer-train의 inner 3-fold H0 OOF confusion으로 6개 클래스를 자동 그룹화한다. 각 그룹의 LGBM specialist는 outer-train만 학습하며, validation에서는 H0의 그룹 확률 질량을 유지한 채 그룹 내부 확률만 재배분한다.

**Global constraints:** `experiments/gs` 밖 실행 의존 금지, test 미열람, fixed class/gene/mutation 규칙 금지, NaN/WT/빈 문자열 event 금지, 모든 선택은 outer-train only.

## Tasks

1. `common/test_h1_auto_confusion_moe.py`에 자동 그룹화·질량 보존·outer validation 미학습 회귀 테스트를 작성한다.
2. `common/h1_auto_confusion_moe.py`에 H0 component 재현, inner OOF group discovery, group specialist 적용을 구현한다.
3. `common/run_h1_auto_confusion_moe.py`에 memory-safe outer loop, CSV/JSON/plot 출력과 판정을 구현한다.
4. `exp/exp-h1-auto-confusion-moe-01.ipynb`에 사용자 실행 셀, tqdm, 결과 표·그래프·자동 판정을 작성한다.
5. 전체 CV 없이 compile, unit tests, static test-read audit, tiny smoke test를 실행한다.
