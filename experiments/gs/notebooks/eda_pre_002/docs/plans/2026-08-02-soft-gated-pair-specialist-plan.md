# exp-gs-002-11 계획 — soft-gated KIRC↔KIPAN specialist

## 문제와 목표

08 최종 LR은 KIRC와 KIPAN을 반복적으로 혼동한다. 11은 26-class primary LR의 전체 암종 확률합을 바꾸지 않고, KIRC/KIPAN 확률 질량 내부 비율만 binary specialist LR로 보정하는지 검증한다.

이 계획은 KIRC↔KIPAN 단일 쌍만 다룬다. LGG↔GBMLGG specialist는 11이 안정적으로 통과한 뒤 별도 실험으로 분리한다.

## 고정 조건

- 08 최종 전처리 전체: H-AS, exact hotspot 4개, contrast 피처, A_pair-only, B/V/A log1p
- primary: 26-class Logistic Regression `lbfgs`, `C=0.07`, `max_iter=2000`, `class_weight='balanced'`
- specialist: 같은 `C`, `max_iter`, `class_weight`, fold별 KIRC/KIPAN train 행만으로 학습하는 binary Logistic Regression
- outer CV: Stratified 5-fold, seeds `42/2024/777`
- 모든 learned feature rule과 모델은 outer fold train만 사용한다.
- test 데이터는 OOF 실험에서 읽거나 참조하지 않는다.

## 확률 보정 설계

outer fold의 validation 행마다 primary 확률을 `p`라 한다.

- `m = p(KIRC) + p(KIPAN)`
- `r_primary = p(KIRC) / m` (`m=0`이면 안전하게 0.5)
- `r_expert = specialist의 KIRC 확률`
- 고정 상수 `ALPHA = 0.30`
- `weight = ALPHA × m`
- `r_final = (1 - weight) × r_primary + weight × r_expert`
- `p_final(KIRC)=m × r_final`, `p_final(KIPAN)=m × (1-r_final)`
- 나머지 24개 클래스 확률은 원본 primary 확률과 완전히 동일하게 유지한다.

확률 질량 `m`이 낮은 행은 expert 영향도 작으므로 별도의 top-2 조건이나 임계값을 두지 않는다. 이는 threshold 탐색을 피하면서도 soft gate를 구현한다.

## 실행 단위

### Unit 1: 확률 반환이 가능한 fold-safe primary/specialist 실행기

**예정 파일**

- `experiments/gs/notebooks/eda_pre_002/common/exp-gs-002-11-specialist-runner.py`

**책임**

- train만 로드하고 08 final feature matrix를 outer fold별로 생성한다.
- primary 26-class LR와 KIRC/KIPAN specialist를 해당 outer train에서 독립적으로 fit한다.
- validation에 대해 primary·expert·blended OOF 확률을 저장한다.
- primary-only와 blended의 Macro F1, accuracy, 클래스별 F1, confusion, runtime, feature 수, convergence warning을 반환한다.

**필수 검증**

- validation 행이 어떤 fit/feature selection에도 포함되지 않는지 확인한다.
- specialist 학습 레이블이 KIRC/KIPAN 두 클래스뿐인지 확인한다.
- 모든 행에서 final 확률 합이 1인지 확인한다.
- KIRC/KIPAN 외 24개 클래스 확률이 primary와 정확히 같은지 확인한다.
- `nan_as_mutation_count=0`, `leakage_check=True`를 저장한다.

### Unit 2: 3-seed OOF 실험 노트북

**예정 파일**

- `experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-11.ipynb`

**책임**

- `42/2024/777`에서 primary-only와 soft-gated specialist를 함께 실행한다.
- tqdm으로 seed·fold 진행을 표시한다.
- seed별 paired delta, 3-seed mean/std, pair 한정 F1, KIRC→KIPAN 및 KIPAN→KIRC 혼동 수 변화를 저장한다.
- 결과 CSV, JSON 감사 파일, 확률 질량/paired delta matplotlib 그래프를 `result/exp-gs-002-11_*`로 저장한다.

**필수 검증**

- `ALPHA=0.30`이 모든 seed에서 동일한지 확인한다.
- baseline과 specialist가 동일 primary feature/configuration을 사용하는지 확인한다.
- 결과 파일만 집계할 때에도 seed 3개와 두 후보가 모두 존재하는지 확인한다.

## 판정 기준

- 3-seed Macro F1 mean이 primary-only보다 높아야 한다.
- 각 seed의 paired delta가 모두 양수여야 한다.
- KIRC/KIPAN 두 클래스의 평균 F1 또는 양방향 혼동 수가 악화되지 않아야 한다.
- 누수·NaN 감사 실패 또는 수렴 경고 증가는 즉시 기각 사유다.
- 기준을 통과해도 LGG↔GBMLGG expert, alpha 변경, 다른 threshold 추가는 다음 별도 실험에서만 한다.

## 범위 밖

- hard 2차 분류 규칙
- top-2 또는 확률합 임계값 탐색
- alpha, specialist 파라미터, 암종쌍을 11 결과에 맞춰 재조정하는 작업
- 평가 데이터 기반 확률 보정 또는 제출 파일 생성
