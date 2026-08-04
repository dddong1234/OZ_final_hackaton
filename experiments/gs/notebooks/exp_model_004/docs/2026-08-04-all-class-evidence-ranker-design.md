# All-class candidate evidence ranker 설계

## 목적

P1+Empirical-Bayes(EB)가 생성한 암종별 확률과 token 증거를 환자×후보 암종 단위에서 직접 비교하는 재순위화 모델을 검증한다. 기존 확률을 단순 혼합하거나 Top-3 후보만 다시 고르는 방식이 아니라, 각 환자의 26개 암종 후보를 모두 평가하므로 base model의 Top-3 밖에 있던 정답도 후보로 남는다.

## 고정 범위

- 실행 위치: `experiments/gs/notebooks/exp_model_004`.
- 1차 screen은 seed `42`와 Stratified 5-fold만 사용한다.
- base: P1 non-EB LR와 P1+EB LR. Selective gate는 비교 기준일 뿐 ranker 입력에 포함하지 않는다.
- 후보 모델: class-balanced Logistic Regression 하나만 사용한다. LightGBM ranker·threshold 탐색·제출 생성은 포함하지 않는다.
- train만 읽는다. test 파일, test 통계, test vocabulary, test scaling은 사용하지 않는다.
- base와 ranker의 모든 supervised statistics는 outer fold의 train 안에서만 만든다.

## 데이터 흐름

각 outer fold에서 다음을 수행한다.

1. outer-train을 inner 5-fold로 다시 나눈다.
2. 각 inner-train에서 P1 non-EB/P1+EB와 EB token 통계를 fit하고, inner-validation의 base 확률과 후보별 EB 증거를 생성한다.
3. inner OOF 결과를 26개 후보 행으로 펼쳐 ranker를 학습한다. 정답 후보 행의 label은 1, 나머지 25개 후보 행의 label은 0이다.
4. outer-train 전체로 base/EB 통계를 다시 fit하고, outer-validation의 후보 행 26개씩을 생성한다.
5. ranker의 후보 positive score를 환자 행별 softmax로 정규화하여 26-class 확률로 변환한다.

## 후보별 입력

- P1 non-EB 후보 확률 및 logit
- P1+EB 후보 확률 및 logit
- 후보 확률과 해당 환자의 최고 경쟁 후보 확률 차
- 후보 암종의 EB evidence 합
- 환자 내 EB evidence 최대값과 상위 3개 evidence 합
- 음수 EB evidence 합과 최소 evidence
- 양수/음수 evidence token 수
- mutation burden의 `log1p`와 burden으로 정규화한 evidence
- low-support token evidence와 일반 token evidence
- P1 non-EB와 P1+EB의 후보 top-1 일치 여부
- 후보 암종 ID one-hot (ranker의 class-specific intercept 역할)

후보 token support와 모든 evidence 값은 해당 fit partition의 train으로만 계산한다. 후보 행별 결측은 0으로 처리하며, NaN은 mutation event로 만들지 않는다.

## 평가와 승격

동일 outer fold에서 P1+EB, fixed selective gate, candidate ranker의 OOF Macro F1을 비교한다.

screen 통과 조건:

- selective gate 대비 Macro F1 `+0.015` 이상
- 5 folds 중 4개 이상 상승
- low-margin (`P1+EB margin < 0.05`) 구간 F1 `+0.03` 이상
- P1+EB 대비 Top-3 정답 포함률 또는 oracle Macro F1 개선
- 수렴 경고 0, `leakage_check=True`, `nan_as_mutation_count=0`

통과한 경우에만 설정을 고정해 seeds `42/777/2024`의 3-seed 검증 파일을 별도로 생성한다. 통과하지 않으면 LightGBM ranker와 추가 파라미터 탐색은 수행하지 않는다.

## 기록물

- seed summary, fold metrics, class metrics
- 후보별 OOF score와 patient-level OOF probabilities
- low-margin 및 Top-k 회복 지표
- config 및 leakage audit JSON
- score/F1/Top-k 시각화가 포함된 실행 노트북

## 오류와 안전 계약

- inner OOF가 아닌 확률로 ranker를 학습하면 즉시 오류로 중단한다.
- 각 환자마다 정확히 26 후보 행과 26개 정규화 확률이 생성되는지 assert한다.
- 모든 확률은 finite 및 row-sum=1인지 assert한다.
- train NaN=0, `nan_as_mutation_count=0`, test 미열람을 audit JSON으로 저장한다.
