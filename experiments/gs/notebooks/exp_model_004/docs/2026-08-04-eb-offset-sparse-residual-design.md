# EB-offset sparse residual model 설계

## 목적

P1+Empirical-Bayes(EB)의 암종별 확률을 고정 기준점으로 보존하고, raw mutation 및 gene×event-type 희소 입력으로 남은 판별 오차만 보정한다. 이 모델은 low-margin 구간을 포함해 EB가 체계적으로 틀리는 경우를 직접 수정하는 신규 축이다.

## 모델

환자 (i), 암종 (c)의 최종 score는 아래와 같다.

`score(i, c) = log(P_EB(i, c)) + sparse_feature(i) @ W[:, c] + bias[c]`

- `log(P_EB)`는 P1+EB 확률의 clipped logit이다.
- `W`와 `bias`는 모두 0으로 초기화한다. 학습 전 예측은 P1+EB와 동일하다.
- score는 행별 softmax로 26-class 확률로 정규화한다.
- loss는 class-balanced multiclass cross-entropy + L2 penalty다.

## 희소 입력

- 제공된 원본 gene mutation binary 4,384개
- mutation 문자열에서 얻은 gene×event-type token의 고정 16,384차원 hash binary

해시 함수와 차원은 상수이며 fit하지 않는다. 따라서 vocabulary 선택·test 기반 token 확장·test 빈도 통계가 발생하지 않는다.

## Nested 검증

seed 42의 Stratified outer 5-fold에서 실행한다.

1. 각 outer-train을 inner 5-fold로 나눈다.
2. inner-train에서만 P1+EB를 학습해 inner-validation offset을 생성한다.
3. outer-train의 모든 inner OOF offset과 해당 sparse 입력으로 residual weight를 학습한다.
4. outer-train 전체로 P1+EB를 재학습해 outer-validation offset을 생성한다.
5. 학습된 residual을 outer-validation sparse 입력에 적용한다.

outer-validation label, 확률, offset은 residual 학습에 사용하지 않는다. test 파일은 읽지 않고 제출파일도 생성하지 않는다.

## 고정 파라미터

- seed: 42
- outer/inner fold: Stratified 5-fold
- batch size: 256
- epochs: 40
- learning rate: 0.05
- L2: 0.001
- hash dimension: 16,384
- class-balanced loss
- early stopping, learning-rate schedule, threshold/weight 탐색 없음

## 비교와 승격

P1+EB, fixed selective gate, EB-offset residual을 동일 OOF에서 비교한다.

screen 통과 조건:

- selective gate 대비 Macro F1 `+0.015` 이상
- 5 folds 중 4개 이상 상승
- low-margin (`P1+EB margin < 0.05`) F1이 gate보다 `+0.03` 이상
- 수렴/NaN 오류 없음, `leakage_check=True`, `nan_as_mutation_count=0`
- 특정 클래스 F1 하락이 `-0.05`를 넘지 않음

통과한 설정만 3-seed 검증한다. 미통과하면 learning rate, epoch, L2, hash dimension을 다시 탐색하지 않고 residual 축을 종료한다.
