# H0 + Complement NB profile blend

H0의 structured LR/LGBM specialist 확률은 그대로 유지한다. 별도로 outer-fold
train에서만 생성한 `gene×functional-event-type` binary vocabulary를
ComplementNB로 학습하고, 사전에 고정한 확률 결합 `0.75 H0 + 0.25 NB`만
검증한다. 가중치·alpha·token support는 탐색하지 않는다.

각 seed에서 H0와 paired 5-fold 비교한다. 채택은 세 seed delta 양수, 평균
delta `+0.005` 이상, 15 fold 중 11개 이상 상승일 때만 가능하다. test는 읽지
않고, validation-only token은 vocabulary에 추가되지 않는다.
