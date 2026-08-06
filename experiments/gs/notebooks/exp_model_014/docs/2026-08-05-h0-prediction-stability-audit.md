# H0 Fold-aligned Prediction Stability Audit

## 목적

실제 제출에 사용한 3-seed H0 bagging이 단일 seed보다 어떤 오류를 회복하거나 새로 만드는지 확인한다. 새 모델·피처·가중치·임계값을 만들지 않는 감사 실험이다.

## 입력과 안전성

`exp_model_010`에서 같은 outer split의 validation 행을 세 seed 모델 모두 학습에서 제외한 OOF 확률만 읽는다. 따라서 seed 간 평균은 유효한 fold-aligned OOF다. 실행기는 `test.csv`를 읽지 않고 모델을 재학습하지 않는다.

## 산출물

- seed별·bagged OOF Macro F1/Accuracy
- 행별 seed 일치도, 확률 분산, margin, entropy
- 단일 seed42 대비 bagging의 recovered/broken 오류
- 클래스별 F1 변화와 예측 안정성별 성능

## 해석

bagging이 회복 오류보다 더 많은 정확 예측을 훼손하거나 불안정 행에서만 개선하면, 추가 seed 수 확대를 성능 전략으로 채택하지 않는다. 반대로 안정적으로 오류를 회복하는 클래스·행 유형이 확인될 때만 다음 일반화 실험의 근거로 사용한다.
