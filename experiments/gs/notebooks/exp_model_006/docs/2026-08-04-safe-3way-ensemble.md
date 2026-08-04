# exp-safe-3way-ensemble-01

## 목적

고정 hotspot·암종쌍 규칙이 있는 기존 팀 코드에 의존하지 않고, 규정 안전 H0의 fold-train sparse feature 위에서 서로 다른 세 결정 경계의 오류 다양성을 확인한다.

## 모델

- Multinomial Logistic Regression: 0.55
- One-vs-Rest Logistic Regression: 0.30
- Base multiclass LightGBM: 0.15

가중치는 사전에 고정하며 탐색하지 않는다. H0의 자동 hard-specialist 결과는 기준 비교용으로만 남긴다.

## 승격 기준

H0 대비 seed42 OOF가 +0.005 이상이고 5개 fold 중 4개 이상 상승해야 3-seed 검증 후보로 본다. 그렇지 않으면 Selective EB gate를 이 앙상블에 결합하지 않는다.
