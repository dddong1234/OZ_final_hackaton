# H0 Selective-EB LR Branch Replacement

## 가설

H0의 LGBM hard-specialist는 유지하고, 기존 구조화 LR보다 암종별 희귀 변이 증거를 보수적으로 수축하는 Empirical-Bayes LR을 LR 분기에 적용하면 오류 다양성을 얻을 수 있다.

## 고정 비교

- H0: `0.80 × 구조화 LR + 0.20 × fold-local LGBM hard specialist`
- 후보: `0.80 × Selective-EB LR + 0.20 × 동일 specialist LGBM`
- Selective rule: EB 확률 Top-1/Top-2 margin이 `0.05` 미만이면 기존 H0 LR 확률을 사용하고, 그 외에는 EB LR 확률을 사용한다.
- 가중치와 threshold는 탐색하지 않는다.

## 누수 방지

각 outer fold에서 event vocabulary, recurrent feature, enrichment, Empirical-Bayes weight, 표준화, specialist 후보 탐지 및 학습은 fit split만 이용한다. validation에는 학습된 변환을 적용한다. OOF screen은 `train.csv`만 읽으며 test를 읽거나 결합하지 않는다. 고정 암종명·유전자명·exact mutation 목록을 사용하지 않는다.

## 판정

seed42에서 H0 대비 `+0.003` 이상이며 최소 4/5 fold가 상승하면 설정을 바꾸지 않고 42/777/2024로 확장한다. 3-seed에서 모든 delta가 양수, 평균 `+0.003` 이상, 15개 fold 중 11개 이상 상승할 때만 제출 후보로 승격한다.

## 실행

`exp/exp-h0-selective-eb-branch-replacement-01.ipynb`에서 먼저 smoke test를 실행한 뒤 `RUN_EXPERIMENT=True`로 바꾼다. fold가 중단되어도 `result/*checkpoint.npz`에서 같은 run ID로 재개한다.
