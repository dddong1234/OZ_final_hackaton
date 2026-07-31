# SDH exp_004 — 변이 유형 기반 피처 조합

exp_003에서 LR과 LightGBM 모두 1위였던 변이 유형 피처를 기준으로 유전자 빈도
필터와 mutation-token hotspot을 조합한다.

## 후보

| Case | 구성 |
| --- | --- |
| 01 | exp_003 변이 유형 1위 전처리 |
| 02–05 | case 01 + 최소 유전자 빈도 10/15/20/30 |
| 06–08 | case 01 + hotspot 20/50/100 |
| 09 | case 01 + 최소 빈도 10 + hotspot 50 |
| 10 | case 01 + 최소 빈도 20 + hotspot 50 |

모든 후보는 gene/token burden과 변이 유형별 `log1p` 개수를 포함한다. 빈도 기준과
hotspot 목록은 각 fold의 train 부분에서만 학습한다.

## 실행 순서

1. 공용 LR seed 42로 10개 후보 비교
2. case 01보다 Macro F1이 0.005 이상 높은 후보 자동 선택
3. 선택 후보와 case 01을 LR seed 42/52/62로 confirmation
4. 안정적인 최고 후보만 공용 LightGBM으로 2차 검증

JupyterLab에서 `experiment.ipynb`를 실행한다. 결과는 `results/`에 저장되며
커밋하지 않는다.

완료된 LR 3-seed 및 LightGBM 결과는 `EXPERIMENT_SUMMARY.md`에 정리했다.
