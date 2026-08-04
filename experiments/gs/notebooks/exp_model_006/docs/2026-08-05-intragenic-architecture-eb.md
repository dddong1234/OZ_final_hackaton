# exp-intragenic-architecture-eb-01 — 3-seed confirmation screen

## Hypothesis

H0는 개별 mutation, event-type, A-pair, recurrent event, topology와
gene×event-type evidence를 사용한다. 이 실험은 같은 유전자 안에 복수
event가 존재하는 구조를 별도 class-evidence로 보존한다.

## Added signal

`GENE__EVENT_COUNT_2PLUS`, `GENE__EVENT_COUNT_3PLUS`,
`GENE__MULTI_MISSENSE`, `GENE__MISSENSE_PLUS_TRUNCATING`,
`GENE__MULTI_FUNCTIONAL_TYPE`, `GENE__SAME_POSITION_MULTI_EVENT`를
row-local 문자열에서 결정론적으로 생성한다. 특정 gene, cancer 또는 exact
mutation은 고정하지 않는다.

각 token의 26-class Empirical-Bayes posterior log-odds는 outer-fold train
안에서만 fit하며, outer train 내부 cross-fit score로 표준화한다. H0의
LR/LGBM/자동 specialist와 0.80/0.20 결합은 그대로 둔다.

## Locked validation

- CV: Stratified 5-fold × seeds `42/777/2024`
- test.csv 미열람, train/test 결합 없음
- WT/blank/NaN은 event 0개
- 통과: 세 seed 모두 양수, 평균 delta ≥ `+0.008`, 15 folds 중 11개 이상 상승
- 통과하지 않으면 token subtype·threshold·support를 재탐색하지 않고 축 종료
