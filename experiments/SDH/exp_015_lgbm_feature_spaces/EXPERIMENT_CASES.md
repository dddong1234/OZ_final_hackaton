# exp15 case 설명

## A. 표현 축

- F00: exp13 전체 피처 기준선
- F01: burden/variant/topology/contrast/enrichment와 T/R 합계만 남긴 core
- F02: F01에서 supervised enrichment 제거
- F03: enrichment 26개만
- F04: enrichment + burden/variant/topology
- F05: mutation gene + enrichment
- F06: core + 모든 mutation gene

## B. 유전자 support 축

- F07~F12: F06의 mutation gene을 outer-fold train support 2/5/10/20/30/50 이상으로 제한
- validation 빈도는 절대 보지 않는다.

## C. 블록 및 binning 축

- F13/F14/F15: core에 gene truncation/recurrent missense/amino pair를 하나씩 추가
- F16: core에 train-discovered recurrent missense와 amino pair를 함께 추가
- F17: full에서 mutation gene 제거
- F18: full에서 amino pair 제거
- F19: full에서 truncation/recurrent 제거
- F20: full에 고정 count 구간(0, 1, 2, 3~4, 5~7, 8+) 추가
- F21: core + support 10 gene + 고정 count 구간

고정 bin 경계는 데이터 통계로 학습하지 않는 행 단위 변환이다.

모든 case의 공통 입력에서 고정 암종쌍 `C__`와 고정 exact mutation
`D__exact`는 제거한다. exact mutation 정보는 outer-fold train support로
자동 선택되는 `R__` recurrent missense만 허용한다.

## D. gain Top-K 축

- F22~F25: full 피처에서 outer-fold train으로만 작은 selector LGBM을 학습하고 gain Top-250/500/1000/2000을 최종 LGBM에 전달
- 같은 fold의 validation은 중요도 계산에 사용하지 않는다.

## 판정

seed 42에서 F00보다 높은 후보 중 성능, fold 표준편차, 피처 수를 함께 본다. 상위 3개만 3-seed로 재검증하며, 3-seed 평균과 세 seed의 방향이 모두 양호할 때 후속 제출 후보로 남긴다.

## 앙상블 판정

- 상위 순수 LGBM 5개와 LR 챔피언을 LGBM 비중 0.10~0.50으로 혼합한다.
- 상위 LGBM 5개끼리도 0.25/0.50/0.75 비율로 혼합한다.
- 점수뿐 아니라 LR과의 예측 불일치율을 기록한다.
- seed 42에서 가장 좋은 LR+LGBM 조합 3개를 3-seed로 다시 돌린다.
