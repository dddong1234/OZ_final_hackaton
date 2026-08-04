# exp_model_003 신규 축 실험 계획

## 고정 기준선

- Baseline: P1 + Empirical-Bayes enrichment LR
- 3-seed OOF Macro F1: 0.533739 ± 0.001667
- LR: lbfgs, C=0.07, max_iter=2000, class_weight=balanced
- CV: Stratified 5-fold; screen seed 42, confirm 42/777/2024

## 전역 안전 계약

- 제공 train mutation string과 fold-train label만 사용한다.
- 외부 data, annotation, sequence/pathway mapping, test 통계/선택/학습은 사용하지 않는다.
- 모든 supervised weight, vocabulary, standardization, selection은 fold-train 내부에서만 fit한다.
- NaN은 mutation/token/event가 아니며 결과에 nan_as_mutation_count=0을 기록한다.

## 생성 순서

1. `exp-eb-topk-parser-audit-01.ipynb`: 기존 EB OOF의 rank/oracle/margin과 parser/구조 support를 진단한다.
2. `exp-point-process-eb-01.ipynb`: exact allele→same codon→local position density→gene×type EB backoff 78 score를 screen한다.
3. `exp-multivariate-eb-01.ipynb`: 독립 EB score matrix의 low-rank class sharing rank 4/8을 screen한다.
4. `exp-macro-f1-decoder-01.ipynb`: audit gate 통과 시에만 class bias/temperature를 nested OOF로 학습한다.
5. `exp-intragenic-architecture-01.ipynb` 및 `exp-4state-dependency-01.ipynb`: audit support 통과 시에만 실행한다.
6. `exp-pretrained-mutation-encoder-01.ipynb`: pretrained model 자체만 허용하며, dependency가 없으면 명확하게 실행 중단한다.

## 승격 규칙

- Screen: EB 대비 +0.010 이상, 4/5 fold 상승, warning 0, leakage true, NaN mutation 0.
- Confirm: 세 seed 양수, 평균 +0.010 이상, 최소 seed +0.005 이상, 11/15 fold 상승, class F1 -0.05 이상 붕괴 없음.
- 돌파구: 3-seed 평균 +0.020 이상 및 모든 seed +0.010 이상.
