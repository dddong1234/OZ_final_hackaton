# Exact-event Empirical-Bayes Screen

## 가설

H0의 gene×event-type EB는 정확 변이의 allele/codon 정보를 하나의 변이 유형으로 합친다. 이 실험은 모든 fold-train gene__normalized_event를 자동 vocabulary로 만들고, posterior-shrunk class evidence를 26개 점수로 압축해 그 정보를 보완한다.

## 고정 조건

- 기준: H0 Selective-EB LR branch + automatic LGBM specialist
- seed42, Stratified 5-fold
- LR: C=0.07, max_iter=2000, balanced
- H0 specialist 0.20, LR branch 0.80, 기존 Selective-EB margin 유지

## 안전 계약

test를 읽거나 train과 결합하지 않는다. exact vocabulary와 EB posterior, inner OOF standardization은 모두 outer-fold train에서만 fit한다. 고정 암종·유전자·변이 목록, support cutoff, top-k, position bin을 사용하지 않는다. WT/빈 문자열/NaN은 event가 아니다.

## 승격 기준

seed42 H0 대비 Macro F1 +0.015 이상 및 5 fold 중 4개 이상 상승일 때만 3-seed 검증으로 확장한다. 미달하면 이 정확 변이 EB 축은 종료한다.
