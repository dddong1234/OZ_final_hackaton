# H3 — Faithful H0 Candidate Evidence Ranker

## 가설

P1+EB와 Selective-EB는 수천 개의 event evidence를 암종별 합산 점수로 압축한다. 이때 **강한 증거 하나**, **약한 증거 여러 개**, **찬성·반박 증거의 동시 존재**는 같은 합계가 될 수 있다. H3는 이 분포를 후보 암종별 19개 shape feature로 남겨 26개 전체 후보 순위를 재조정한다.

## 기준선

기준은 3-seed 검증을 통과한 H0 Selective-EB이다.

`0.80 × Selective-EB LR + 0.20 × fold-train 자동 LGBM specialist`

H3는 이 확률을 버리지 않는다. H0 log-probability에 shared pairwise ranker residual을 더한 뒤 softmax로 재정규화한다.

## 누수 방지

각 outer fold에서:

1. outer-train 안에서만 inner 3-fold H0 Selective-EB OOF 확률과 EB shape를 생성한다.
2. 이 inner OOF 행으로만 pairwise ranker를 학습한다.
3. alpha `0.10/0.20`은 outer-train inner OOF에서만 선택한다.
4. outer-train 전체로 EB/ranker를 다시 fit하고 outer validation에 적용한다.

seed42 OOF에서는 test.csv를 읽지 않는다. 고정 암종, 유전자, exact mutation 목록을 쓰지 않는다.

## 판정

seed42에서 다음을 모두 만족해야 3-seed로 확장한다.

- H0 대비 Macro F1 `+0.015` 이상
- 5 fold 중 4개 이상 상승
- low-margin F1 하락이 `-0.003` 이내
- 한 fold 또는 한 클래스가 개선 대부분을 만들지 않음
- 수렴 경고 0, leakage True, NaN mutation 0
