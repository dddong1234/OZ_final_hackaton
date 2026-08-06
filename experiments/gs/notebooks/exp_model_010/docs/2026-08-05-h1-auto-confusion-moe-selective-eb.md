# H1 — 자동 혼동 그룹 Mixture-of-Experts

## 목적

최종 H0 Selective-EB 확률을 버리지 않고, outer-fold train의 inner OOF에서 자동으로 발견한 암종 혼동 그룹 내부의 비율만 전담 LGBM expert가 재분배한다. 각 그룹의 H0 확률 질량은 보존하므로 26개 전체 후보의 그룹 간 상대 신뢰는 바꾸지 않는다.

## 안전 계약

- seed42, Stratified outer 5-fold / inner 3-fold
- inner OOF는 그룹 탐지에만 사용하며 outer validation은 그룹 탐지·EB·expert fit에 사용하지 않는다.
- 모든 그룹은 label 이름을 하드코딩하지 않고 fold-train confusion에서 자동 생성한다.
- test는 읽지 않고, train/test concat도 없다.
- H0는 현 제출 파이프라인의 Selective-EB branch와 automatic specialist를 그대로 사용한다.

## 판정

기준 H0는 seed42 `0.547915`이다. H1이 `+0.005` 미만이면 기각, `+0.015` 이상 및 4/5 fold 양수·low-margin 하락 제한을 만족하면 강한 검증 후보로 본다. 통과할 때만 독립 3-seed 검증과 제출 bagging을 진행한다.
