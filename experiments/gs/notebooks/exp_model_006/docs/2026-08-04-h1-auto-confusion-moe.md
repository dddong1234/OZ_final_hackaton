# exp-h1-auto-confusion-moe-01

## 가설

H0가 전체 26개 암종의 후보 그룹 확률을 잘 정했지만, 그룹 내부의 유사 암종 순위에서 손실이 남아 있다는 가설을 검증한다.

## 방법

- H0: self-contained 구조화 LR 80% + 자동 pair hard-specialist LGBM 20%.
- 각 outer fold의 train 내부에서만 3-fold H0 OOF를 만들고, 그 confusion으로 26개 라벨을 6개 그룹으로 자동 병합한다.
- 그룹별 balanced LGBM specialist가 group 내부 확률만 예측한다.
- H0가 부여한 그룹 확률 질량은 보존하고, 그룹 내부 비율만 specialist 확률로 교체한다.

## 안전 계약

test를 읽지 않으며 train/test를 결합하지 않는다. 암종·유전자·exact mutation을 고정 목록으로 사용하지 않는다. group discovery, feature vocabulary, enrichment, specialist는 모두 outer-fold train에서만 fit한다. WT·빈 문자열·NaN은 event가 아니다.

## 판정

이번 self-contained H0 seed42 `0.544744`를 비교 기준으로 한다. `+0.015` 이상과 5 fold 중 4개 이상 상승, low-margin F1 하락이 `-0.003` 이내일 때 강한 검증 후보로 분류한다.
