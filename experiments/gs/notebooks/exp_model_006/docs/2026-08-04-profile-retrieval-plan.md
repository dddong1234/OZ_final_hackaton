# exp-profile-retrieval-01 — train-only exact mutation-profile retrieval

## 가설

동일한 전체 `gene__event` 프로필이 outer-fold train에 존재할 때, 그 프로필의 암종 분포는 H0의 일부 오류를 보완할 수 있다. 이는 고정 암종·유전자·hotspot 규칙을 쓰지 않는 데이터 기반 retrieval이다.

## 고정 계약

- Outer CV: Stratified 5-fold, seed 42
- 기준: self-contained H0 (`LR 0.80 + automatic hard-specialist LGBM 0.20`)
- Profile key: 같은 행의 `gene__event`를 중복 제거·정렬한 문자열
- Posterior: outer-fold train class count + class-prior strength 1.0
- 결합: **일치 행에만** `0.80 × H0 + 0.20 × profile posterior`; 불일치 행은 H0 유지
- test는 읽지 않으며 train/test 결합, 고정 class/gene/mutation 규칙, threshold 탐색을 하지 않는다.

## 해석 기준

동일 profile이 train에 재등장하는 비율, purity, retrieval이 H0 오답을 복구한 수와 H0 정답을 망가뜨린 수를 함께 본다. 점수 상승이 작거나 coverage가 낮으면 제출 후보로 승격하지 않고 retrieval 축을 종료한다.
