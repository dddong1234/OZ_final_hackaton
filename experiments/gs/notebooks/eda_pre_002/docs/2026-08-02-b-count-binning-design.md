# exp-gs-002-10 설계 — B count 고정 binning

## 목표

08 최종 Logistic Regression 전처리는 변경하지 않고, B 행별 count의 고정 구간 one-hot을 추가했을 때 3-seed OOF Macro F1이 안정적으로 개선되는지 확인한다.

## 고정 기준선

- H-AS backbone
- exact hotspot 4개
- 혼동 암종쌍 contrast 피처
- A_pair-only
- B/V/A count log1p
- Logistic Regression: lbfgs, `C=0.07`, `max_iter=2000`, `class_weight='balanced'`
- Stratified 5-fold, CV seeds `42/2024/777`
- fold-train only, leakage check, test NaN을 mutation event로 만들지 않음

## 추가 B bin 피처

| 원본 count | 기준 상태 | 추가 one-hot |
|---|---|---|
| mutated gene 수 | 0 | 1, 2, 3–4, 5–7, 8+ |
| 전체 event 수 | 0 | 1, 2, 3–4, 5–7, 8+ |
| multi-event gene 수 | 0 | 1, 2+ |

0 구간은 모델 절편의 기준 상태로 남긴다. 따라서 총 12개 피처를 추가한다. 구간은 정답·OOF·평가 데이터가 아닌, 낮은 이산 count 분리와 오른쪽 꼬리의 넓은 묶음이라는 사전 고정 원칙으로 정했다.

## 비교와 판정

- 08 최종 baseline과 B-binning 후보를 같은 3개 seed에서 각각 실행한다.
- seed별 paired delta와 mean/std를 저장한다.
- 3/3 seed가 양수이고, 누수·NaN·수렴 감사 결과가 기준선과 동일할 때만 채택 후보로 검토한다.
- 개선이 없거나 한 seed에서 의미 있는 하락이 나오면 미검출/기각으로 종료한다.
- 결과 후 bin 경계·피처·파라미터를 다시 조정하지 않는다.
