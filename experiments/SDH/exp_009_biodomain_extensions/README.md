# SDH exp_009 — 단백질 표기 구조 피처 확장

exp_007 functional full을 기준으로, 변이 문자열 안에 이미 존재하는 단백질 표기
구조(A)와 행 내부 변이 분포(S)를 추가한다. 외부 hotspot·pathway·임상 annotation은
사용하지 않는다.

## 고정 조건

- Logistic Regression `solver="lbfgs"`, `C=0.07`, `max_iter=2000`
- `class_weight="balanced"`
- Stratified 5-fold
- 1차 seed 42, leaderboard 1·2·3위 모두 seed 42/52/62 확인

## 후보

| Case | 구성 |
| --- | --- |
| 01 | exp_007 functional full 기준 |
| 02 | 기준 + A 전체: ref/alt/pair/position |
| 03 | 기준 + S 전체: count/topology/distribution |
| 04 | 기준 + A + S |
| 05 | 기준 + A ref/alt |
| 06 | 기준 + A ref→alt pair |
| 07 | 기준 + A protein position bins |
| 08 | 기준 + S count/topology |
| 09 | 기준 + S diversity/entropy/dominant share |

A와 S의 zero-variance 출력 열은 각 fold train에서만 제거한다. 기존 base의
truncating/recurrent 목록도 fold train에서만 학습된다.

`experiment.ipynb`는 각 case를 한 셀씩 실행하도록 작성되어 있다. 1차 평가 후
leaderboard의 1·2·3위가 자동으로 confirmation 목록에 들어가므로, 마지막 셀까지
실행해 두면 세 후보가 순서대로 3-seed 확인된다. 실제 학습은 사용자가 직접 실행한다.
