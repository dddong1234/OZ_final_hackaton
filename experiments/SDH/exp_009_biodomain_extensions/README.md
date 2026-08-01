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

## 실행 결과

### 1차 seed 42

| 순위 | Case | 구성 | OOF Macro F1 |
| ---: | --- | --- | ---: |
| 1 | 06 | functional full + A ref→alt pair | **0.46286** |
| 2 | 01 | functional full 기준 | 0.43287 |
| 3 | 04 | functional full + A + S | 0.43258 |

### 3-seed 확인

| 순위 | Case | OOF Macro F1 평균 ± 표준편차 | 기준 대비 |
| ---: | --- | ---: | ---: |
| 1 | 06 | **0.46360 ± 0.00109** | **+0.03171** |
| 2 | 04 | 0.43614 ± 0.00323 | +0.00425 |
| 3 | 01 | 0.43189 ± 0.00325 | 기준 |

모든 확인 fold에서 `ConvergenceWarning`은 0건이었다. ref→alt pair는 20개
아미노산의 치환 방향을 행별 개수로 표현한 블록이며, position·분포 통계를 함께
넣은 것보다 단독 추가가 더 효과적이었다.

## 제출 및 LB 기록

`case_06_plus_A_pair`를 전체 train에 fit해 생성한 제출의 실제 LB Macro F1은
**0.34238**이었다(사용자 제출 결과). OOF 확인값 0.46360과 차이가 커서, 이 FE가
검증 분할에서는 강하지만 리더보드 test 분포에는 그대로 일반화되지 않았을 가능성이
있다. 별도 참고 구현 `biodomain02`의 LB 0.35097보다 **0.00859 낮다**.

이 LB 기록은 전처리 피처나 임계값을 결정하는 데 사용하지 않았으며, 결과 해석과
후속 연구 우선순위를 정하는 사후 관찰로만 기록한다.
