# exp_010 실험 요약

## 목적

exp009의 `functional full + A pair raw`와 챔피언 보고서의 개선 요소를 동일한 LR
조건에서 하나씩 추가한다. 목표는 OOF 상승 자체보다 각 블록의 독립 기여와 누수
안전성을 확인하는 것이다.

## 순차 설계

1. A pair raw → A pair `log1p`
2. S topology/distribution 추가
3. train-only confusion contrast 추가
4. train-only exact mutation top-4 추가

Contrast는 fold train의 label별 유전자 변이율 차이로 선택한다. Exact top-4는 fold
train의 exact event support와 class concentration으로 선택한다. 따라서 validation과
test의 label·통계는 후보 생성에 사용하지 않는다.

## 결과 기록

실행 후 `leaderboard_seed42.csv`와 `leaderboard_confirmation.csv`를 기준으로 아래
표를 채운다.

| Case | 설명 | 1차 OOF Macro F1 | 3-seed 평균 ± 표준편차 | 증분 |
| --- | --- | ---: | ---: | ---: |
| 01 | exp009 pair raw | 미실행 | 미실행 | 기준 |
| 02 | + pair log1p | 미실행 | 미실행 | 02-01 |
| 03 | + S | 미실행 | 미실행 | 03-02 |
| 04 | + train contrast | 미실행 | 미실행 | 04-03 |
| 05 | + train exact top-4 | 미실행 | 미실행 | 05-04 |

## 판정 규칙

- 한 단계 추가 후 3-seed 평균이 상승하고 표준편차가 과도하게 늘지 않는지 확인
- 모든 후보 생성 규칙이 fold train에서만 학습되는지 확인
- LB 제출은 Case 05까지 무조건 진행하지 않고, OOF 안정성과 누수 점검 후 결정
