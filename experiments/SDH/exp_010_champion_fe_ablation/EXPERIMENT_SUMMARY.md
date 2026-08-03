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

## 결과

| Case | 설명 | seed 42 OOF Macro F1 | 3-seed 평균 ± 표준편차 | 직전 case 대비 |
| --- | --- | ---: | ---: | ---: |
| 01 | exp009 pair raw | 0.46286 | 미확인 | 기준 |
| 02 | + pair log1p | **0.48208** | **0.48248 ± 0.00098** | **+0.01923** |
| 03 | + S | 0.47844 | 0.47885 ± 0.00047 | -0.00363 |
| 04 | + train contrast | 0.47561 | 0.47406 ± 0.00129 | -0.00479 |
| 05 | + train exact top-4 | 0.47509 | 미확인 | -0.00052 |

3-seed 확인에는 seed 42/52/62를 사용했다. 모든 실행에서 수렴 경고는 0회였다.

## 해석 및 판정

- A pair count의 `log1p` 변환은 raw count보다 seed 42에서 +0.01923, 3-seed
  평균 0.48248로 가장 크고 안정적인 개선을 보였다.
- S topology/distribution은 8개 열을 추가했지만 case 02보다 3-seed 평균이
  0.00363 낮았다.
- train-only confusion contrast도 S 위에서 0.00479 추가 하락했다.
- train-only exact top-4는 seed 42에서 contrast 위에 추가 이득이 없었다.
- 따라서 exp010 채택안은 `case_02_plus_pair_log1p`이며 나머지 누적 블록은
  채택하지 않는다.

## 비교 시 주의

exp010의 기준은 정확한 GS B04 챔피언 파이프가 아니라 SDH exp009의 functional
full/hotspot50 계열이다. B04 seed 42의 0.47814보다 case 02의 0.48208이 0.00394
높지만, 구현 기준이 다르므로 이 결과만으로 챔피언 갱신을 선언하지 않는다. 다음
실험에서는 B04 코드를 그대로 고정하고 신규 표현을 각각 독립적으로 추가한다.

## 누수 점검

- recurrent, contrast와 exact top-4 후보는 각 fold train에서만 선택했다.
- validation에는 fold train에서 확정한 열과 규칙을 적용만 했다.
- test 데이터와 외부 annotation은 후보 생성에 사용하지 않았다.
