# exp_011 실험 요약

## 최종 요약

- 채택: `B04 + gene×event-type class-enrichment 26개`
- B04 3-seed CV: `0.47930 ± 0.00253`
- exp011 3-seed CV: **`0.52395 ± 0.00202`** (`+0.04465`)
- B04 Public LB: `0.38711`
- exp011 Public LB: **`0.43525`** (`+0.04814`)
- gene enrichment 결합은 평균 +0.00003, 표준편차 증가로 기각
- 상세 설계·계산식·클래스별 결과: `TEAM_REPORT.md`

## 질문

현재 챔피언 B04 LR을 고정했을 때, 희귀 클래스 신호를 압축하는 train-only
class-enrichment 또는 행 내부 비율 표현이 OOF Macro F1을 개선하는가?

## 검증 원칙

- 모든 case는 B04에서 독립적으로 출발한다.
- B04 원본 seed 42 OOF `0.4781416885`가 재현되는지 먼저 확인한다.
- supervised enrichment는 outer fold-train 안에서 다시 5-fold cross-fit한다.
- validation/test의 label, 분포, vocabulary 빈도는 학습 규칙에 사용하지 않는다.
- seed 42에서 양의 개선을 보인 후보만 조합·3-seed 확인 대상으로 삼는다.

## 결과

| Case | seed 42 OOF Macro F1 | B04 대비 | 3-seed 평균 ± 표준편차 | 판정 |
| --- | ---: | ---: | ---: | --- |
| B04 | 0.47786 | 기준 | 0.47930 ± 0.00253 | 기준 |
| + burden bins | 0.47921 | +0.00134 | 미확인 | 보류 |
| + row profile | 0.47572 | -0.00214 | 미확인 | 기각 |
| + gene enrichment | 0.51157 | +0.03370 | 미확인 | 유망 |
| + gene×type enrichment | **0.52640** | **+0.04854** | **0.52395 ± 0.00202** | **채택** |
| + exact event enrichment | 0.47534 | -0.00253 | 미확인 | 기각 |

상위 두 블록을 합친 gene-type+gene 조합은 3-seed `0.52398 ± 0.00282`였다.
단독 대비 평균 차이는 +0.00003에 불과하고 표준편차가 커 채택하지 않았다.
