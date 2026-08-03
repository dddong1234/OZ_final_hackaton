# exp_011 실험 요약

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

실행 후 `leaderboard_seed42.csv`, `leaderboard_confirmation.csv`와 클래스별 F1을
기준으로 기록한다.

| Case | seed 42 OOF Macro F1 | B04 대비 | 3-seed 평균 ± 표준편차 | 판정 |
| --- | ---: | ---: | ---: | --- |
| B04 | 미실행 | 기준 | 미실행 | 대기 |
| + burden bins | 미실행 | 미실행 | 미실행 | 대기 |
| + row profile | 미실행 | 미실행 | 미실행 | 대기 |
| + gene enrichment | 미실행 | 미실행 | 미실행 | 대기 |
| + gene×type enrichment | 미실행 | 미실행 | 미실행 | 대기 |
| + exact event enrichment | 미실행 | 미실행 | 미실행 | 대기 |
