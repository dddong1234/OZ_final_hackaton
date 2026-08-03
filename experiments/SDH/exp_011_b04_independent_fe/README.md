# SDH exp_011 — B04 고정 독립 FE ablation

GS B04 챔피언 파이프라인과 LR을 그대로 고정하고, B04에 없는 표현 블록을 각각
하나씩 독립적으로 추가한다. 여러 아이디어를 누적한 상태에서 비교하지 않으므로
각 점수 변화는 해당 블록의 효과로 해석할 수 있다.

## 고정 기준

- B04: `H-AS-LR-exact-confusion-pairs-Apair-log1p`
- LR: `solver=lbfgs`, `C=0.07`, `max_iter=2000`, `class_weight=balanced`
- Stratified 5-fold
- 1차 seed 42, 상위 후보 seed 42/52/62 확인
- B04 코드는 GS 원본 모듈을 직접 불러와 재구현 차이를 방지

## 독립 후보

| Case | B04에 추가하는 블록 | 학습 정보 |
| --- | --- | --- |
| 00 | 없음 | B04 재현 기준 |
| 01 | 고정 burden one-hot bin 12개 | 행 내부 고정 규칙 |
| 02 | event type 비율과 burden 비율 profile | 행 내부 연산 |
| 03 | 암종별 gene enrichment score 26개 | outer fold-train 내부 cross-fit |
| 04 | 암종별 gene×event-type enrichment score 26개 | outer fold-train 내부 cross-fit |
| 05 | 암종별 exact event enrichment score 26개 | outer fold-train 내부 cross-fit |

Enrichment의 outer-train 행도 자신의 label로 만든 가중치를 직접 받지 않도록 내부
5-fold OOF 점수를 만든다. outer validation에는 outer fold-train 전체에서 학습한
가중치를 적용만 한다.

## 최종 결과

`gene×event-type enrichment` 26개를 추가한 Case 04를 채택했다.

- B04 CV: `0.47930 ± 0.00253`
- Case 04 CV: **`0.52395 ± 0.00202`** (`+0.04465`)
- B04 Public LB: `0.38711`
- Case 04 Public LB: **`0.43525`** (`+0.04814`)

계산식, 블록별 상세 설명, 클래스별 변화와 누수 방지 구조는
`TEAM_REPORT.md`에 정리했다.

## 실행

`experiment.ipynb`를 위에서부터 한 셀씩 실행한다. 결과 CSV는 `results/` 아래에
저장되지만 커밋하지 않는다. 독립 후보가 끝나면 양의 개선을 보인 상위 2개 블록의
조합을 선택적으로 확인한다.
