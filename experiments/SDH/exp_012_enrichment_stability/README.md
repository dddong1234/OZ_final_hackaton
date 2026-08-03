# SDH exp_012 — gene×type enrichment 안정화와 하락 클래스 분석

exp011의 새 챔피언인 `B04 + gene×event-type class-enrichment`를 독립적으로
재현하고, 모델을 바꾸지 않은 상태에서 세 가지 후속 질문을 검증한다.

1. LIHC, DLBC, HNSC, LUSC의 class score 분포와 주요 오분류 상대는 무엇인가?
2. token 최소 support와 shrinkage를 소수 값만 변경하면 안정성이 개선되는가?
3. 26개 class score 전체와 하락 클래스 score를 제외한 표현 중 무엇이 나은가?

## 고정 조건

- GS B04 원본 파이프라인
- Logistic Regression `C=0.07`, `max_iter=2000`, `class_weight=balanced`
- Stratified 5-fold
- seed42 전체 독립 screen
- B04를 포함한 상위 비기준 2개를 seed42/52/62로 확인
- enrichment outer-train 입력은 내부 5-fold OOF cross-fit

## Case

| Case | 변경점 |
| --- | --- |
| 00 | B04 |
| 01 | exp011 winner: support10, shrink20, class score 26개 |
| 02 | support5 |
| 03 | support20 |
| 04 | shrink10 |
| 05 | shrink50 |
| 06~09 | LIHC, DLBC, HNSC, LUSC score를 각각 하나씩 제외 |
| 10 | 하락 4개 class score 동시 제외 |

모든 비기준 case는 Case 01 또는 B04에서 독립적으로 출발하며 누적하지 않는다.
실행은 `experiment.ipynb`를 위에서부터 한 셀씩 진행한다.
