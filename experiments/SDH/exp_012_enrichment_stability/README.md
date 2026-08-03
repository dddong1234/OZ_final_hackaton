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

## 결과 요약

- seed 42 최고: `case_04_shrink10` — OOF Macro F1 `0.52918`
- 3-seed 최고: `case_04_shrink10` — `0.52824 ± 0.00187`
- 동일 3-seed B04: `0.47930 ± 0.00253` (`+0.04893`)
- exp011 winner: `0.52395 ± 0.00202` (`+0.00428`)
- 차선: `case_02_support5` — `0.52667 ± 0.00229`
- score 제외 실험은 하락 클래스 자체를 회복시키지 못했고 전체 Macro F1도 대부분
  낮아졌다. 최종적으로 26개 class score를 모두 유지한다.
- 모든 실행에서 수렴 경고는 0회였다.

따라서 exp012의 최종 후보는
`B04 + gene×event-type enrichment(support=10, shrinkage=10, 26 scores)`다.
여기서 shrinkage는 LR의 `C`가 아니라 enrichment token weight를 줄이는 FE
파라미터다. Public LB는 아직 확인하지 않았다.

자세한 결과와 클래스별 해석은 `EXPERIMENT_SUMMARY.md`와 `TEAM_REPORT.md`를
참고한다.
