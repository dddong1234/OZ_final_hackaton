# SDH exp_007 실험 요약

## 실험 질문

지금까지 유망했던 burden, mutation type, fold-train hotspot, recurrent
missense, truncating gene 피처를 조합했을 때 고정 Logistic Regression에서 가장
좋은 FE 구성은 무엇인가?

## 고정 조건

- Logistic Regression: `solver="lbfgs"`, `C=0.07`, `max_iter=2000`
- `class_weight="balanced"`
- Stratified 5-fold
- 1차 비교: seed 42
- 확인 실험: 상위 3개 신규 후보와 reference를 seed 42/52/62로 비교
- 데이터 의존적 피처 목록은 각 fold의 학습 분할에서만 산출

## 1차 결과 — seed 42

| 순위 | Case | OOF Macro F1 | 기준 대비 | Accuracy | 피처 수 범위 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | burden3 + types + hotspot50 + truncating + recurrent complement | **0.43287** | +0.02360 | 0.41848 | 7,908–7,995 |
| 2 | burden3 + types + hotspot50 + truncating | 0.42992 | +0.02065 | 0.41622 | 7,771–7,841 |
| 3 | burden3 + types + hotspot50 + recurrent complement | 0.41994 | +0.01067 | 0.40832 | 4,415–4,444 |
| 4 | burden3 + types + recurrent missense(min 5) | 0.41876 | +0.00949 | 0.40768 | 4,389–4,415 |
| 5 | burden3 + types + hotspot100 | 0.41755 | +0.00828 | 0.40606 | 4,334–4,338 |
| 6 | burden3 + types + hotspot50 | 0.41616 | +0.00689 | 0.40639 | 4,284–4,288 |
| 7 | burden3 + types + hotspot50 + min-count10 | 0.41596 | +0.00669 | 0.40622 | 3,867–3,917 |
| 8 | burden2 + types + hotspot50 reference | 0.40927 | 0 | 0.40090 | 4,283–4,287 |

모든 case에서 `ConvergenceWarning`은 0건이었다.

## 3-seed 확인 결과

| 순위 | Case | OOF Macro F1 평균 ± 표준편차 | 기준 대비 | Accuracy 평균 |
| ---: | --- | ---: | ---: | ---: |
| 1 | burden3 + types + hotspot50 + truncating + recurrent complement | **0.43189 ± 0.00325** | **+0.02076** | 0.41918 |
| 2 | burden3 + types + hotspot50 + truncating | 0.42896 ± 0.00295 | +0.01782 | 0.41714 |
| 3 | burden3 + types + hotspot50 + recurrent complement | 0.41924 ± 0.00153 | +0.00811 | 0.40897 |
| 4 | burden2 + types + hotspot50 reference | 0.41113 ± 0.00134 | 0 | 0.40300 |

확인 대상 네 조합 모두 15개 fold에서 수렴 경고가 발생하지 않았다.

## 해석

1. **truncating gene flags가 가장 큰 신규 신호다.** truncating 조합은 3-seed에서
   reference보다 `+0.01782` 높았다. 어느 유전자에 기능상실 형태의 변이가
   존재하는지가 암종 구분에 강한 정보를 제공한 것으로 해석한다.
2. **recurrent missense도 유효하다.** hotspot과 겹치는 token을 제외한 조합은
   reference보다 `+0.00811` 높았다. hotspot 50 밖에도 gene-event 쌍으로 표현할
   때 유효한 반복 변이가 남아 있었다.
3. **두 기능성 블록은 완전히 중복되지 않는다.** functional full은
   truncating-only보다 `+0.00294` 높았다. 다만 이 증분은 작고 피처 수도 늘어나
   별도 paired ablation으로 재확인할 가치가 있다.
4. seed 42에서 burden3는 burden2보다 `+0.00689` 높아 exp_006의 가능성을
   지지한다. 그러나 burden3 단독 조합은 3-seed 확인 대상이 아니므로 독립적인
   3-seed 증분으로 단정하지 않는다.
5. min-count10은 피처를 약 400개 줄였지만 burden3+hotspot50보다 `-0.00020`
   낮아 최고 성능 목적에서는 채택하지 않는다.

## 결론

현재 최고 FE는 **burden3 + mutation types + fold-train hotspot50 + truncating gene
flags + hotspot 비중복 recurrent missense(min 5)**이다.

- 최고 성능 후보: case 07 functional full
- 단순화 후보: case 04 truncating-only
- 가벼운 안정 후보: case 06 recurrent complement

최종 후보로 case 07을 우선 채택한다. 다만 case 07과 case 04의 차이가
`+0.00294`이므로, 다음 단계에서는 같은 fold의 seed별 paired 차이와 암종별 F1을
비교해 recurrent block 유지 여부를 결정한다.

## 규칙 준수

외부 유전자 목록이나 test 데이터 통계를 사용하지 않았다. 모든 데이터 의존적
목록은 각 CV fold의 학습 분할에서만 학습했고 validation에는 적용만 했다.
