# exp-gs-002-05/06 설계

- 공통 기준: 04 best `H-AS + exact 4개`.
- 05: fold-train에서 10회 이상 발생하고 단일 암종 집중도 60% 이상인 exact event를 점수 순 top-3으로 선택해 이진 피처로 추가한다. 기존 exact 4개는 다시 선택하지 않는다.
- 06: KIRC↔KIPAN, LGG↔GBMLGG 각각에서 fold-train mutation rate 차이가 큰 유전자 5개를 선택한다. 각 쌍에 mutation count와 signed contrast score를 추가한다.
- 모든 선택은 fold-train label과 mutation matrix로만 수행하며 validation/test는 후보 선택에 사용하지 않는다.
- Logistic `C=0.07`, `max_iter=2000`, 5-fold, seeds 42/2024/777, NaN·누수 계약을 유지한다.
