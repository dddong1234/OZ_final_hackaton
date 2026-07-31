# SDH exp_005 실험 요약

## 목적

모델 파라미터와 교차검증 조건은 고정하고, mutation 문자열을 더 구체적으로
표현하는 전처리가 암종 분류 Macro F1을 개선하는지 확인했다.

희귀한 exact mutation을 모두 범주형 vocabulary로 만들면 차원이 지나치게 커지므로,
FeatureHasher로 고정 차원에 투영하는 방식을 비교했다. 해싱은 각 샘플 행 내부의
문자열만 변환하며 test 통계나 test vocabulary를 사용하지 않는다.

## 검증 설정

- 모델: 공용 benchmark의 Logistic Regression
- 검증: Stratified 5-fold
- 1차 비교: seed 42
- 확인 실험: 1차 기준 대비 Macro F1이 0.005 이상 높은 후보만 3-seed
- 주 지표: OOF Macro F1

## 결과

### 1차 비교 — seed 42

| 순위 | 전처리 | OOF Macro F1 | OOF Accuracy |
| ---: | --- | ---: | ---: |
| 1 | mutation types + fold-train hotspot 50 | 0.38582 | 0.37978 |
| 2 | mutation types reference | 0.37803 | 0.37413 |
| 3 | gene+codon hash 16K | 0.37052 | 0.38042 |
| 4 | gene+mutation-type hash 4K | 0.36974 | 0.37123 |
| 5 | gene+exact mutation hash 16K | 0.36349 | 0.37720 |
| 6 | gene+exact mutation hash 8K | 0.36237 | 0.37462 |
| 7 | gene+codon hash 8K | 0.36223 | 0.37446 |
| 8 | gene+codon hash 4K | 0.36031 | 0.36768 |
| 9 | exact+codon+gene-type hash 각 4K | 0.35814 | 0.36930 |
| 10 | gene+exact mutation hash 4K | 0.35467 | 0.36591 |

### 3-seed 확인

| 전처리 | OOF Macro F1 평균 ± 표준편차 | OOF Accuracy 평균 |
| --- | ---: | ---: |
| mutation types + fold-train hotspot 50 | **0.37770 ± 0.00704** | 0.37446 |
| mutation types reference | 0.37041 ± 0.00668 | 0.36854 |

hotspot50은 reference 대비 Macro F1이 **0.00729** 높았으며 3-seed에서도 우위가
유지됐다.

## 해석

- 모든 희귀 변이를 보존하는 것보다 fold의 학습 분할에서 반복 관측된 hotspot만
  선택하는 편이 일반화에 유리했다.
- exact mutation 해싱은 희소하고 재현성이 낮은 문자열까지 보존해 기준보다
  낮았다.
- codon 해싱은 exact 해싱보다 대체로 좋았고 16K에서 가장 나았지만, mutation
  types 기준은 넘지 못했다.
- 해시 차원이 커질수록 결과가 개선된 것은 작은 차원에서 충돌 손실이 있었음을
  시사한다. 다만 16K에서도 유효 신호 부족을 극복하지 못했다.
- codon 16K는 Accuracy는 가장 높았지만 Macro F1이 낮았다. 다수 암종의 적중을
  늘리는 대신 소수 암종 성능을 희생했을 가능성이 있어 주 지표 기준으로 채택하지
  않는다.
- 세 종류 해시를 동시에 사용한 case 10은 중복 표현과 잡음이 늘어 가장 좋은
  단일 해시보다 낮았다.

## 결론 및 후속 연구

최종 채택 후보는 **mutation types + fold-train hotspot 50**이다. Feature hashing
계열은 현재 구성에서는 후속 모델 후보에서 제외한다.

다음 실험은 전체 희귀 문자열을 추가하기보다, 각 fold의 학습 분할에서만
유전자-암종 연관 후보를 산출하고 validation에는 적용만 하는 cross-fit 방식처럼
선택적으로 신호를 압축하는 전처리를 검토한다. 외부 논문의 유전자-암종 관계는
피처 선택이나 임계값에 사용하지 않고 train-only EDA 결과의 사후 해석에만 쓴다.
