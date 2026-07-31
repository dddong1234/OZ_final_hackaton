# SDH exp_006 실험 요약

## 상태

실행 및 3-seed 확인 완료.

## 가설

gene burden과 token burden이 비슷한 총량을 나타내더라도, 여러 변이가 함께 기록된
유전자 수는 변이 집중도를 별도로 표현하므로 추가 분류 신호를 제공할 수 있다.

## 비교와 판정

- reference: 현재 최고 `types + hotspot50`와 burden 2종
- candidate: reference + `log1p(multi-mutated-gene count)`
- 모델 파라미터는 공용 benchmark에서 변경하지 않는다.
- seed 42로 1차 비교 후 두 case 모두 seed 42/52/62로 확인한다.
- 3-seed paired 증분이 일관될 때만 채택한다.

## 결과

| 구성 | seed 42 OOF Macro F1 | 3-seed 평균 ± 표준편차 |
| --- | ---: | ---: |
| burden 2종 reference | 0.38582 | 0.37770 ± 0.00704 |
| burden 3종 | 0.38504 | **0.38258 ± 0.00218** |

3-seed 평균에서 burden 3종은 reference보다 `+0.00488` 높았다.

seed별 paired 증분은 다음과 같다.

| seed | burden 3종 - burden 2종 |
| ---: | ---: |
| 42 | -0.00078 |
| 52 | +0.00693 |
| 62 | +0.00848 |

## 판정

평균과 안정성은 개선됐지만 3/3 seed에서 모두 양수는 아니므로 **조건부 유망
후보**로 판정한다. 다음 실험의 새 공통 LR 조건인 `C=0.07`,
`max_iter=2000`에서 기존 최고 FE 조합과 함께 다시 비교한다.
