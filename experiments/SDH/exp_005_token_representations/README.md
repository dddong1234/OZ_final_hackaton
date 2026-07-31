# SDH exp_005 — Mutation token 표현 비교

모델 파라미터, fold, seed를 변경하지 않고 mutation 문자열 표현만 비교한다. 모든
후보는 공용 `run_preprocessing_benchmark()`의 고정 Logistic Regression을 사용한다.

## 가설

hotspot top N은 빈도가 낮은 구체 변이를 버린다. Feature hashing은 test에서
vocabulary를 학습하지 않으면서 모든 행의 mutation token을 고정 차원에 보존할 수
있다.

## 후보

| Case | 전처리 |
| --- | --- |
| 01 | 변이 유형 reference |
| 02 | reference + fold-train hotspot 50 |
| 03–05 | gene+exact mutation hash 4K/8K/16K |
| 06–08 | gene+codon hash 4K/8K/16K |
| 09 | gene+mutation-type hash 4K |
| 10 | exact+codon+gene-type hash 각 4K |

해시 피처는 샘플 한 행 안에서만 생성되며 test 통계와 test vocabulary를 사용하지
않는다. 1차 seed 42에서 reference보다 0.005 이상 높은 후보만 3-seed confirmation
대상으로 자동 선택한다.

## 결론

3-seed 확인 결과 `mutation types + fold-train hotspot 50`이 OOF Macro F1
`0.37770 ± 0.00704`로 reference의 `0.37041 ± 0.00668`보다 높았다. Feature
hashing 후보는 모두 1차 비교에서 reference보다 낮아 채택하지 않았다.

상세 결과와 해석은 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)에 기록했다.
