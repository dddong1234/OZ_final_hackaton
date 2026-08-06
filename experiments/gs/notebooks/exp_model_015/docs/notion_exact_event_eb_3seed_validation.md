# Exact-event Empirical-Bayes 3-seed 검증

## 결론

**Exact-event EB는 3개 seed와 15개 outer fold에서 모두 기준 H0를 넘어, 현재 채택 가능한 신규 LR 분기입니다.**

- 기준 H0 Selective-EB: `0.547256 ± 0.003723`
- Exact-event EB: **`0.568441 ± 0.002310`**
- 평균 개선: **`+0.021186 ± 0.002115`**
- seed별 최소 개선: **`+0.018750`**
- fold 상승: **15 / 15**

이번 결과는 특정 hotspot이나 암종쌍을 사람이 지정한 결과가 아닙니다. 각 fold의 train 데이터에서 자동으로 관찰된 정확 변이 이벤트를 암종별 증거로 변환한 결과입니다.

---

## 1. 실험 목적

기존 H0는 `gene × functional event type` 단위의 암종별 증거를 사용한다. 예를 들어 동일 유전자라도 missense인지 nonsense인지 정도까지는 구분하지만, `R132H`와 같은 정확한 변이 사건은 완전히 분리하지 않는다.

이번 실험은 정상화된 mutation 문자열을 이용해 `gene__exact_event`를 자동 생성하고, 각 사건이 26개 암종을 얼마나 지지하거나 반박하는지를 Empirical-Bayes 방식으로 압축했다.

목표는 희소한 정확 변이를 그대로 수천 개의 one-hot 열로 추가하는 것이 아니라, **환자당 26개 암종 evidence score로 압축해 일반화 가능한 신호로 사용하는 것**이었다.

---

## 2. 사용한 전처리와 모델 구성

### 기존 H0 구조화 전처리

- 유전자별 mutation 여부 binary
- 행별 mutation burden 및 event-type count
- truncating event 요약
- fold-train에서만 선택한 recurrent missense event
- amino-acid substitution direction(A-pair) `log1p` count
- event topology/diversity 요약
- `gene × event-type` 암종별 EB score

### 이번에 추가한 Exact-event EB

각 mutation cell에서 WT·빈 문자열·NaN을 제외하고 이벤트를 파싱한다. 같은 이벤트가 반복되면 환자·유전자 안에서 한 번만 유지한다.

각 outer fold에서:

1. outer-fold train에서만 `gene__normalized_exact_event` vocabulary 생성
2. 정확 변이별 전체 발생률을 prior로 사용
3. 암종별 발생률을 전역 prior 쪽으로 posterior shrinkage
4. 각 환자의 활성 exact event를 합산하여 암종별 26개 evidence score 생성
5. outer train 내부 5-fold OOF score로만 평균·표준편차를 fit하고 표준화
6. validation에는 train에서 학습된 vocabulary·posterior·scaler를 적용만 함

최종 분류 구조는 기존 H0와 동일하게 유지했다.

```text
Selective exact-event EB Logistic Regression 80%
  + automatic LGBM specialist                 20%
```

Selective gate도 기존 고정 규칙 그대로 사용했다. exact-event LR의 top-1/top-2 확률 margin이 `0.05` 미만이면 non-EB LR 확률을 사용하고, 그 외에는 exact-event EB LR 확률을 사용한다. 새 threshold나 blend 비율 탐색은 하지 않았다.

---

## 3. 검증 환경

| 항목 | 고정값 |
| --- | --- |
| 데이터 | Train 약 6,201행, 26개 암종, 약 4,384개 유전자 |
| 평가 지표 | OOF Macro F1 |
| Outer CV | Stratified 5-fold |
| CV seeds | `42 / 777 / 2024` |
| LR | `lbfgs`, `C=0.07`, `max_iter=2000`, `class_weight='balanced'` |
| specialist | fold-train에서 자동 발견한 유사 암종쌍 2개에 대한 LGBM |
| 비교 기준 | 동일 fold의 H0 Selective-EB |

---

## 4. 3-seed 결과

| Seed | H0 Selective-EB | Exact-event EB | 변화 |
| --- | ---: | ---: | ---: |
| 42 | 0.547915 | **0.570154** | **+0.022239** |
| 777 | 0.543247 | **0.565814** | **+0.022568** |
| 2024 | 0.550605 | **0.569356** | **+0.018750** |
| 평균 ± 표준편차 | 0.547256 ± 0.003723 | **0.568441 ± 0.002310** | **+0.021186 ± 0.002115** |

| 보조 지표 | H0 | Exact-event EB |
| --- | ---: | ---: |
| 평균 OOF Accuracy | 0.558405 | **0.584798** |
| 평균 feature 수 | 8,217.53 | 8,243.53 |
| 추가 feature 수 | - | 약 26개 |
| 수렴 경고 | 0 | 0 |
| leakage check | True | True |
| NaN이 mutation으로 파싱된 수 | 0 | 0 |

정확 변이 표현을 추가했지만, LR 입력에는 26개 EB score만 늘어난다. 정확 변이 vocabulary 전체를 one-hot으로 직접 주입하지 않아 feature 차원이 과도하게 증가하지 않는다.

---

## 5. 클래스별 변화

큰 개선이 한 클래스에만 몰리지 않았다. 주요 개선 클래스는 다음과 같다.

| 클래스 | F1 변화 |
| --- | ---: |
| LGG | **+0.09999** |
| GBMLGG | **+0.07498** |
| KIPAN | **+0.05497** |
| KIRC | **+0.05152** |
| DLBC | **+0.05106** |
| PCPG | **+0.04737** |
| TGCT | **+0.03336** |
| BRCA | **+0.02907** |

하락도 함께 확인했다.

| 클래스 | F1 변화 |
| --- | ---: |
| BLCA | -0.02174 |
| CESC | -0.01271 |
| UCEC | -0.00607 |
| OV | -0.00527 |

최대 하락은 `-0.02174`로 사전 안전선인 `-0.05`보다 작다. 따라서 특정 소수 클래스를 크게 희생해 평균 점수만 올린 결과로 보기는 어렵다.

---

## 6. 누수 및 규정 점검

- test 데이터는 OOF 실험에서 읽지 않음
- train/test concat 없음
- vocabulary는 각 outer-fold train에서만 생성
- EB posterior, 표준화, recurrent event, specialist pair는 fold-train에서만 학습
- validation은 transform 및 평가에만 사용
- 고정 암종명, 유전자명, exact mutation/hotspot 목록 없음
- test를 이용한 one-hot, feature selection, scaling, 결측 통계 없음
- WT·빈 문자열·NaN은 event를 만들지 않음
- 모든 seed에서 `leakage_check=True`, `nan_as_mutation_count=0`, convergence warning `0`

---

## 7. 해석과 다음 단계

정확 변이는 희소하지만, 암종별로 매우 강한 구분력을 가질 수 있다. 이번 결과는 `gene × event-type` 수준의 증거만 사용하는 것보다, **발생한 정확 변이 자체를 posterior-shrunk 암종 evidence로 반영하는 편이 더 효과적**임을 보여준다.

다음 제출 후보는 42/777/2024 seed를 각각 full train으로 학습한 뒤 test 확률을 동등 평균하는 Exact-event EB 3-seed bagging 구성이다. 이 제출은 새 파라미터 탐색 없이, 이번 검증에서 고정된 구성만 그대로 적용한다.

> 주의: OOF 상승이 Public LB 상승을 보장하지는 않는다. 따라서 제출 후 LB는 별도 일반화 지표로 기록하되, LB를 보고 threshold·가중치·event 목록을 재탐색하지 않는다.

---

## 결과 파일

- 검증 브랜치: `gs/exp_005`
- 3-seed 검증 커밋: `f0befda` — `seed42 screen에서 통과한 exact-event Empirical-Bayes 구성을 그대로 42/777/2024에서 검증`
- 3-seed 실행기: `common/run_exact_event_eb_3seed_validation.py`
- 실행 노트북: `exp/exp-exact-event-eb-02-3seed-validation.ipynb`
- 3-seed 요약: `result/exp-exact-event-eb-01_3seed_aggregate.csv`
- 채택 결정: `result/exp-exact-event-eb-01_3seed_decision.json`
- seed42 OOF 확률: `result/exp-exact-event-eb-01_seed42_oof_probabilities.csv`
- seed777 OOF 확률: `result/exp-exact-event-eb-01_seed777_oof_probabilities.csv`
- seed2024 OOF 확률: `result/exp-exact-event-eb-01_seed2024_oof_probabilities.csv`
- seed별 클래스/Fold 지표: `result/exp-exact-event-eb-01_seed*_class_metrics.csv`, `result/exp-exact-event-eb-01_seed*_fold_metrics.csv`
