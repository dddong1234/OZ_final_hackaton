# exp13 이후 모델 다양성 확보 전략

## 1. 한눈에 보는 결론

exp13은 앞으로 진행할 모든 모델 실험의 **공통 기준이자 시작점**이다.

- 기준 단일 모델: Logistic Regression
- 기준 3-seed OOF Macro F1: **0.5282357120**
- 기준 Public LB: exp12와 같은 예측이므로 **0.4388787816**
- 기준 피처: standalone B04 8,399개 + class enrichment 26개
- 기준 누수 계약: train/fold-train에서만 모든 학습형 전처리를 fit

지금까지 여러 FE를 검증하면서 개선 폭이 점차 줄었다. 앞으로는 동일한 LR에
피처를 계속 덧붙이는 것보다, 각 팀원이 하나의 모델 계열을 맡아 그 모델에
적합한 표현을 만들고 마지막에 서로 다른 예측을 앙상블하는 것을 우선한다.

단일 모델 점수가 기준 LR보다 조금 낮더라도 LR이 틀린 샘플을 맞히는 모델은
앙상블에 가치가 있다. 따라서 후속 실험은 단일 Macro F1과 함께 **예측
다양성 및 오답 보완성**을 반드시 평가한다.

---

## 2. 모든 모델이 지켜야 할 공통 계약

### 2.1 데이터와 누수 방지

1. 원본 train과 test를 합쳐 전처리를 fit하지 않는다.
2. vocabulary, 빈도, support, 결측 통계, 피처 선택, 표준화는 train에서만 fit한다.
3. CV에서는 위 결정을 각 fold의 train 분할에서만 다시 학습한다.
4. validation/test는 이미 학습된 변환을 적용만 한다.
5. label을 사용하는 피처는 inner cross-fit으로 학습 행의 자기 label 사용을 막는다.
6. test 행 수, test 전체 통계, test-only token을 모델 결정에 사용하지 않는다.
7. 독립적인 한 샘플 내부의 burden·mutation count·type 비율 계산은 허용한다.

### 2.2 공통 검증 조건

- Stratified 5-fold
- seeds: `42`, `52`, `62`
- 주 지표: OOF Macro F1
- 보조 지표: OOF Accuracy, 클래스별 F1
- 동일 fold index와 동일 클래스 확률 순서 사용
- 모델 선택 전 seed 42로 screen, 채택 후보만 3-seed 확인

### 2.3 확률 파일 계약

최종 앙상블을 다시 학습 없이 검증할 수 있도록 모든 모델은 hard prediction뿐
아니라 확률을 저장한다.

OOF 확률 파일:

```text
ID, SUBCLASS, seed, fold, prob_<class 1>, ..., prob_<class 26>
```

Test 확률 파일:

```text
ID, seed, prob_<class 1>, ..., prob_<class 26>
```

반드시 기록할 메타데이터:

- 실험 ID와 담당자
- 모델과 전체 파라미터
- 사용 피처 및 피처 수
- seed와 fold 정의
- 클래스 이름 및 확률 열 순서
- convergence warning 수
- leakage audit 결과
- 실행 시간

OOF/test probability는 저장소에 커밋하지 않고 각 실험의 `results/`에 보관한다.

---

## 3. 모델별 전처리 연구 방향

공통 출발점은 exp13이지만 모든 모델이 안전 baseline의 약 8.2천개 피처를 그대로 사용할 필요는
없다. 모델이 정보를 처리하는 방식에 맞춰 표현을 다시 설계한다.

### 3.1 Logistic Regression

LR은 현재 챔피언이자 앙상블 기준축이다. 고차원 sparse 이진·count 피처에
강하므로 exp13 표현을 그대로 사용한다.

우선 연구:

- 규제 C는 `0.07`, `max_iter=2000`으로 고정
- class-enrichment score의 안전한 표현 개선
- 희귀 클래스의 독립적인 train-only 피처
- fixed contrast와 fold-train 자동 혼동쌍 비교

새로운 차원의 아이디어가 아니라면 LR 피처를 계속 누적하지 않는다.

### 3.2 LightGBM

LightGBM은 수천 개의 극희소 이진 열보다 저차원 집계형 피처에서 경쟁력이 있을
가능성이 높다.

우선 연구:

- burden·mutation type·topology와 enrichment 26개 중심의 dense 표현
- train-only support로 선택한 핵심 유전자/변이 피처
- count binning 또는 구간화
- 유전자군·공변이 count 요약
- 전체 안전 sparse 입력과 축소 입력 비교

처음에는 팀에서 정한 고정 LGBM 파라미터로 피처 표현만 비교하고, 유망 표현을
확정한 뒤에만 제한적인 모델 파라미터 연구를 진행한다.

### 3.3 CatBoost

CatBoost는 범주 정보를 처리할 수 있지만 4천여 유전자 원본 문자열을 모두
범주형으로 넣는 방식은 너무 무겁고 희귀 범주가 많다.

우선 연구:

- train-only로 선택한 핵심 유전자 mutation type
- 희귀 token을 `OTHER`로 통합한 범주 표현
- 저차원 수치형 집계 + 일부 범주형 피처
- 범주 어휘 및 rare threshold를 fold-train에서만 fit

### 3.4 선형 SVM 또는 SGD 계열

LR과 같은 sparse 표현을 사용할 수 있으면서 손실함수와 결정경계가 달라
앙상블 다양성을 기대할 수 있다.

우선 연구:

- LinearSVC 또는 SGDClassifier
- hinge, modified-huber 등 손실함수 비교
- 확률 앙상블을 위한 fold-safe probability calibration
- LR과의 오답 및 확률 상관 분석

확률 보정기를 사용할 때도 calibration fit은 fold-train 내부에서 수행한다.

---

## 4. 예측 다양성 평가 기준

다양성은 단순히 두 모델이 다른 답을 내는 정도가 아니다. **챔피언 LR이 틀린
샘플을 신규 모델이 맞히는지**가 핵심이다. 모든 계산은 동일한 OOF 행과 fold를
맞춘 상태에서 수행한다.

### 4.1 예측 불일치율

```text
disagreement = 두 모델의 예측이 다른 행 수 / 전체 행 수
```

- 너무 낮으면 앙상블로 얻을 새로운 정보가 적다.
- 너무 높으면 신규 모델이 단순히 약하거나 불안정한지 확인해야 한다.
- 5~25%는 우선 검토 범위일 뿐 절대적인 합격선은 아니다.

### 4.2 챔피언 오답 복구율

```text
recovery rate
= LR이 틀리고 신규 모델이 맞힌 행 수 / LR이 틀린 전체 행 수
```

신규 모델이 LR의 약점을 얼마나 보완하는지 보여주는 핵심 지표다.

### 4.3 역손실률

```text
reverse loss rate
= LR이 맞고 신규 모델이 틀린 행 수 / LR이 맞힌 전체 행 수
```

복구율은 높고 역손실률은 낮은 모델이 좋은 후보다. 단순 행 개수 외에 반드시
클래스별 결과도 함께 확인한다.

### 4.4 Double-fault rate

```text
double fault = 두 모델이 모두 틀린 행 수 / 전체 행 수
```

이 값이 낮을수록 두 모델의 약점이 덜 겹친다.

### 4.5 Oracle 성능

```text
oracle correct = LR correct OR 신규 모델 correct
```

두 모델 중 정답인 모델을 매번 완벽하게 고른다고 가정한 이론적 상한이다.
실제 제출 점수가 아니며, 조합의 잠재력만 평가한다. Oracle Macro F1이 LR과
거의 같다면 두 모델의 오류가 대부분 겹친다.

### 4.6 확률 상관계수

두 모델의 OOF `행 × 클래스` 확률을 펼쳐 Pearson correlation을 계산한다.

- 0.99 이상: 거의 같은 확률 구조
- 0.95~0.99: 비슷하지만 작은 앙상블 이득 가능
- 0.80~0.95: 유망한 다양성 후보
- 매우 낮음: 신규 모델 자체 성능이 낮거나 확률이 불안정한지 확인 필요

상관계수가 낮다고 무조건 좋은 것은 아니다. 무작위 모델도 상관은 낮다.

### 4.7 클래스별 보완성

전체 평균과 함께 다음을 기록한다.

| 클래스 | LR F1 | 신규 모델 F1 | 차이 | LR 오답 복구 수 |
| --- | ---: | ---: | ---: | ---: |

특히 LR F1이 낮은 희귀 클래스와 주요 혼동 클래스에서 개선되는지 확인한다.

---

## 5. 앙상블 후보 채택 기준

### 5.1 단일 모델 1차 조건

- 3-seed Macro F1이 챔피언 대비 대략 `-0.02` 이내
- leakage audit 통과
- 수렴 및 NaN 문제 없음
- 26개 클래스 확률과 클래스 순서 정상
- 동일 OOF fold 사용

`-0.02`는 자동 탈락선이 아니라 다양성 검토를 위한 우선순위 기준이다.

### 5.2 다양성 2차 조건

- 예측 불일치율 5% 이상
- LR 오답을 의미 있게 복구
- Oracle Macro F1이 LR보다 최소 `+0.01`
- LR이 약한 클래스 중 하나 이상에서 독립적인 개선
- 확률 상관이 지나치게 높지 않음

### 5.3 실제 blend 최종 조건

다양성 지표는 후보 선별용이다. 최종 판단은 OOF probability blend로 한다.

고정 가중치 후보:

```text
LR 95% + 신규 5%
LR 90% + 신규 10%
LR 85% + 신규 15%
LR 80% + 신규 20%
LR 70% + 신규 30%
```

채택 권장 기준:

- 3-seed 평균 Macro F1 `+0.001` 이상
- 3개 seed 중 최소 2개에서 개선
- 특정 seed에서 큰 하락 없음
- 희귀 클래스 F1의 심각한 붕괴 없음
- 선택한 가중치를 test 결과를 보고 바꾸지 않음

같은 OOF에서 많은 가중치를 탐색하면 과적합될 수 있다. 따라서 위의 소수 고정
grid로 screen하고, 승자를 고른 뒤 별도 seed에서 확인하거나 가중치를 lock한다.

---

## 6. 팀 공통 결과표

각 모델 담당자는 다음 표를 채워 공유한다.

| 항목 | 결과 |
| --- | ---: |
| 단일 3-seed Macro F1 | |
| exp13 LR 대비 차이 | |
| 예측 불일치율 | |
| 확률 상관계수 | |
| LR 오답 복구율 | |
| 역손실률 | |
| Double-fault rate | |
| Oracle Macro F1 | |
| 개선된 주요 클래스 | |
| 최고 고정 blend | |
| Blend 3-seed Macro F1 | |
| 개선된 seed 수 | |
| Leakage audit | |

---

## 7. LR 챔피언 + 담당 모델 앙상블 가이드

각 팀원은 담당 모델의 단일 성능을 확인한 뒤, 공통 exp13 LR 챔피언과
**2-model soft voting** 실험을 진행할 수 있다. 이 단계의 목적은 담당 모델이
LR의 오답을 실제로 보완하는지 확인하는 것이다.

### 7.1 필요한 입력

같은 seed와 fold에서 생성된 다음 파일이 필요하다.

```text
exp13 LR OOF probabilities
담당 모델 OOF probabilities
exp13 LR test probabilities
담당 모델 test probabilities
```

두 OOF 파일은 다음이 완전히 같아야 한다.

- 행 수와 ID 순서
- 정답 `SUBCLASS`
- seed와 fold
- 26개 클래스 이름 및 확률 열 순서

하나라도 다르면 확률을 섞기 전에 merge 및 재정렬하고 assertion으로 확인한다.

```python
assert lr_oof["ID"].equals(model_oof["ID"])
assert lr_oof["SUBCLASS"].equals(model_oof["SUBCLASS"])
assert lr_oof["fold"].equals(model_oof["fold"])
assert probability_columns_lr == probability_columns_model
```

### 7.2 확률 정규성 검사

각 행의 확률은 유한하고 합이 1이어야 한다.

```python
assert np.isfinite(lr_probability).all()
assert np.isfinite(model_probability).all()
np.testing.assert_allclose(lr_probability.sum(axis=1), 1.0, atol=1e-6)
np.testing.assert_allclose(model_probability.sum(axis=1), 1.0, atol=1e-6)
```

LinearSVC처럼 원래 확률을 만들지 않는 모델은 fold-train 내부에서 calibration한
확률을 사용한다. validation 전체에 calibration을 fit하면 안 된다.

### 7.3 다양성 보고서를 먼저 만든다

blend 전에 다음을 계산한다.

- 담당 모델 단독 Macro F1
- LR과의 prediction disagreement
- probability correlation
- LR 오답 복구율
- 역손실률
- double-fault rate
- oracle Macro F1
- 클래스별 F1 및 LR 오답 복구 수

단독 점수가 낮고 LR 오답 복구도 거의 없다면 blend grid를 대규모로 탐색하지
않는다. 반대로 단독 점수가 조금 낮아도 희귀 클래스 복구와 Oracle 상승이 크면
앙상블 후보로 유지한다.

### 7.4 고정 weight grid

LR을 기준 모델로 두고 다음 다섯 조합만 먼저 확인한다.

| Case | LR | 담당 모델 |
| --- | ---: | ---: |
| blend_95_05 | 0.95 | 0.05 |
| blend_90_10 | 0.90 | 0.10 |
| blend_85_15 | 0.85 | 0.15 |
| blend_80_20 | 0.80 | 0.20 |
| blend_70_30 | 0.70 | 0.30 |

```python
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)

rows = []
for model_weight in WEIGHTS:
    lr_weight = 1.0 - model_weight
    blended = lr_weight * lr_probability + model_weight * model_probability
    prediction = classes[blended.argmax(axis=1)]
    rows.append(
        {
            "lr_weight": lr_weight,
            "model_weight": model_weight,
            "f1_macro": f1_score(y_true, prediction, average="macro"),
            "accuracy": accuracy_score(y_true, prediction),
        }
    )
```

0.01 간격의 대규모 weight sweep은 같은 OOF에 과적합될 수 있으므로 처음부터
실행하지 않는다.

### 7.5 seed 42 screen과 3-seed confirmation

1. seed 42에서 담당 모델 단독 및 다섯 고정 blend를 계산한다.
2. seed 42에서 LR보다 `+0.001` 이상인 조합만 후보로 남긴다.
3. 최고점 하나만 고르기보다 인접한 weight의 안정성도 확인한다.
4. weight를 잠근 뒤 seeds 42/52/62에 동일하게 적용한다.
5. 3seed 중 최소 2개가 개선되고 평균이 `+0.001` 이상일 때 채택한다.

예를 들어 seed 42에서 `85/15`가 승자라면 seed 52·62에서도 weight를 다시
고르지 않고 그대로 `85/15`를 사용한다.

### 7.6 클래스별 안전성 확인

전체 Macro F1이 상승해도 특정 희귀 클래스가 크게 무너질 수 있다. 최종 후보는
다음을 LR과 비교한다.

```text
클래스별 F1 변화
클래스별 recall 변화
주요 confusion pair의 양방향 오류 수
예측 클래스 수와 클래스별 예측 개수
```

한 클래스의 F1이 크게 하락한 경우 평균 상승만 보고 바로 채택하지 않는다.

### 7.7 test 확률 blend

OOF에서 weight와 모델 구성을 확정한 뒤에만 test 확률을 섞는다.

```python
test_blended = (
    locked_lr_weight * lr_test_probability
    + locked_model_weight * model_test_probability
)
test_prediction = classes[test_blended.argmax(axis=1)]
```

반드시 OOF와 같은 클래스 순서를 사용한다. Public LB 결과를 본 뒤 weight를
바꾸지 않는다.

### 7.8 팀원별 제출물

각 담당자는 다음을 공유한다.

1. 담당 모델 단일 3-seed 결과
2. exp13 LR 대비 다양성 보고서
3. 다섯 fixed blend의 seed 42 결과
4. 잠근 weight의 3-seed 결과
5. 클래스별 F1 비교
6. OOF/test probability 파일 경로와 클래스 순서
7. 최종 채택 또는 기각 판단 및 이유

권장 실험 이름:

```text
<model>-standalone
exp13-lr-plus-<model>-blend
```

예:

```text
lgbm-aggregated-fe
exp13-lr-plus-lgbm-blend
```

---

## 8. 후속 연구 실행 순서

1. exp13 standalone LR을 공통 환경에서 재현한다.
2. 담당 모델에 exp13 전체 안전 피처를 넣어 첫 기준점을 만든다.
3. 모델 특성에 맞는 전처리 표현을 한 축씩 비교한다.
4. seed 42에서 유망한 표현만 남긴다.
5. seeds 42/52/62로 단일 성능과 안정성을 확인한다.
6. 동일 fold의 OOF/test 확률을 저장한다.
7. exp13 LR과 다양성 지표 및 클래스별 보완성을 계산한다.
8. 소수의 고정 가중치 blend를 3-seed OOF로 검증한다.
9. 채택 파이프라인끼리만 최종 앙상블 후보를 구성한다.
10. 제출 전 독립 실행, permutation-label, test 변경 불변성 감사를 수행한다.

---

## 9. 실험 해석 원칙

- 단일 점수 상승과 앙상블 가치는 서로 다른 목표다.
- LR보다 낮은 모델도 LR의 오답을 복구하면 유지한다.
- LR과 거의 같은 예측을 내는 모델은 단일 점수가 높아도 앙상블 가치가 작을 수 있다.
- 모든 모델을 무조건 섞지 않고 OOF에서 보완성이 확인된 모델만 섞는다.
- Public LB를 본 뒤 가중치, 피처, threshold를 재조정하지 않는다.
- 모델 차이와 전처리 차이를 한 실험에서 동시에 여러 개 바꾸지 않는다.

최종 목표는 가장 높은 단일 모델 여러 개를 모으는 것이 아니라, **안전하게
재현되는 서로 다른 강점의 파이프라인을 확보하는 것**이다.
