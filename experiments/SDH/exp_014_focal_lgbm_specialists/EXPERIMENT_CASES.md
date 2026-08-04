# exp14 실험별 상세 설명

## 1. 실험 전체 구조

exp14는 세 단계로 진행한다.

```text
exp13 피처 고정
→ 메인 LGBM loss와 class weight 비교
→ 메인 승자에 pair specialist 추가
→ exp13 LR과 확률 blend
```

각 단계에서 여러 요소를 동시에 바꾸지 않는다. 메인 모델을 비교할 때는
specialist와 blend를 사용하지 않고, specialist를 비교할 때는 seed 42에서
선택한 메인 LGBM을 고정한다. LR blend는 LGBM 계열 후보를 확정한 뒤 수행한다.

## 2. 공통 전처리와 검증 조건

모든 메인 모델은 exp13 standalone 챔피언과 같은 피처를 사용한다.

| 블록 | 설명 |
| --- | --- |
| G | 유전자별 변이 존재 여부 |
| B | 변이 유전자 수, 전체 event 수, multi-event 유전자 수 |
| V | missense, nonsense 등 mutation type별 개수 |
| T | 유전자별 truncating mutation과 전체 개수 |
| R | fold-train support 5 이상 recurrent missense |
| A | 아미노산 치환 방향 380개 |
| S | 유전자 내부 mutation topology 8개 |
| Exact | 정확한 mutation 4개 |
| Contrast | 주요 혼동 암종쌍의 train-only 변이율 contrast |
| Enrichment | gene×mutation-type의 클래스별 cross-fit score 26개 |

full-train 기준 피처 수는 8,425개다. fold별로 train에서 활성·비상수 피처를
선택하므로 CV 피처 수는 조금 달라질 수 있다.

공통 검증:

- Stratified 5-fold
- seed 42 screen
- 후보 확정 후 seeds 42/52/62
- 모든 vocabulary, support, contrast, 표준화는 outer fold-train only
- inner enrichment seed: `outer_seed × 100 + fold`
- validation은 학습된 변환을 적용만 함
- 기준 지표: OOF Macro F1
- 보조 지표: Accuracy, 클래스별 F1, LR 대비 다양성

## 3. 공통 메인 LGBM 파라미터

이미지로 전달받은 설정을 최초 기준점으로 고정한다.

```python
boosting_type = "gbdt"
n_estimators = 400
learning_rate = 0.05
num_leaves = 25
reg_alpha = 0.0
reg_lambda = 0.0
min_child_samples = 10
min_child_weight = 1e-3
deterministic = True
force_col_wise = True
```

exp14의 1차 목적은 세부 LGBM 튜닝이 아니라 loss와 class weight 효과를 분리하는
것이다. 따라서 위 파라미터는 모든 메인 case에서 변경하지 않는다.

---

## 4. 메인 LGBM 실험

### main_01_multiclass_balanced

연구 질문:

> 일반 multiclass LGBM에 클래스 빈도 역비례 가중치를 사용하면 exp13 피처에서
> 어느 정도의 기준 성능과 LR 보완성을 얻는가?

설정:

```text
objective = multiclass
class_weight = balanced
```

역할:

- exp14의 표준 LGBM 기준점
- 희귀 클래스를 더 크게 학습
- 기존 ensemble 실험에서 자주 사용된 방식과 비교 가능

주의:

- 희귀 클래스 recall은 좋아져도 다수 클래스 precision이 하락할 수 있다.
- 단일 점수뿐 아니라 LR 오답 복구율을 함께 본다.

### main_02_multiclass_unweighted

연구 질문:

> 클래스 가중치 없이 데이터 원래 분포를 학습하는 것이 balanced보다 안정적인가?

설정:

```text
objective = multiclass
class_weight = None
```

main_01과 다른 것은 class weight뿐이다. balanced가 실제로 도움이 되는지
확인하는 control case다.

가능한 해석:

- main_01 우세: 희귀 클래스 보정이 필요함
- main_02 우세: balanced가 희귀 클래스에 과도하게 맞추거나 확률을 왜곡함
- 단일 점수는 비슷하지만 확률 상관이 낮음: 둘 다 앙상블 후보가 될 수 있음

### main_03_focal_g1

연구 질문:

> 쉬운 샘플의 영향력을 줄이고 어려운 샘플에 집중하는 focal loss가 Macro F1과
> LR 오답 복구를 개선하는가?

설정:

```text
objective = multiclass focal
gamma = 1.0
alpha = 0.25
class_weight = None
```

원리:

일반 cross-entropy는 이미 잘 맞히는 샘플도 계속 학습한다. focal loss는 정답
확률이 높은 쉬운 샘플의 gradient를 줄여 경계 샘플과 반복 오답에 더 집중한다.

`alpha=0.25`는 모든 클래스에 같은 상수로 적용되므로 희귀 클래스를 직접 구분해
가중하는 값은 아니다. 핵심 실험 변수는 `gamma=1`의 난이도 집중 효과다.

구현 참고:

- gradient: analytical focal gradient
- Hessian: LightGBM 안정성을 위한 positive diagonal approximation
- custom objective의 예측은 raw logit이므로 softmax를 직접 적용

### main_04_focal_g2

연구 질문:

> focal의 어려운 샘플 집중 강도를 높이면 추가 개선되는가, 아니면 어려운 오답과
> 노이즈에 과도하게 맞추는가?

설정:

```text
objective = multiclass focal
gamma = 2.0
alpha = 0.25
class_weight = None
```

main_03과 다른 것은 gamma뿐이다.

- gamma 1: 비교적 완만한 쉬운 샘플 억제
- gamma 2: 쉬운 샘플을 더 강하게 억제

main_04가 main_03보다 낮다면 이 데이터에서는 gamma 2가 너무 공격적이라는
뜻이다. 특히 희귀 클래스의 label noise를 따라가는지 클래스별 F1을 확인한다.

### main_05_focal_g1_balanced

연구 질문:

> focal의 어려운 샘플 집중과 balanced class weight를 함께 사용하면 상호보완되는가,
> 아니면 희귀·어려운 샘플을 이중으로 과대평가하는가?

설정:

```text
objective = multiclass focal
gamma = 1.0
alpha = 0.25
class_weight = balanced
```

main_03과 비교해 class weight 결합 효과만 본다. 이 case는 성능 향상 가능성도
있지만 가장 과보정될 위험이 큰 case다.

판정:

- main_05 > main_03: class imbalance와 sample difficulty 보정이 상호보완
- main_05 < main_03: 희귀·어려운 샘플에 이중 가중되어 불안정
- Macro F1 상승 시에도 특정 클래스 precision 붕괴 여부 확인

---

## 5. 메인 모델 선택 방법

메인 5개 case를 모두 실행한 뒤 seed 42 OOF Macro F1이 가장 높은 LGBM을
`SELECTED_MAIN`으로 자동 선택한다.

```python
SELECTED_MAIN = max(
    main_cases,
    key=lambda name: results[name].summary["oof_f1_macro"],
)
```

자동 선택 결과를 그대로 사용해도 되지만 다음 조건을 함께 확인한다.

- fold 한 개의 큰 상승에만 의존하지 않는가
- 수렴 및 확률 이상이 없는가
- 26개 클래스를 모두 예측하는가
- 최고점과 차이가 매우 작은 case가 더 단순하거나 안정적이지 않은가

점수가 사실상 같은 경우 더 단순한 `multiclass` 또는 `focal γ=1`을 우선한다.

---

## 6. Pair specialist 공통 원리

메인 LGBM을 학습한 뒤 다음 두 혼동쌍을 위한 binary LGBM을 별도로 학습한다.

```text
KIRC ↔ KIPAN
LGG ↔ GBMLGG
```

각 outer fold에서 specialist는 fold-train 중 해당 두 클래스의 행만 사용한다.
validation label은 학습이나 routing 결정에 사용하지 않는다.

### KIRC/KIPAN specialist

```python
n_estimators = 10
learning_rate = 0.10
num_leaves = 20
min_child_samples = 20
```

### LGG/GBMLGG specialist

```python
n_estimators = 100
learning_rate = 0.02
num_leaves = 20
min_child_samples = 10
```

specialist는 새로운 클래스를 추가하지 않는다. 메인 모델이 두 클래스에 할당한
확률의 합을 유지하면서 두 클래스 내부 비율만 조정한다.

---

## 7. Specialist 실험

### spec_01_k_soft_mass_030

연구 질문:

> KIRC/KIPAN specialist 하나만 사용했을 때 해당 혼동쌍과 전체 Macro F1이
> 개선되는가?

설정:

```text
pair = KIRC/KIPAN
mode = soft mass
alpha = 0.30
```

쌍별 단독 효과를 분리하기 위한 case다. pair probability mass가 클수록 specialist
보정이 강하게 작동한다.

### spec_02_l_soft_mass_030

연구 질문:

> LGG/GBMLGG specialist 하나만 사용했을 때 해당 혼동쌍과 전체 Macro F1이
> 개선되는가?

설정:

```text
pair = LGG/GBMLGG
mode = soft mass
alpha = 0.30
```

spec_01과 함께 두 specialist 중 실제 기여하는 쌍을 구분한다.

### spec_03_both_soft_mass_015

연구 질문:

> 두 specialist를 약하게 함께 적용하면 메인 확률을 훼손하지 않고 안정적인
> 보정이 가능한가?

```text
pairs = both
mode = soft mass
alpha = 0.15
```

가장 보수적인 두 쌍 동시 보정이다.

### spec_04_both_soft_mass_030

연구 질문:

> 전달받은 보정 강도 0.30에서 두 specialist의 효과가 합쳐지는가?

```text
pairs = both
mode = soft mass
alpha = 0.30
```

두 specialist soft correction의 중심 기준이다.

### spec_05_both_soft_mass_050

연구 질문:

> specialist의 영향력을 더 키우면 혼동쌍 개선이 커지는가, 아니면 메인 모델의
> 올바른 확률을 지나치게 뒤집는가?

```text
pairs = both
mode = soft mass
alpha = 0.50
```

0.15/0.30/0.50을 통해 보정 강도의 방향성을 확인한다. 0.50만 우연히 높고 인접
강도에서 불안정하다면 채택에 주의한다.

### spec_06_both_soft_predicted_030

연구 질문:

> 메인 argmax가 네 대상 클래스에 속한 행만 제한적으로 soft correction하면
> 전체 행을 보정하는 soft-mass보다 안전한가?

```text
pairs = both
mode = soft predicted-only
alpha = 0.30
```

메인 예측이 pair 밖인 행은 전혀 수정하지 않는다. 적용 범위가 좁어 변경 행 수는
감소하지만, pair가 2순위인 경계 샘플을 복구하지 못할 수 있다.

### spec_07_both_hard_predicted

연구 질문:

> 이미지에서 제안한 것처럼 메인 argmax가 pair에 속한 샘플을 specialist 결과로
> 완전히 다시 분류하면 성능이 개선되는가?

```text
pairs = both
mode = hard predicted-only
alpha = 1.00
```

가장 공격적인 보정이며 control 성격이 강하다. specialist가 메인 모델의 정답까지
뒤집을 수 있으므로 changed rows, 양방향 confusion과 pair class F1을 반드시 본다.

---

## 8. Soft-mass 계산 원리

메인 모델의 두 클래스 확률 합을 다음처럼 둔다.

```text
pair_mass = p(left) + p(right)
```

메인 모델의 pair 내부 left 비율과 specialist의 left 확률을 계산한다.

```text
main_ratio = p(left) / pair_mass
specialist_ratio = specialist_p(left)
```

soft-mass에서는 보정 가중치가 `alpha × pair_mass`다.

```text
weight = alpha × pair_mass
new_ratio = (1-weight) × main_ratio + weight × specialist_ratio
```

pair 가능성이 낮은 행은 거의 수정하지 않고, 두 클래스 가능성이 높은 행만 더
강하게 수정한다. 최종적으로 pair 확률 합은 원래 값 그대로 유지된다.

---

## 9. LR 대비 다양성 평가

메인 및 specialist 모든 후보를 exp13 LR과 비교한다.

| 지표 | 의미 |
| --- | --- |
| Disagreement | LR과 다른 label을 예측한 비율 |
| Probability correlation | 26개 OOF 확률의 유사도 |
| Recovery rate | LR 오답 중 LGBM이 복구한 비율 |
| Reverse loss rate | LR 정답을 LGBM은 틀린 비율 |
| Double fault | 두 모델이 동시에 틀린 비율 |
| Oracle Macro F1 | 둘 중 하나라도 맞는다고 가정한 잠재 상한 |

단일 LGBM이 LR보다 낮더라도 recovery와 oracle이 충분히 높으면 blend 후보로
유지한다.

---

## 10. LR + LGBM blend 실험

seed 42에서 선택된 최고 LGBM 계열 후보를 LR과 섞는다.

| Case | LR | LGBM 계열 |
| --- | ---: | ---: |
| blend_95_05 | 0.95 | 0.05 |
| blend_90_10 | 0.90 | 0.10 |
| blend_85_15 | 0.85 | 0.15 |
| blend_80_20 | 0.80 | 0.20 |
| blend_70_30 | 0.70 | 0.30 |

확률을 가중 평균한 뒤 argmax로 예측한다. seed 42에서 선택한 weight는 이후
seed마다 다시 고르지 않고 `LOCKED_MODEL_WEIGHT`로 고정한다.

---

## 11. 3-seed confirmation

seed 42에서 다음 세 값을 잠근다.

```text
CONFIRM_MAIN_CASE
CONFIRM_SPECIALIST_CASE
LOCKED_MODEL_WEIGHT
```

그 상태로 seeds 42/52/62를 실행한다.

채택 권장 기준:

- LR blend 평균 delta `+0.001` 이상
- 3seed 중 최소 2개에서 LR보다 개선
- 특정 seed에서 큰 하락 없음
- 주요 희귀 클래스 F1의 심각한 붕괴 없음
- 수렴·확률·누수 검사 통과

단일 LGBM 채택 기준과 LR blend 채택 기준은 구분한다. 단일 점수가 LR보다 낮아도
blend가 안정적으로 상승하면 모델 다양성 확보에 성공한 것으로 판단한다.

---

## 12. 결과 파일

노트북은 다음 파일을 `results/`에 저장한다.

```text
main_seed42_leaderboard.csv
specialist_seed42_leaderboard.csv
diversity_vs_exp13_lr_seed42.csv
lr_lgbm_blend_seed42.csv
class_metrics_seed42.csv
oof_exp13_lr_seed42.csv
oof_<selected candidate>_seed42.csv
confirmation_3seed.csv
```

확률 및 결과 파일은 Git에 커밋하지 않는다. 최종 문서에는 요약 지표와 재현
설정만 기록한다.
