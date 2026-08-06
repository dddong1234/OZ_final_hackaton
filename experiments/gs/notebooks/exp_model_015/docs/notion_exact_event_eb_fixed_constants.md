# Exact-event EB 3-seed — 고정 상수 및 자동 학습 규칙

## 최종 결과

| 항목 | 값 |
| --- | ---: |
| 3-seed OOF Macro F1 | **0.568441 ± 0.002310** |
| H0 대비 평균 OOF 변화 | **+0.021186** |
| Public LB | **0.5086** |
| 제출 구성 | Exact-event EB H0, seeds 42/777/2024 동등 평균 |

이 문서는 코드에 고정한 **일반 규칙과 모델 상수**를 명시한다. 특정 암종명, 유전자명, exact mutation/hotspot 목록을 입력 규칙으로 고정하지 않았다.

---

## 1. 고정한 검증·bagging 상수

| 구분 | 상수 | 값 | 의미 |
| --- | --- | --- | --- |
| 외부 CV | outer folds | 5 | Stratified 5-fold OOF |
| 내부 cross-fit | inner folds | 5 | supervised EB train feature 생성 |
| 검증/제출 seeds | `VALIDATED_SEEDS` | `(42, 777, 2024)` | 사전 고정한 3개 seed |
| 제출 평균 | seed weight | `1/3, 1/3, 1/3` | 세 모델 test 확률 단순 평균 |
| seed42 screen 통과 기준 | OOF delta | `+0.015` | 3-seed 확장 전 screen 기준 |
| 3-seed 채택 평균 변화 | mean delta | `≥ +0.010` | H0 대비 평균 향상 |
| 3-seed 채택 최소 변화 | min delta | `≥ +0.005` | 모든 seed에서의 최소 향상 |
| 3-seed 채택 fold | positive fold | `≥ 11/15` | fold 안정성 기준 |
| 클래스 안전선 | mean F1 delta | `≥ -0.05` | 특정 클래스 붕괴 방지 |

이번 결과는 모든 seed가 양수이고 `15/15` fold가 상승했으며, 최소 seed delta는 `+0.018750`이었다.

---

## 2. Mutation 문자열 파싱의 고정 규칙

이는 외부 생물학 지식이나 특정 mutation 목록이 아니라, 제공된 cell 문자열을 일관되게 읽기 위한 일반 문법이다.

| 항목 | 고정 규칙 |
| --- | --- |
| WT | `WT`는 event 0개 |
| 빈 문자열 | event 0개 |
| NaN | event 0개. 문자열 `"nan"`으로 변환하지 않음 |
| 접두사 | `p.`는 제거 후 표준화 |
| 구분자 | 공백, `;`, `,`, `|`를 복수 event 분리자로 처리 |
| 중복 | 같은 환자·유전자·event는 1회만 유지 |
| event type | `MISSENSE`, `SYNONYMOUS`, `NONSENSE`, `FRAMESHIFT`, `SPLICE`, `INFRAME_INDEL`, `OTHER` |
| truncating type | `NONSENSE`, `FRAMESHIFT`, `SPLICE` |
| A-pair | 표준 20개 아미노산의 서로 다른 ref→alt 조합 380개 |

`nan_as_mutation_count=0`을 각 seed 결과에 기록하며, 3-seed 검증에서 모두 0이었다.

---

## 3. 구조화 H0 입력의 고정 상수

| 블록 | 상수 | 값 | 의미 |
| --- | --- | --- | --- |
| recurrent missense | `RECURRENT_MIN_COUNT` | `5` | outer-fold train에서 5회 이상 관측된 missense exact event만 recurrent block에 포함 |
| gene×type enrichment | 최소 support | `10` | 구조화 H0의 기존 enrichment vocabulary 필터 |
| gene×type enrichment | alpha | `1.0` | 발생률 smoothing |
| gene×type enrichment | shrinkage | `10.0` | support 기반 수축 |
| gene×type enrichment | clip | `[-4, 4]` | log-odds 과대값 제한 |
| burden/A-pair/event count | 변환 | `log1p` | 큰 count의 영향 완화 |

이 블록의 gene 목록, recurrent event 목록, enrichment vocabulary는 **각 outer-fold train에서 자동 생성**된다. test 또는 validation에서 선택하지 않는다.

---

## 4. Empirical-Bayes(EB) 상수

Exact-event EB와 Selective-EB에서 공통으로 쓰는 posterior 수축 상수다.

| 상수 | 값 | 역할 |
| --- | ---: | --- |
| `EB_ALPHA` | `1.0` | 전체 event 발생률 prior smoothing |
| `EB_SHRINKAGE` | `20.0` | 희귀 event의 암종별 발생률을 전역 prior 쪽으로 수축 |
| `EB_CLIP` | `4.0` | class log-odds를 `[-4, 4]` 범위로 제한 |
| Exact-event vocabulary support | `>0` 그리고 `< fold-train 행 수` | fold-train에서 실제 관측되고 상수열이 아닌 모든 exact event 사용 |
| Exact-event support cutoff | 없음 | 특정 빈도 이상 event만 사람이 선택하지 않음 |
| Exact EB score 정규화 | `sqrt(active exact event count)` | event가 많은 샘플의 evidence 합이 과도해지는 것을 완화 |

암종별 exact-event score는 다음 개념으로 계산한다.

```text
exact-event class evidence
= logit(P(event | class, posterior))
  - logit(P(event | not-class, posterior))

sample class score
= 활성 exact-event evidence 합 / sqrt(활성 exact-event 수)
```

각 암종의 score는 inner OOF train feature의 mean/std로 표준화한다. validation/test 평균·표준편차를 사용하지 않는다.

---

## 5. Logistic Regression과 Selective gate

| 항목 | 고정값 |
| --- | --- |
| solver | `lbfgs` |
| C | `0.07` |
| max_iter | `2000` |
| class weight | `balanced` |
| model random state | 현재 seed/fold seed |
| gate margin | `0.05` |

Gate 규칙:

```text
exact-event EB LR의 top-1 probability − top-2 probability < 0.05
    → non-EB LR probability 사용
그 외
    → exact-event EB LR probability 사용
```

`0.05`는 Exact-event EB 실험에서 재탐색하지 않고 기존 Selective-EB 계약을 유지한 값이다.

---

## 6. Automatic LGBM specialist와 최종 결합

| 항목 | 고정값 |
| --- | --- |
| main multiclass LGBM estimators | `400` |
| main learning rate | `0.05` |
| main num_leaves | `25` |
| main min_child_samples | `10` |
| specialist binary LGBM estimators | `100` |
| specialist learning rate | `0.02` |
| specialist num_leaves | `20` |
| specialist min_child_samples | `10` |
| LGBM class weight | `balanced` |
| LR branch weight | `0.80` |
| specialist branch weight | `0.20` |

Specialist 암종쌍은 고정 목록이 아니다. 각 fold-train의 mutation-binary class centroid cosine similarity를 계산해 가장 유사한 두 암종쌍을 자동 선택한다. Specialist는 선택된 쌍 안에서만 확률 비율을 조정하고, 해당 쌍의 전체 확률 질량은 유지한다.

최종 확률:

```text
0.80 × selective exact-event EB LR probability
+ 0.20 × automatic specialist LGBM probability
```

---

## 7. 자동 학습되는 값과 고정 상수의 구분

| 고정 상수 | fold-train 자동 학습 |
| --- | --- |
| parser 문법, event taxonomy | exact-event vocabulary |
| `EB_ALPHA=1`, shrinkage=20, clip=4 | event별 support와 global prior |
| LR/LGBM hyperparameter | 암종별 posterior log-odds |
| selective margin=0.05, 0.8/0.2 blend | recurrent missense event 목록 |
| 3-seed와 동등 평균 | gene×type vocabulary 및 enrichment weight |
| generic A-pair 380 정의 | automatic specialist 암종쌍 |

따라서 `BRAF V600E`와 같은 특정 mutation, `KIRC-KIPAN`과 같은 특정 암종쌍은 코드 상수나 규칙이 아니다. 해당 정보가 사용되더라도 현재 fold-train에서 관찰·계산되어 자동 선택된 결과일 뿐이다.

---

## 8. 규정 안전성

- outer OOF에서는 `test.csv`를 읽지 않음
- 제출 때 test는 full-train에서 이미 fit된 변환을 적용하고 예측하는 용도로만 사용
- train/test concat 없음
- test 기반 vocabulary, one-hot, scaling, feature selection, 결측 통계 없음
- `leakage_check=True`, `nan_as_mutation_count=0`, convergence warning `0`

## 관련 파일

- 3-seed 검증 runner: `common/run_exact_event_eb_3seed_validation.py`
- seed42 screen runner: `common/run_exact_event_eb_screen.py`
- 3-seed 제출 runner: `../../submission/generate_submission_h0_exact_event_eb_3seed.py`
- 3-seed 결과: `result/exp-exact-event-eb-01_3seed_aggregate.csv`
