# exp-gs-002-08 상세 실험 보고서

## 1. 결론

08은 07의 성능 상승이 **아미노산 치환쌍(A_pair-only) 인코딩** 때문인지, **log1p count 스케일링** 때문인지 분해하고, 최고 구성이 새로운 CV 분할에서도 유지되는지 검증한 실험이다.

- 기존 3-seed에서 최종 구성은 **OOF Macro F1 `0.478502 ± 0.002484`**를 기록했다.
- 독립적인 새 seed `52 / 62 / 31415`에서 **`0.479080 ± 0.002491`**로 같은 수준을 재현했다.
- 새 seed에서 동일한 log1p 조건의 비교군보다 평균 **`+0.006700`** 높았다.
- 새 seed 전 실행에서 누수 없음, test NaN의 mutation 처리 없음, 수렴 경고 0건을 확인했다.

최종 채택 구성은 다음과 같다.

> **H-AS + exact hotspot 4개 + 혼동 암종쌍 contrast + A_pair-only + log1p**

---

## 2. 실험 배경과 질문

07에서 A_pair-only와 log1p를 함께 적용했을 때 큰 성능 상승이 관찰됐다. 다만 두 변경이 동시에 적용되어 다음 질문에 답할 수 없었다.

1. 성능 상승은 A_pair-only 때문인가?
2. 성능 상승은 log1p 때문인가?
3. 두 요소는 각각 효과가 있는가?
4. 점수가 기존 CV seed에만 우연히 맞은 것은 아닌가?

08A는 1~3을 위한 **2×2 분해 실험**이고, 08B는 4를 위한 **새-seed 안정성 검증**이다.

---

## 3. 공통 데이터·모델·안전 조건

| 항목 | 설정 |
|---|---|
| Train / Test | 6,201행 / 2,546행 |
| 유전자 컬럼 | 4,384개 |
| target | 26개 암종 (`SUBCLASS`) |
| 평가 지표 | OOF Macro F1 |
| 모델 | Logistic Regression (`lbfgs`) |
| 파라미터 | `C=0.07`, `max_iter=2000`, `class_weight='balanced'`, model seed 42 |
| CV | Stratified 5-Fold |
| 08A seed | `42 / 2024 / 777` |
| 08B 새 seed | `52 / 62 / 31415` |

### 누수·NaN 규칙

- recurrent missense, 혼동 암종쌍 유전자 선택, non-constant 피처 필터링은 각 fold의 **train split만** 사용한다.
- test는 fit, 통계, feature ranking, 후보 선택에 사용하지 않는다.
- test의 237개 NaN은 mutation event로 파싱하지 않으며, `nan_as_mutation_count=0`을 assert한다.
- 제출 시에는 전체 train에서 선택 규칙을 계산하고 test에는 고정된 규칙을 적용만 한다.

---

## 4. 08에서 고정한 기존 전처리

08은 모든 전처리를 다시 탐색하지 않았다. 아래 구성은 이전 실험에서 유지 근거가 확인되어 고정했고, 08에서는 A 인코딩과 count scale만 변경했다.

### 4.1 홍주님 H-AS backbone

| 블록 | 처리 내용 |
|---|---|
| G | 유전자별 mutation 유무(0/1) |
| B | 변이 유전자 수, 총 이벤트 수, 복수 이벤트 유전자 수 |
| V | missense, nonsense, frameshift 등 변이 유형별 이벤트 수 |
| T | 유전자별 truncating mutation 및 총 truncating 수 |
| R | fold-train에서 5회 이상 발생한 recurrent missense event |
| A | 단백질 아미노산 변화 |
| S | 이벤트 다양성, entropy, dominant share 등 변이 구조 요약 |

### 4.2 Exact hotspot 4개

각 피처는 특정 유전자에 변이가 있다는 일반 피처가 아니라, 특정 아미노산 위치의 정확한 치환이 존재하는지 나타내는 binary 피처다.

- `BRAF V600E`
- `IDH1 R132H`
- `PIK3CA H1047R`
- `PIK3CA E545K`

04 ablation에서 이 중 하나라도 제거하면 점수가 하락해 최종 구성에 모두 유지했다.

### 4.3 혼동 암종쌍 contrast

대상 암종쌍은 `KIRC↔KIPAN`, `LGG↔GBMLGG`다. 각 fold-train에서 다음을 계산한다.

```text
contrast(gene) = mutation_rate(left cancer) - mutation_rate(right cancer)
```

두 암종에서 합산 mutation 발생 수가 10회 이상인 유전자만 후보로 두고, `|contrast|`가 큰 상위 5개를 선택한다. 선택 유전자의 mutation 수 합계와 암종 방향을 반영한 signed contrast score를 피처로 추가한다. 이 블록은 exact hotspot 4개 기준 점수를 `0.433479`에서 `0.438495`로 올렸다.

---

## 5. 08에서 비교한 두 전처리 축

### 5.1 A_all과 A_pair-only

기존 A_all은 총 426개 피처다.

| 구성 | 수 | 의미 |
|---|---:|---|
| ref 아미노산 | 20 | 변이 전 아미노산 종류별 수 |
| alt 아미노산 | 20 | 변이 후 아미노산 종류별 수 |
| ref→alt 치환쌍 | 380 | 치환 방향별 수 |
| 위치 구간 | 6 | 단백질 위치 범위별 수 |
| 합계 | 426 | A_all |

A_pair-only는 `R132H`를 `R→H`로 보고, 서로 다른 아미노산 치환 방향 380개만 남긴다. ref 단독, alt 단독, 위치 구간 46개는 제거한다. 즉 실제 치환 관계에 집중한 인코딩이다.

### 5.2 raw와 log1p

raw는 count를 그대로 사용한다. log1p는 B, V, A의 count형 피처에 아래 변환을 적용한다.

```text
log1p(x) = log(1 + x)
```

큰 변이 count의 영향을 완화하고 Logistic Regression이 더 안정적으로 수렴하도록 만드는 것이 목적이다.

---

## 6. 08A: 2×2 분해

### 설계

| | raw count | log1p count |
|---|---|---|
| A_all | 기존 06 구성 | A_all + log1p |
| A_pair-only | A_pair-only + raw | A_pair-only + log1p |

모든 조합에서 H-AS, exact hotspot 4개, contrast 규칙, 모델 파라미터, 5-Fold CV는 동일하다. 3개 seed의 OOF Macro F1 평균으로 비교했다.

### 결과

| 구성 | 평균 피처 수 | OOF Macro F1 평균 ± 표준편차 | 수렴 경고 합계 | A_all+raw 대비 |
|---|---:|---:|---:|---:|
| A_all + raw | 8,219.53 | 0.438495 ± 0.004021 | 15 | 기준 |
| A_pair-only + raw | 8,173.53 | 0.461318 ± 0.004256 | 15 | +0.022823 |
| A_all + log1p | 8,219.53 | 0.475519 ± 0.000337 | 0 | +0.037024 |
| **A_pair-only + log1p** | **8,173.53** | **0.478502 ± 0.002484** | **0** | **+0.040006** |

### 해석

1. **log1p의 주효과가 가장 컸다.** A_all 기준 `+0.037024` 상승했고 수렴 경고가 15건에서 0건으로 사라졌다.
2. **A_pair-only도 독립 효과가 있었다.** raw 조건에서도 A_all보다 `+0.022823` 높았다.
3. **결합 구성이 최고점이었다.** A_pair-only + log1p가 기준보다 `+0.040006` 높았다.
4. **피처 수는 46개 줄었다.** 불필요한 A 요약보다 치환 방향 자체가 더 유용했음을 시사한다.

---

## 7. 08B: 새-seed 안정성 검증

### 검증 전 고정

08A 결과를 본 뒤 `A_pair-only + log1p`를 승자로 확정하고 winner lock 파일에 기록했다. 이후 08B에서는 새 피처, threshold, 파라미터를 탐색하지 않았다.

비교군은 `A_all + log1p`다. 따라서 log1p는 고정하고 A 전체 표현과 A_pair-only 표현의 차이만 평가한다.

### 새 seed 결과

| Seed | A_all + log1p | A_pair-only + log1p | 차이 |
|---:|---:|---:|---:|
| 52 | 0.472689 | **0.481597** | +0.008908 |
| 62 | 0.472059 | **0.476615** | +0.004556 |
| 31,415 | 0.472391 | **0.479027** | +0.006636 |
| 평균 ± 표준편차 | 0.472380 ± 0.000315 | **0.479080 ± 0.002491** | **+0.006700** |

세 새 seed 모두에서 최종 구성이 더 높았다. 새 평균 `0.479080`은 기존 3-seed 평균 `0.478502`와 같은 수준이므로, 특정 분할에만 맞은 결과로 보기 어렵다.

---

## 8. 혼동 암종쌍 선택 유전자 안정성

각 새 seed와 각 fold의 train split마다 선택 유전자를 저장했다. 암종쌍별 총 선택 기회는 `3 seeds × 5 folds = 15회`다.

### LGG ↔ GBMLGG

| 유전자 | 선택 횟수 | 선택률 | 평균 절대 contrast |
|---|---:|---:|---:|
| IDH1 | 15 / 15 | 100.0% | 0.376045 |
| ATRX | 15 / 15 | 100.0% | 0.147678 |
| TP53 | 15 / 15 | 100.0% | 0.139951 |
| PTEN | 15 / 15 | 100.0% | 0.127752 |
| EGFR | 15 / 15 | 100.0% | 0.110404 |

상위 5개가 모든 fold와 seed에서 동일하게 선택되어 매우 안정적이다.

### KIRC ↔ KIPAN

| 유전자 | 선택 횟수 | 선택률 | 평균 절대 contrast |
|---|---:|---:|---:|
| VHL | 15 / 15 | 100.0% | 0.153067 |
| TP53 | 15 / 15 | 100.0% | 0.031632 |
| MET | 11 / 15 | 73.3% | 0.023665 |
| CDC27 | 9 / 15 | 60.0% | 0.025596 |
| DOCK4 | 5 / 15 | 33.3% | 0.024721 |

이 쌍은 3~5위 후보가 일부 바뀌지만 VHL과 TP53은 모든 경우에 반복 선택됐다. 핵심 신호는 안정적이고 경계권 후보만 split에 따라 교체된다.

---

## 9. 안전성 감사

| 점검 | 결과 | 의미 |
|---|---|---|
| fold-train only | 통과 | validation label이나 test를 피처 선택에 사용하지 않음 |
| `leakage_check` | 새 seed 6개 실행 모두 `True` | 파이프라인 규칙 준수 |
| `nan_as_mutation_count` | 새 seed 6개 실행 모두 `0` | test NaN 237개가 mutation event가 아님 |
| 수렴 경고 | 새 seed 6개 실행 모두 0건 | final log1p 구성의 수렴 안정성 확인 |
| 승자 변경 | 없음 | 08A 이후 고정 구성만 08B에서 재평가 |

---

## 10. 최종 제출 구성

1. H-AS backbone(G, B, V, T, R, A, S)
2. exact hotspot 4개
3. KIRC↔KIPAN 및 LGG↔GBMLGG contrast 피처
4. A_pair-only 380개
5. B/V/A count의 log1p 변환

| 항목 | 최종 설정 |
|---|---|
| 모델 | Logistic Regression (`lbfgs`) |
| 파라미터 | `C=0.07`, `max_iter=2000`, `class_weight='balanced'` |
| full-train seed | 42 |
| 제출 파일 | `experiments/gs/notebooks/submission/submission_exp-gs-002-08_Apair-log1p_seed42.csv` |
| 제출 행/컬럼 | 2,546행, `ID` / `SUBCLASS` |
| full-train feature 수 | 8,399 |
| 수렴 경고 | 0건 |

## 11. 해석 시 주의점

- OOF Macro F1은 leaderboard 점수가 아니라 train 내부 교차검증 추정치다.
- 여러 전처리 후보를 순차적으로 탐색했으므로 새 seed 검증은 선택 편향을 줄이지만 완전히 제거하지는 않는다.
- KIRC↔KIPAN의 일부 경계권 유전자는 fold에 따라 교체된다. 최종 피처는 개별 유전자 고정이 아니라 선택된 유전자의 count/contrast 요약을 사용한다.
- test NaN을 mutation으로 해석하지 않았으며, 그 결측의 생물학적 원인을 별도 모델로 추정하지는 않았다.

## 12. 재현 산출물

| 용도 | 파일 |
|---|---|
| 08 실험 노트북 | `experiments/gs/notebooks/eda_pre_002/exp/exp-gs-002-08.ipynb` |
| 08A 2×2 요약 | `experiments/gs/notebooks/eda_pre_002/result/exp-gs-002-08A_factorial_summary.csv` |
| 08B 선택 빈도 | `experiments/gs/notebooks/eda_pre_002/result/exp-gs-002-08B_confusion_selection_frequency.csv` |
| 최종 단일 실행기 | `experiments/gs/notebooks/submission/exp-gs-002-final_single_run.py` |
