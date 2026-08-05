# SDH exp_018 — hierarchical gene×A-pair / position enrichment

## 한 줄 요약

안전한 exp13 standalone B04+gene-type LR을 고정하고, 세밀한 변이 표현을 수천 개
원시 열로 추가하지 않고 **26개 암종 점수 또는 부모 통계 대비 residual 26개**로
압축해 검증한다.

## 왜 이 실험인가

기존 실험에서 다음 두 축은 각각 강했다.

- A-pair: 정확 위치를 버리고 아미노산 치환 방향을 표현해 처음 보는 변이에도 작동했다.
- gene×mutation-type enrichment: 유전자와 변이 유형의 암종별 관련성을 26개 점수로
  압축해 큰 OOF 상승을 만들었다.

그러나 `IDH1__R>H`처럼 **유전자와 치환 방향을 결합한 supervised enrichment**와,
희귀한 세밀 통계를 안정적인 `IDH1__MISSENSE` 통계로 되돌리는 계층형 backoff는
아직 결과가 없었다. exp18은 이 교차점을 검증한다.

## 누수 방지 계약

초기 ENS-011a의 폐기된 `concat(train, test) → make_context()` 경로를 사용하지 않는다.

각 outer fold에서 다음 순서를 지킨다.

1. fit 원본과 validation 원본을 별도로 파싱한다.
2. exact, gene-type, gene-A-pair, position vocabulary는 fit에서만 만든다.
3. validation/test는 fit vocabulary에 투영만 한다.
4. support, log-odds, backoff 가중치, 표준화 평균·표준편차는 fit에서만 계산한다.
5. label을 쓰는 점수는 fit 내부 5-fold cross-fit으로 생성한다.
6. 사람 손으로 고정한 암종명, 고정 contrast pair, 고정 exact hotspot은 사용하지 않는다.

`hierarchical_enrichment.py`에는 raw train/apply concat 경로가 없고 각 fold audit에
`raw_train_apply_concat=False`, `vocabulary_source=outer_fold_fit_only`를 기록한다.

## 새 토큰

### Gene × A-pair

```text
BRAF V600E   → BRAF__V>E
IDH1 R132H   → IDH1__R>H
PIK3CA E545K → PIK3CA__E>K
```

정확 위치 600, 132, 545는 버리되 어떤 유전자에서 어떤 아미노산 방향이 발생했는지는
보존한다. 부모 토큰은 각각 `BRAF__MISSENSE`, `IDH1__MISSENSE` 등이다.

### Gene × type × position bin

```text
IDH1 R132H → IDH1__MISSENSE__P50_002
TP53 R248Q → TP53__MISSENSE__P50_004
```

기본은 50-aa bin이다. 비교용으로 기존 6개 coarse bin도 한 case에서 확인한다.

## 계층형 residual 계산

fine token의 class log-odds를 `w_fine`, 부모 gene-type의 log-odds를 `w_parent`,
fine support를 `s`, backoff 강도를 `k`라고 하면 추가 가중치는 다음과 같다.

```text
reliability = s / (s + k)
residual_weight = reliability × (w_fine - w_parent)
```

- support가 작으면 residual이 0에 가까워져 부모 통계를 따른다.
- support가 충분하면 fine token만의 차이를 더 많이 반영한다.
- LR에는 부모 점수와 중복된 전체 fine 점수가 아니라 추가 정보만 들어간다.

한 환자의 26개 점수는 보유 fine token의 residual weight를 합하고 활성 토큰 수의
제곱근으로 나눈다. 이후 inner-cross-fit 학습 점수의 평균·표준편차로 표준화한다.

## 17개 case

| 구간 | case | 내용 |
| --- | --- | --- |
| 기준 | e00 | B04, enrichment 없음 |
| 기준 | e01 | 현재 gene-type enrichment baseline |
| A-pair | e02 | gene-A-pair만 사용 |
| A-pair | e03 | gene-type + 독립 A-pair |
| A-pair | e04 | gene-type + A-pair residual |
| 위치 | e05 | 50-aa position만 사용 |
| 위치 | e06 | gene-type + 독립 position |
| 위치 | e07 | gene-type + position residual |
| 위치 | e08 | 6-bin position residual |
| 조합 | e09 | 세 블록을 모두 독립 점수로 사용 |
| 조합 | e10 | A-pair와 position을 모두 residual로 사용 |
| 조합 | e11~e12 | 독립/잔차 혼합 대조 |
| 안정성 | e13~e14 | fine support 5/20 |
| 안정성 | e15~e16 | backoff 5/20 |

모델 파라미터는 모든 FE case에서 `lbfgs, C=0.07, max_iter=2000, balanced`로 고정한다.

## 실행 단계

`experiment.ipynb`를 위에서부터 한 셀씩 실행한다.

1. 데이터와 17개 case 확인
2. seed 42 fold 피처 준비
3. seed 42 전 case LR 스크리닝
4. baseline과 서로 다른 상위 후보 최대 6개 자동 선택
5. seeds 52/62 추가 확인
6. 3-seed 평균·최소 delta 판정
7. 승자에 한해 고정 3-way 모델 구성 확인

### 채택 기준

- 1차 우선순위: seed42 baseline 대비 `+0.003` 이상
- 3-seed 최종: 평균 delta 양수, 모든 seed delta 양수
- 가능하면 평균 `+0.003` 이상을 실질 승자로 본다.
- seed42만 좋은 case는 채택하지 않는다.

3-way 결과는 FE 승자를 모델에 이식할 가치가 있는지 보는 선택 단계다. 가중치는
`multinomial 0.55 + OVR 0.30 + LGBM 0.15`로 고정하며 OOF에서 다시 고르지 않는다.

## 예상 시간

파싱과 nested cross-fit 점수는 fold당 한 번 준비해 17개 case가 공유한다. 가장 오래
걸리는 부분은 LR 재학습이다.

- seed42: 17 cases × 5 folds = 85 LR fits
- 확인: 최대 7 cases × 2 추가 seeds × 5 folds = 최대 70 LR fits
- 선택 3-way: 2 cases × 3 seeds × 5 folds × 3 models = 최대 90 model fits

마지막 3-way는 선택 사항이다. 컴퓨터를 쓰는 중이면 seed42 스크리닝까지만 실행하고,
자는 동안 3-seed와 3-way 셀을 실행하는 편이 좋다.

## 확정 결과

### 신규 FE

17개 case 중 기존 `e01_gene_type_baseline`을 넘은 후보는 없었다.

| case | 3-seed Macro F1 | baseline 대비 | 양수 seed |
| --- | ---: | ---: | ---: |
| e01 gene-type baseline | **0.527609** | 기준 | - |
| e06 gene-type + position50 independent | 0.525972 | -0.001637 | 0/3 |
| e09 all independent | 0.522840 | -0.004769 | 0/3 |
| e03 gene-type + amino independent | 0.522208 | -0.005401 | 0/3 |
| e08 position6 residual | 0.519031 | -0.008578 | 0/3 |
| e04 amino residual | 0.516656 | -0.010952 | 0/3 |

Gene×A-pair는 B04의 raw A-pair 380개와 중복됐고, position score도 독립적인
일반화 이득을 만들지 못했다. 부모 gene-type log-odds를 차감한 residual은 모든
설정에서 더 크게 하락했다. 신규 FE 전부 기각, 기존 gene-type enrichment를 유지한다.

### 부가 모델 다양성 확인

신규 FE 승자가 없어서 baseline 표현에만 기존 고정 3-way를 적용했다.

| 모델 | 3-seed 평균 |
| --- | ---: |
| Multinomial LR | 0.527609 |
| OVR LR | 0.523628 |
| LGBM | 0.485716 |
| 고정 0.55/0.30/0.15 | **0.542392** |

이 수치는 과거 전체 OOF에서 정한 고정 weight를 재사용한 참고값이다. 최종 성능
추정과 weight 결정에는 B10의 outer-fold-local inner 3-fold 선택 결과를 사용한다.
세 seed OOF 확률 평균 `0.554191`도 outer split이 다른 배깅 상한으로만 보며 LB
기대값으로 직접 사용하지 않는다.

수렴 경고는 0건이었고 모든 fold audit에서 raw train/apply 결합 없음, fit-only
vocabulary, 고정 암종명·고정 exact hotspot 없음이 확인됐다.
