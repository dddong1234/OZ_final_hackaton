# SDH exp_011 팀 공유 보고서 — B04 고정 class-enrichment FE

## 요약

- 기존 챔피언 **B04 파이프라인과 Logistic Regression을 그대로 고정**하고,
  신규 FE 블록 5종을 각각 독립적으로 추가해 효과를 비교했다.
- 최종 채택 피처는 **gene×event-type class-enrichment score 26개**다.
- 동일 seed 42/52/62 기준 OOF Macro F1은 B04
  `0.47930 ± 0.00253`에서 **`0.52395 ± 0.00202`**로 상승했다.
  절대 개선폭은 **+0.04465**다.
- Public LB는 B04 `0.38711`에서 **`0.43525`**로 상승했다.
  절대 개선폭은 **+0.04814**로, 로컬 CV 개선이 실제 LB에도 전달됐다.
- gene enrichment를 함께 넣은 52개 조합은 `0.52398 ± 0.00282`로 단독 대비
  +0.00003에 불과하고 변동성이 커서 채택하지 않았다.
- 성능 상승은 특히 `KIRC↔KIPAN`, `LGG↔GBMLGG` 혼동 완화에서 컸다.
- supervised 통계는 각 outer fold-train 안에서 다시 내부 5-fold OOF로 만들었다.
  validation/test의 label이나 통계는 피처 학습에 사용하지 않았다.

## 1. 실험 배경과 질문

기존 최고 모델인 B04는 변이 문자열을 gene, burden, event type, truncation,
recurrent missense, 아미노산 치환과 행 내부 구조로 세분화한 고차원 희소
파이프라인이다. 그러나 OOF 오류 분석에서는 다음 네 암종의 상호 혼동이 컸다.

- `KIRC ↔ KIPAN`
- `LGG ↔ GBMLGG`

B04에도 두 혼동쌍을 위한 train-only contrast가 들어 있지만, 각 샘플이 26개
암종의 전반적인 mutation signature와 얼마나 가까운지를 직접 나타내는 표현은
없었다. 이에 다음 질문을 검증했다.

> 모델과 B04를 바꾸지 않고, fold-train에서 학습한 암종별 mutation signature를
> 26개의 압축 점수로 추가하면 Macro F1과 희귀·혼동 클래스 오류가 개선되는가?

모델 탐색과 FE 효과가 섞이지 않도록 모든 case는 B04에서 독립적으로 출발했다.

## 2. 고정 기준: B04 챔피언 파이프라인

실험 기준은 GS 원본 코드의 다음 candidate를 직접 불러와 사용했다.

`H-AS-LR-exact-confusion-pairs-Apair-log1p`

### 2.1 B04의 기본 피처

| 블록 | 의미 | 생성 방법 |
| --- | --- | --- |
| G | gene mutation presence | 해당 유전자에 WT가 아닌 변이가 있으면 1 |
| B | mutation burden 3종 | 변이 유전자 수, 전체 event 수, multi-event 유전자 수 |
| V | event type count 7종 | MISSENSE, SYNONYMOUS, NONSENSE, FRAMESHIFT, SPLICE, INFRAME_INDEL, OTHER |
| T | truncation | NONSENSE/FRAMESHIFT/SPLICE가 발생한 gene flag와 총개수 |
| R | recurrent missense | fold-train에서 5회 이상 나온 gene×exact missense event |
| A-pair | 아미노산 치환 방향 | 20개 아미노산의 ref→alt 조합 380개 count |
| S | 행 내부 구조 8종 | 유전자별 event 개수, type 다양성, entropy, dominant share 등 |

B, V와 A-pair count에는 `log1p`가 적용된다. G, T, R처럼 vocabulary 또는 활성
열이 필요한 블록은 각 fold-train에서만 결정하고 validation에는 적용만 한다.

### 2.2 고정 exact mutation 4종

- `BRAF V600E`
- `IDH1 R132H`
- `PIK3CA H1047R`
- `PIK3CA E545K`

### 2.3 confusion-pair contrast

각 outer fold-train에서 다음 두 쌍의 유전자 변이율 차이를 계산한다.

- `KIRC vs KIPAN`
- `LGG vs GBMLGG`

두 클래스 합산 support가 10 이상인 유전자 중 변이율 차이가 큰 상위 5개를 골라,
변이 개수와 signed contrast score를 생성한다. validation 통계는 선택에 사용하지
않는다.

### 2.4 고정 모델과 검증

- Logistic Regression `solver="lbfgs"`
- `C=0.07`, `max_iter=2000`
- `class_weight="balanced"`
- Stratified 5-fold
- 1차 seed 42
- 최종 확인 seed 42/52/62
- 평가 지표: OOF Macro F1

현재 실행 환경에서 B04 seed 42는 `0.47786`으로 재현됐다. 저장된 과거 값
`0.47814`와 차이는 -0.00028이고, fold별 평균 피처 수 `8,175.2`가 일치해
sklearn/BLAS 수치 차이 범위로 판단했다.

## 3. B04에 독립적으로 추가한 신규 FE

### 3.1 Case 01 — 고정 burden bin 12개

B04에는 이미 세 burden의 `log1p` 값이 있다. LR에 burden의 구간별 비선형성을
직접 제공하기 위해 다음 고정 one-hot bin을 추가했다.

- 변이 유전자 수: `1`, `2`, `3~4`, `5~7`, `8+`
- 전체 event 수: `1`, `2`, `3~4`, `5~7`, `8+`
- multi-event 유전자 수: `1`, `2+`

구간은 사전에 고정돼 다른 train/test 행의 통계를 사용하지 않는다. 0은 모든 bin이
0인 상태로 표현한다.

### 3.2 Case 02 — row-local mutation profile

B04의 원본 count가 같은 샘플 크기 효과를 함께 담는다는 점을 보완하기 위해,
샘플 한 행 안에서만 계산되는 비율을 추가했다.

- `전체 event 수 / 변이 유전자 수`
- `multi-event 유전자 수 / 변이 유전자 수`
- `truncating 유전자 수 / 변이 유전자 수`
- 7개 event type 각각의 `해당 type event 수 / 전체 event 수`

총 10개 후보 중 fold-train에서 상수가 아닌 9개가 실제로 남았다. 이 블록은 label을
전혀 사용하지 않으며, train에서 계산한 평균과 표준편차로만 표준화한다.

### 3.3 Case 03 — gene class-enrichment 26개

token을 단순한 `gene mutation presence`로 정의한다. 각 암종 `c`와 token `t`에
대해 fold-train에서 다음 log-odds 차이를 계산한다.

```text
raw_weight(c,t)
  = log((n_pos + α) / (N_pos - n_pos + α))
  - log((n_neg + α) / (N_neg - n_neg + α))
```

- `n_pos`: 암종 c에서 token t가 등장한 샘플 수
- `N_pos`: 암종 c의 전체 샘플 수
- `n_neg`, `N_neg`: c가 아닌 암종에서의 대응 수
- smoothing `α=1`
- token 최소 support `10`

희귀 token의 과대 가중치를 줄이기 위해 다음 shrinkage를 곱하고 ±4로 제한했다.

```text
weight(c,t) = clip(raw_weight(c,t) × support/(support+20), -4, 4)
```

한 샘플의 암종별 점수는 보유 token 가중치의 합을 활성 token 수의 제곱근으로
나눈 값이다.

```text
score_c(x) = Σ weight(c,t) / sqrt(number of active tokens)
```

26개 암종에 대해 계산하므로 최종적으로 dense 피처 26개가 추가된다.

### 3.4 Case 04 — gene×event-type class-enrichment 26개

최종 채택한 피처다. Case 03과 계산식은 같지만 token 정의를 더 세분화했다.

```text
token = gene + event_type
예: TP53__MISSENSE, APC__NONSENSE, IDH1__MISSENSE
```

동일 유전자 변이라도 missense와 truncating 계열은 생물학적 의미가 다를 수 있다.
gene presence만 사용할 때 섞이는 기능적 차이를 7개 event type 수준에서 분리한다.

전체 데이터의 row-local 파싱에서 관찰된 gene×type vocabulary는 14,469개였지만,
각 fold에서는 fold-train support 10 이상인 token만 가중치 학습에 사용한다. 이
수천 개 token을 모델에 직접 추가하지 않고 암종별 signature score 26개로 압축해,
고차원 희소성 증가를 최소화했다.

### 3.5 Case 05 — exact gene-event class-enrichment 26개

token을 다음처럼 exact mutation 문자열까지 세분화했다.

```text
token = gene + exact event
예: BRAF__V600E, IDH1__R132H
```

계산식과 안전장치는 Case 03/04와 동일하다. 관찰 vocabulary가 226,795개로 매우
크고 대부분 희귀하므로, support 10 필터를 사용해도 fold별 signature 변동과 기존
R/exact 블록과의 중복이 클 수 있다는 가설을 확인했다.

### 3.6 Case 06 — 상위 2개 조합

seed 42에서 가장 높았던 `gene×event-type enrichment` 26개와 `gene enrichment`
26개를 함께 넣어 총 52개 class score를 추가했다. 독립 효과가 확인된 블록의
조합이 추가 상승을 만드는지 확인하기 위한 마지막 ablation이다.

## 4. supervised FE의 누수 방지 구조

Class-enrichment는 label을 사용하는 supervised FE이므로 일반 transform보다 더
엄격하게 cross-fit했다.

### 4.1 OOF 검증 시

1. 전체 train을 outer train/validation으로 분리한다.
2. outer train을 다시 내부 5-fold로 나눈다.
3. 각 inner-train에서 enrichment weight를 학습한다.
4. inner-holdout에 적용해 outer-train용 OOF enrichment를 완성한다.
5. outer train 전체에서 weight를 다시 학습한다.
6. outer validation에는 이 weight를 적용만 한다.
7. outer-train OOF enrichment의 평균·표준편차로 두 행렬을 표준화한다.

따라서 LR 학습 행도 자신의 label이 포함된 weight를 직접 받지 않는다.

### 4.2 최종 제출 시

1. 전체 train 내부 5-fold OOF enrichment로 LR 학습 입력을 만든다.
2. 전체 train label로 최종 enrichment weight를 학습한다.
3. test 각 행에는 최종 weight를 적용만 한다.
4. test label, test 분포, test 평균·빈도는 사용하지 않는다.

train/test를 함께 cache로 파싱하는 부분은 문자열을 행별로 정규화하기 위한
결정론적 처리다. 실제 vocabulary support, class weight, 활성 열, 표준화 통계는
train index에서만 계산한다. test-only token은 train support를 충족할 수 없어
학습 피처나 가중치에 영향을 주지 않는다.

## 5. 실험 결과

### 5.1 seed 42 독립 ablation

| Case | 추가 FE | OOF Macro F1 | B04 대비 | 판정 |
| --- | --- | ---: | ---: | --- |
| 00 | B04 | 0.47786 | 기준 | 기준 |
| 01 | burden bins | 0.47921 | +0.00134 | 소폭 개선 |
| 02 | row profile | 0.47572 | -0.00214 | 기각 |
| 03 | gene enrichment | 0.51157 | +0.03370 | 유망 |
| 04 | gene×event-type enrichment | **0.52640** | **+0.04854** | 3-seed 확인 |
| 05 | exact event enrichment | 0.47534 | -0.00253 | 기각 |

### 5.2 3-seed 확인

| Case | seed 42 | seed 52 | seed 62 | 평균 ± 표준편차 | B04 대비 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B04 | 0.47786 | 0.48286 | 0.47718 | 0.47930 ± 0.00253 | 기준 |
| gene×event-type | **0.52640** | 0.52400 | **0.52145** | 0.52395 ± 0.00202 | +0.04465 |
| gene-type + gene | 0.52433 | **0.52724** | 0.52036 | **0.52398 ± 0.00282** | +0.04468 |

조합은 단독보다 평균이 0.00003 높을 뿐이며 표준편차는 0.00202에서 0.00282로
커졌다. seed 42와 62에서는 오히려 단독이 높다. 따라서 피처 수와 변동성을 고려해
gene×event-type 26개 단독을 최종 채택했다.

모든 실행에서 LR 수렴 경고는 0회였다.

## 6. 클래스별 영향

### 6.1 주요 개선 클래스

| 클래스 | Support | B04 F1 | gene×type F1 | 변화 |
| --- | ---: | ---: | ---: | ---: |
| KIRC | 334 | 0.25913 | 0.63986 | **+0.38073** |
| KIPAN | 515 | 0.23266 | 0.58417 | **+0.35151** |
| GBMLGG | 461 | 0.36778 | 0.50557 | **+0.13779** |
| LGG | 229 | 0.51588 | 0.63469 | **+0.11881** |
| TGCT | 124 | 0.49450 | 0.59915 | +0.10465 |
| SARC | 198 | 0.19942 | 0.25517 | +0.05575 |
| LAML | 158 | 0.49110 | 0.52097 | +0.02986 |

`KIRC/KIPAN/LGG/GBMLGG` 네 클래스의 평균 F1은 `0.34386`에서 `0.59107`로
**+0.24721** 상승했다. B04의 pair contrast보다 26개 암종 전체의 gene×type
signature가 혼동쌍 경계를 더 직접적으로 제공한 것으로 해석한다.

Support 200 미만 14개 클래스 평균 F1도 `0.46182`에서 `0.47373`으로
**+0.01191** 상승했다. 다만 희귀 클래스 전체가 일관되게 개선된 것은 아니다.

### 6.2 하락 클래스

| 클래스 | Support | B04 F1 | gene×type F1 | 변화 |
| --- | ---: | ---: | ---: | ---: |
| LIHC | 158 | 0.45331 | 0.41418 | -0.03913 |
| DLBC | 38 | 0.51983 | 0.48722 | -0.03260 |
| HNSC | 223 | 0.39911 | 0.37636 | -0.02275 |
| LUSC | 178 | 0.55990 | 0.54243 | -0.01747 |

후속 연구에서는 전체 enrichment를 더 복잡하게 만들기보다, 이 하락 클래스의
token support·weight 분산과 주요 오분류 상대를 train-only 기준으로 분석할 필요가
있다.

## 7. Public LB 결과

최종 제출은 `case_04_b04_plus_gene_type_enrichment` 단독, seed 42 전체 train
학습으로 생성했다.

| 파이프라인 | 3-seed CV Macro F1 | Public LB Macro F1 |
| --- | ---: | ---: |
| 기존 B04 | 0.47930 | 0.38711 |
| B04 + gene×event-type enrichment | **0.52395** | **0.43525** |
| 개선폭 | **+0.04465** | **+0.04814** |

CV→LB 차이는 B04 약 -0.09219, exp011 약 -0.08870으로 비슷한 수준이다. 이전
실험처럼 CV 개선이 LB에서 사라지지 않았고, 오히려 개선폭이 거의 동일하게
재현됐다. 따라서 현재 결과는 로컬 과적합만으로 보기보다 실제 test 일반화가
확인된 새 챔피언 후보로 판단한다.

제출 파일명:

`submission_exp011_b04_gene_type_enrichment_seed42.csv`

## 8. 최종 판정과 후속 연구

### 최종 채택

`B04 + gene×event-type class-enrichment 26개`

채택 이유는 다음과 같다.

1. 모든 3개 seed에서 B04보다 크게 높다.
2. 26개 압축 피처만으로 OOF +0.04465를 달성했다.
3. 52개 조합과 평균이 같지만 더 단순하고 안정적이다.
4. 주요 혼동쌍의 클래스별 F1이 크게 개선됐다.
5. Public LB에서도 +0.04814가 재현됐다.

### 기각 또는 보류

- row profile: B04가 이미 count와 S 구조를 충분히 포함해 추가 비율이 중복됨
- exact-event enrichment: vocabulary가 지나치게 희소하고 기존 recurrent/exact와 중복
- gene + gene-type 조합: 추가 평균 이득 없이 변동성과 피처 수만 증가
- burden bins: 소폭 개선했지만 enrichment 대비 효과가 작아 단독 3-seed 미확인

### 권장 후속 연구

1. LIHC, DLBC, HNSC, LUSC의 class score 분포와 주요 오분류 상대 분석
2. enrichment 최소 support와 shrinkage를 대규모 탐색하지 않고 소수 ablation
3. class별 score를 모두 넣는 방식과 하락 클래스 score를 제외하는 방식 비교
4. seed 42 단일 제출과 3개 inner-cross-fit seed 확률 앙상블 비교
5. 제출 전에 permutation-label sanity check로 supervised FE 경로 재감사

모델이나 외부 annotation을 추가하기 전에 현재 26개 gene×type score를 새 고정
기준으로 삼고, 하락 클래스만 독립적으로 보완하는 것이 다음 우선순위다.
