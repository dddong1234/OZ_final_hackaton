# exp14 전처리·피처 엔지니어링 발표 가이드

## 1. 한 줄 요약

exp14의 전처리는 유전자별 mutation 문자열을 단순히 펼치는 것이 아니라,
한 환자의 변이를 **유전자 존재 여부, 전체 변이량, 변이 종류, 반복 위치,
아미노산 치환, 유전자 내부 구조, 암종별 train 연관성**이라는 여러 숫자 관점으로
바꾸는 과정이다.

```text
원본 문자열
TP53="R175H R248H", APC="Q1367*", KRAS="WT"

                         ↓

TP53 변이 있음                 1
전체 변이 event 수             3
missense 수                    2
nonsense 수                    1
truncating 유전자 수           1
R→H 아미노산 치환 수           2
한 유전자의 최대 mutation 수   2
암종별 train 연관 점수          26개
```

겉보기에는 피처 수가 많지만, 핵심은 하나의 mutation 기록을 여러 방식으로
요약해 모델이 비교할 수 있는 숫자로 바꾸는 것이다.

---

## 2. 왜 이런 전처리가 필요한가

원본 데이터는 각 유전자가 열이고 셀 안에는 `R175H`, `Q1367*`, `WT` 같은
문자열이 들어 있다. 머신러닝 모델은 문자열 자체의 생물학적 의미를 알지 못한다.

예를 들어 모델에 아래 두 값을 그대로 전달하면:

```text
R175H
Q1367*
```

모델은 `R175H`가 아미노산 치환이고 `Q1367*`가 단백질을 중간에 끝낼 수 있는
변이라는 사실을 자동으로 이해하지 못한다. 따라서 문자열에서 규칙적으로
읽을 수 있는 정보를 숫자로 분해한다.

이 실험의 출발점은 다음 두 가지였다.

1. **데이터 구조에서 출발한 가설**: 한 셀에 `WT`, 단일 mutation, 복수 mutation이
   섞여 있으므로 존재 여부만 쓰면 많은 정보가 사라진다.
2. **모델 특성에서 출발한 가설**: LGBM은 단순 유전자 존재 여부뿐 아니라 burden,
   mutation type, topology 같은 숫자 피처 사이의 비선형 조건을 학습할 수 있다.

예를 들어 LGBM은 다음과 같은 조합을 tree 분기로 표현할 수 있다.

```text
TP53 변이가 있고
AND truncating 변이 수가 2개 이상이고
AND 전체 mutation burden이 높은 경우
```

LR은 각 피처의 독립적인 가중합에 강하고, LGBM은 이런 조건 조합에 강하므로
두 모델의 예측 오류가 달라질 가능성도 기대했다.

---

## 3. 전처리 전체 흐름

```text
유전자별 mutation 문자열
        ↓
문자열 정규화와 token 분리
        ↓
행 단위 피처 생성
  G: mutation gene 존재
  B: mutation burden
  V: mutation type count
  T: truncating mutation
  A: amino-acid substitution
  S: mutation topology
        ↓
fold-train 통계 피처 생성
  R: recurrent exact missense
  E: class enrichment
        ↓
고정 외부지식 피처 제거
  C__: 제거
  D__exact: 제거
        ↓
희소 숫자 행렬
        ↓
LR / multiclass LGBM / focal LGBM
```

---

## 4. 예시 환자 한 명을 변환해 보기

다음과 같은 환자를 가정한다.

| 유전자 | 원본 값 |
| --- | --- |
| TP53 | `R175H R248H` |
| BRAF | `V600E` |
| APC | `Q1367*` |
| KRAS | `WT` |

### 4.1 문자열 정규화

```text
" r175h "       → ["R175H"]
"V600E,G469A"   → ["V600E", "G469A"]
"WT"            → []
NaN              → []
```

공백을 제거하고 대문자로 통일하며 쉼표·세미콜론·`|` 등을 mutation 구분자로
처리한다. `WT`, 빈 문자열, 결측치는 기록된 mutation이 없는 것으로 처리한다.

### 4.2 변환 결과 요약

| 관점 | 생성되는 정보 | 예시 값 |
| --- | --- | ---: |
| 유전자 존재 | TP53에 mutation이 있는가 | 1 |
| 유전자 존재 | KRAS에 mutation이 있는가 | 0 |
| burden | mutation 유전자 수 | 3 |
| burden | 전체 event 수 | 4 |
| burden | 복수 event 유전자 수 | 1 |
| mutation type | missense 수 | 3 |
| mutation type | nonsense 수 | 1 |
| truncation | truncating 유전자 수 | 1 |
| amino pair | R→H 치환 수 | 2 |
| topology | mutation 2개인 유전자 수 | 1 |
| topology | 한 유전자의 최대 event 수 | 2 |

---

## 5. 피처 블록별 설명과 실험 근거

## 5.1 G — Gene mutation presence

유전자에 mutation이 하나라도 있으면 1, 없으면 0이다.

```text
TP53="R175H" → G__TP53=1
TP53="WT"    → G__TP53=0
```

### 실험 근거

가장 기본적인 정보는 “어떤 유전자가 변했는가”이다. mutation 문자열을 지나치게
세분화하면 같은 유전자의 서로 다른 희귀 mutation이 모두 별도 열로 갈라질 수 있다.
G 피처는 exact 위치가 달라도 같은 유전자 수준의 공통 신호를 보존한다.

### 모델이 얻는 정보

```text
특정 유전자의 mutation 존재 여부
여러 유전자의 동시 mutation 조합
```

---

## 5.2 B — Mutation burden

한 환자가 mutation을 얼마나 많이 가지고 있는지 요약한다.

```text
B__mutated_gene_count
B__event_count
B__multi_event_gene_count
```

예시:

```text
TP53="R175H R248H"
BRAF="V600E"
APC="Q1367*"

mutation 유전자 수       = 3
전체 event 수            = 4
복수 event 유전자 수     = 1
```

큰 count가 지나치게 강한 영향을 주지 않도록 일부 값에는 `log1p`를 적용한다.

\[
x' = \log(1+x)
\]

### 실험 근거

원본 데이터에서 행마다 mutation이 기록된 열의 수와 한 셀 안 token 수가 다르다.
유전자별 이진 피처만 사용하면 “mutation이 전반적으로 많은 환자”와 “일부
유전자에만 mutation이 있는 환자”의 전체적인 차이를 모델이 매번 수천 개 열에서
다시 계산해야 한다. burden은 그 차이를 직접 제공하는 행 단위 요약이다.

---

## 5.3 V — Mutation type counts

mutation 표기 규칙을 이용해 다음 유형으로 분류한다.

```text
MISSENSE
SYNONYMOUS
NONSENSE
FRAMESHIFT
SPLICE
INFRAME_INDEL
OTHER
```

예시:

```text
R175H   → MISSENSE
Q1367*  → NONSENSE
R97FS   → FRAMESHIFT
SPLICE  → SPLICE
```

환자별로 각 type의 개수를 센다.

```text
V__missense_event_count = 3
V__nonsense_event_count = 1
```

### 실험 근거

같은 mutation 개수라도 구성은 다를 수 있다.

```text
환자 A: missense 5개
환자 B: missense 2개 + nonsense 1개 + frameshift 1개 + splice 1개
```

두 환자의 총 event 수는 같지만 mutation의 형태가 다르다. V 피처는 유전자
이름과 별개로 mutation 구성 차이를 모델에 전달한다.

---

## 5.4 T — Truncating mutation

다음 유형을 truncating 계열로 묶는다.

```text
NONSENSE
FRAMESHIFT
SPLICE
```

예시:

```text
APC="R1450H" → G__APC=1, T__APC=0
APC="Q1367*" → G__APC=1, T__APC=1
```

### 실험 근거

G 피처는 유전자에 mutation이 있다는 사실만 표현한다. 같은 유전자 mutation도
표기 유형이 다르므로 mutation의 존재와 mutation의 형태를 분리해 전달할 필요가
있다. T 피처는 외부 유전자 목록을 사용하지 않고 현재 행의 mutation 표기만으로
계산하는 결정론적 피처이다.

---

## 5.5 R — Fold-train recurrent exact missense

같은 유전자와 같은 exact missense token이 fold-train에서 반복되면 피처로 만든다.

```text
fold-train에서 GENE_X__A100B가 5회 등장
→ R__GENE_X__A100B 생성
```

환자가 해당 mutation을 가지면 1, 아니면 0이다.

### 실험 근거

G 피처만으로는 같은 유전자 안의 서로 다른 위치를 구분하지 못한다. 반대로 모든
exact token을 피처로 만들면 한 번만 등장한 희귀 열이 폭발한다. 따라서 fold-train
support가 일정 수준 이상인 exact missense만 선택해 두 극단 사이를 절충한다.

### 규정 안전성

- exact mutation 이름을 외부에서 고정하지 않는다.
- 각 outer-fold train에서 support를 다시 계산한다.
- validation에만 등장한 mutation은 선택되지 않는다.
- 현재 최소 support는 5다.

---

## 5.6 A — Amino-acid substitution direction

missense token에서 원래 아미노산과 변경 아미노산을 추출한다.

```text
R175H → R→H
R248H → R→H
V600E → V→E
```

예시 환자에서는 다음 값이 생긴다.

```text
A_pair__R_to_H = 2
A_pair__V_to_E = 1
```

20개 아미노산에서 자기 자신으로 바뀌는 경우를 제외하므로 최대 380개 방향이
존재한다.

\[
20 \times 19 = 380
\]

### 실험 근거

exact 위치가 서로 달라도 같은 치환 방향을 공유할 수 있다. exact token은 너무
희소하고 mutation type count는 너무 넓은 요약이므로, amino-pair는 두 표현의
중간 수준에 해당한다.

---

## 5.7 S — Mutation topology

mutation이 환자 내부에서 어떻게 분포하는지를 표현한다.

```text
mutation이 1개인 유전자 수
mutation이 2개인 유전자 수
mutation이 3개 이상인 유전자 수
여러 mutation type이 섞인 유전자 수
한 유전자의 최대 mutation 수
mutation type 종류 수
mutation type entropy
가장 많은 mutation type의 비율
```

예:

```text
환자 A: TP53=R175H, BRAF=V600E, APC=Q1367*
환자 B: TP53=R175H R248Q R273H
```

두 환자는 event가 3개로 같지만 A는 여러 유전자에 흩어지고 B는 한 유전자에
집중된다. topology는 이 차이를 표현한다.

### 실험 근거

EDA에서 볼 수 있는 한 셀의 복수 token과 행별 mutation 분포 차이를 직접
요약하려는 피처다. 모든 계산은 한 환자 행 내부에서 끝나므로 다른 환자나 test
통계를 사용하지 않는다.

---

## 5.8 E — Cross-fitted class enrichment

fold-train에서 특정 `유전자×mutation type` 조합이 어떤 암종에 상대적으로 자주
나타나는지 점수화한다.

예:

```text
암종 A의 TP53__MISSENSE 보유율 = 70%
나머지 암종의 보유율           = 10%
```

이 경우 TP53 missense를 가진 환자의 암종 A score가 올라간다.

희귀 패턴이 과도하게 커지지 않도록 support shrinkage를 적용한다.

\[
w'_{c,j}
=
w_{c,j}
\frac{support_j}{support_j+\lambda}
\]

### 실험 근거

희소한 유전자·mutation type 피처 수천 개를 모델이 직접 조합하는 대신, fold-train
안에서 관측된 클래스별 방향을 26개 class score로 압축한다. 특히 희귀 클래스의
증거를 별도 점수로 보존하려는 목적이 있다.

### 규정 안전성

label을 사용하는 supervised 피처이므로 outer-fold train 내부에서 다시
cross-fitting한다.

```text
outer-fold train
  → inner train으로 enrichment weight 계산
  → inner validation 행에 적용

outer validation
  → outer-fold train 전체로 계산한 weight를 적용만 함
```

자기 label로 자기 score를 직접 만드는 구조가 아니다.

---

## 6. 제거한 피처와 이유

현재 exp14 안전 버전은 다음 피처를 모델 입력에서 제거한다.

```text
C__         고정 암종쌍 contrast
D__exact_   사전에 지정된 exact mutation
```

외부에 알려진 암종 관계나 mutation 이름을 모델 구조에 고정하면 외부 지식 사용으로
오해받을 수 있다. 대신 동일한 아이디어를 train 통계로 바꿨다.

| 제거한 방식 | train-only 대안 |
| --- | --- |
| 고정 exact mutation | fold-train support 기반 recurrent missense `R__` |
| 고정 암종쌍 | fold-train mutation prevalence 유사도 기반 자동 암종쌍 |

코드에서는 제거 후 한 번 더 검사한다.

```python
assert not any(
    name.startswith(("C__", "D__exact_"))
    for name in names
)
```

---

## 7. 왜 LGBM으로 다시 실험했는가

동일한 전처리를 LR과 LGBM에 적용하더라도 학습 방식이 다르다.

### LR

대략 다음 가중합을 학습한다.

\[
score_c = b_c + \sum_j w_{c,j}x_j
\]

각 피처의 독립적인 방향을 안정적으로 학습하는 데 강하다.

### LGBM

여러 조건을 tree 분기로 결합할 수 있다.

```text
if TP53 mutation == 1:
    if truncating_count >= 2:
        if burden >= threshold:
            특정 암종 방향
```

따라서 exp14의 연구 질문은 단순히 “LGBM이 LR보다 높은가?”가 아니다.

```text
LR이 잘 보는 선형·희소 신호와
LGBM이 잘 보는 비선형 조건 조합이
서로 다른 오류를 만드는가?
```

단일 LGBM의 점수가 LR보다 낮더라도 LR이 틀린 일부 환자를 안정적으로 맞히면
앙상블 가치가 있을 수 있다.

---

## 8. Train-discovered specialist의 근거

다중분류 모델 하나가 26개 암종을 동시에 구분하면 변이 패턴이 비슷한 암종쌍에서
판단이 흔들릴 수 있다. 그렇다고 암종쌍을 외부 지식으로 고정하지 않고, 각
outer-fold train에서 암종별 유전자 mutation prevalence를 계산한다.

암종 (c)의 벡터:

\[
v_c = [P(G_1=1|c), P(G_2=1|c), \ldots, P(G_p=1|c)]
\]

두 암종의 cosine similarity:

\[
sim(a,b)=\frac{v_a \cdot v_b}{\|v_a\|\|v_b\|}
\]

유사도가 높은 두 쌍을 fold-train에서 선택하고, 해당 두 암종만 구분하는 binary
LGBM specialist를 학습한다.

```text
메인 모델: 26개 암종을 전체적으로 구분
specialist: train에서 변이 패턴이 비슷한 두 암종의 내부 비율만 재검토
```

specialist는 두 암종에 배정된 총 확률을 바꾸지 않고 내부 비율만 보정한다.

\[
m=P(a)+P(b)
\]

\[
P'(a)=m\,r_{new},\qquad P'(b)=m(1-r_{new})
\]

이 실험은 고정 암종 지식을 사용하지 않으면서, 다중분류 모델의 국소적인 약점을
별도 모델로 보완할 수 있는지를 확인한다.

---

## 9. 발표에서 사용할 수 있는 핵심 메시지

### 짧은 버전

> 원본 mutation 문자열을 유전자 존재 여부, burden, mutation type, truncation,
> 반복 mutation, 아미노산 치환, topology로 구조화했습니다. 라벨을 이용하는
> enrichment는 fold 내부 cross-fitting으로 생성했고, 외부 지식으로 오해될 수
> 있는 고정 암종쌍과 고정 mutation 피처는 제거했습니다.

### 전처리 의도 강조 버전

> 같은 mutation 기록에서도 “어느 유전자인가”, “어떤 변이 유형인가”, “환자에게
> 얼마나 많은가”, “한 유전자에 집중됐는가”는 서로 다른 정보입니다. 이를 별도
> 피처로 분리해 LR에는 안정적인 희소 신호를, LGBM에는 비선형 조합 가능성을
> 제공했습니다.

### 규정 안전성 강조 버전

> vocabulary, recurrent mutation, class enrichment, 자동 암종쌍은 모두 각
> outer-fold train에서 다시 학습했습니다. validation과 test에는 이미 학습된
> 변환을 적용만 했으며, 외부 유전자·mutation·암종쌍 목록은 사용하지 않았습니다.

---

## 10. 발표 슬라이드 구성 예시

### 슬라이드 1 — 문제

```text
원본은 mutation 문자열
→ 모델이 문자열의 구조를 직접 이해하지 못함
```

### 슬라이드 2 — 한 환자 변환 예시

```text
TP53="R175H R248H", APC="Q1367*"
→ mutation gene 2개
→ event 3개
→ missense 2개, nonsense 1개
→ truncating gene 1개
→ R→H 치환 2개
→ 한 유전자 최대 event 2개
```

### 슬라이드 3 — 피처 블록

| 블록 | 의미 |
| --- | --- |
| G | 어떤 유전자가 변했는가 |
| B | 전체 변이가 얼마나 많은가 |
| V | mutation 종류가 무엇인가 |
| T | truncating 계열인가 |
| R | train에서 반복된 exact missense인가 |
| A | 어떤 아미노산 치환인가 |
| S | mutation이 어떻게 분포하는가 |
| E | train의 어떤 암종 패턴과 닮았는가 |

### 슬라이드 4 — 검증 안전성

```text
fold-train: 통계와 vocabulary 학습
fold-valid: 적용만 수행
test: 최종 후보 확정 후 transform/predict만 수행
```

### 슬라이드 5 — 모델 연구 질문

```text
LR의 안정적인 희소 신호
          +
LGBM의 비선형 조합과 다른 오류
          ↓
안전한 OOF 앙상블 후보 확보
```

---

## 11. 해석 시 주의사항

- train 통계로 선택된 유전자를 곧바로 공인 바이오마커라고 부르지 않는다.
- feature importance가 높다는 것은 해당 모델이 많이 사용했다는 뜻이지 생물학적
  인과관계를 증명한 것이 아니다.
- fold마다 recurrent mutation과 자동 암종쌍이 달라질 수 있다.
- 단일 seed 최고점보다 42/52/62 세 seed의 평균과 방향을 우선한다.
- 기존 exp14 점수는 고정 암종쌍 버전이므로 안전 버전 재실행 후 갱신해야 한다.
- 현재 blend grid는 탐색용이다. 최종 weight는 팀 검증 계약에 맞는 nested
  outer-fold 절차로 확정해야 한다.

---

## 12. 규정 안전 버전 최종 결과

고정 exact mutation과 고정 암종쌍을 제거한 뒤 전체 실험을 다시 실행했다.

| 단계 | Seed 42 OOF Macro F1 |
| --- | ---: |
| 안전 버전 LR | 0.526130 |
| balanced multiclass LGBM | 0.476313 |
| train-discovered hard specialist | 0.492332 |
| LR 80% + specialist LGBM 20% | **0.543679** |

3-seed 결과:

| Seed | LR | Specialist LGBM | Blend | LR 대비 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.526130 | 0.492332 | **0.543679** | +0.017549 |
| 52 | 0.529272 | 0.488738 | **0.540053** | +0.010780 |
| 62 | 0.527424 | 0.505154 | **0.535802** | +0.008378 |
| 평균 | 0.527609 | 0.495408 | **0.539845** | **+0.012236** |

레거시 고정 암종쌍 blend 평균 `0.538052`보다 안전 버전이 `+0.001793`
높았다. 규정 위험 요소를 제거하면서 성능도 유지된 것이 아니라 오히려 소폭
개선됐다. 이는 외부에 알려진 암종쌍을 고정하지 않아도 fold-train 통계만으로
앙상블에 유용한 보조 모델을 만들 수 있음을 보여준다.
