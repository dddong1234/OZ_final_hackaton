# 암종 분류를 위한 암 유전체 변이 데이터 EDA 분석 보고서

---

## Executive Summary (요약)

본 분석 보고서는 6,201개 암 샘플 데이터를 바탕으로 **유전자 변이 패턴, 암종별 대표 변이, 변이 타입 분포, 변이 부담(TMB), 그리고 유전자 공동 변이(Co-occurrence)**를 종합적으로 분석한 결과입니다.

### 핵심 인사이트 (Key Insights)
1. **변이 빈도의 극심한 불균형**: `TP53`(28.5%)을 비롯한 상위 유전자에 변이가 집중되어 있으나, 유전자 길이(Gene Length)에 의한 노이즈 유전자(`RYR2`, `SYNE1` 등)와 실제 드라이버 유전자를 구분하는 것이 핵심입니다.
2. **암종별 특이 피처(Hotspot & Feature Importance)**: 단일 유전자 유무(`0/1`)뿐만 아니라, `PIK3CA_H1047R`, `BRAF_V600E`와 같은 **Specific Hotspot Mutation** 및 **암종 내 변이율(Ratio)** 도입이 필수적입니다.
3. **Tumor Mutation Burden (TMB) 차이**: 암종별 중앙값 변이 수가 2개(`THYM`)에서 85.5개(`SKCM`)까지 극단적인 차이를 보이며, 이는 그 자체로 강력한 분류 신호(Classification Signal)로 작용합니다.
4. **공동 변이 패턴(Co-occurrence)**: 뇌종양(`IDH1 + TP53 + ATRX`), 대장암(`APC + TP53`) 등 주요 암종에서 정교한 콤보 신호가 확인되었으며, 단순 횟수가 아닌 **Lift / Jaccard Index** 중심의 피처 엔지니어링이 요구됩니다.

---

## 1. 전체 유전자 변이 빈도 분석

전체 6,201개 샘플 중 가장 높게 관측된 주요 유전자 변이 비율 분석 결과입니다.

### 1.1 상위 변이 유전자 비율

| 순위 | 유전자 (Gene) | Mutation Count | Mutation Ratio (%) | Wild Type (WT) Ratio (%) | 비고 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **TP53** | **1,770** | **28.5%** | **71.5%** | 전체 샘플 4개 중 1개 이상 관측 |
| **2** | **PIK3CA** | - | **11.1%** | 88.9% | 주요 Oncogene |
| **3** | **RYR2** | - | **10.4%** | 89.6% | 긴 유전자 길이 (Passenger 가능성) |
| **4** | **SYNE1** | - | **10.4%** | 89.6% | 긴 유전자 길이 (Passenger 가능성) |
| **5** | **PCLO** | - | **9.6%** | 90.4% | 긴 유전자 길이 |
| **6** | **RYR1** | - | **7.6%** | 92.4% | - |
| **7** | **SPTA1** | - | **7.4%** | 92.6% | - |
| **8** | **KMT2D** | - | **7.3%** | 92.7% | 후성유전학 관련 드라이버 |
| **9** | **IDH1** | - | **7.1%** | 92.9% | 신경교종 특이 드라이버 |
| **10** | **BRAF** | - | **7.1%** | 92.9% | 주요 Targetable Mutation |

### 1.2 유전자 성격별 분류 및 해석

```
                    [ 유전자 변이 상위 집단 ]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
[ Cancer Driver Genes ]                         [ Passenger / Long Genes ]
- TP53, PIK3CA, IDH1, BRAF, APC, PTEN           - RYR2, SYNE1, PCLO, DMD, AHNAK
- 암 발생 및 진행에 직접 관여                   - 유전자 길이가 길어 변이가 누적됨
- 특정 암종에서 강력한 구분력 제공              - 샘플의 전체 변이 부담(TMB)과 비례
```

* **인사이트**: 단순히 전체 변이 빈도가 높다고 해서 암종 구분력이 높다고 단정할 수 없습니다. 향후 모델링 시 단순 빈도보다 **암종별 특이도(Specificity)** 중심의 피처 가공이 필요합니다.

---

## 2. 암종별 대표 유전자 분석

각 암종별로 최다 변이를 보이는 대표 유전자 목록입니다. 절대 개수는 암종별 전체 샘플 수(`subclass_sample_count`)의 영향을 받으므로 암종 내 변이율로 해석해야 합니다.

### 2.1 암종별 주요 변이 유전자 개수

#### 유방암 (BRCA, Total Samples = 786)
* **PIK3CA**: 250개
* **TP53**: 218개
* **CDH1**: 73개
* **GATA3**: 58개
* **MAP3K1**: 56개

#### 대장암 (COAD)
* **APC**: 160개 *(대장암 계열에서 가장 압도적인 특이 피처)*
* **TP53**: 115개
* **SYNE1**: 62개
* **FBXW7**: 37개
* **DMD**: 35개

#### 방광암 (BLCA)
* **TP53**: 55개
* **KMT2D**: 31개
* **PIK3CA**: 23개
* **SYNE1**: 22개
* **SPTAN1**: 19개

#### 미만성 거대 B세포 림프종 (DLBC, Total Samples = 38)
* **KMT2D**: 15개
* **BTG2**: 12개
* **SYNE1**: 10개
* **BTG1**: 10개
* **PIM1**: 9개

### 2.2 개선 과제
$$ 	ext{Amended Mutation Rate} = rac{	ext{mutated\_sample\_count}}{	ext{subclass\_sample\_count}} $$
샘플 수가 큰 BRCA(786개)와 작은 DLBC(38개)의 절대 비교 불균형을 해소하기 위해, **암종 내 비율(Amended Mutation Rate)** 변환을 차기 단계에서 적용합니다.

---

## 3. 암종별 대표 세부 변이 (Hotspot Mutation)

단순 유전자 유무(`Gene Level`)를 넘어 **`암종 + 유전자 + 구체적 변이`** 수준의 세부 변이 분석 결과입니다.

```
[ 피처 확장 예시 ]
  - 기존: PIK3CA (0 or 1)
  - 확장: PIK3CA_H1047R (0 or 1) / PIK3CA_E545K (0 or 1) / PIK3CA_E542K (0 or 1)
```

### 3.1 주요 암종별 Hotspot 변이 분포

| 암종 (Subclass) | 유전자 & 변이 (Gene & Mutation) | 관측 횟수 (Count) | 해석 및 특이사항 |
| :--- | :--- | :---: | :--- |
| **BRCA** | `PIK3CA H1047R` | **92** | Kinase domain 대표 핫스팟 변이 |
| | `PIK3CA E545K` | **40** | Helical domain 핫스팟 변이 |
| | `PIK3CA E542K` | **26** | Helical domain 핫스팟 변이 |
| | `TP53 R175H` | **12** | DNA-binding domain 변이 |
| | `GATA3 G334fs` | **8** | Frameshift 변이 |
| **COAD** | `BRAF V600E` | **19** | 대장암 예후 관련 주요 핫스팟 |
| | `TP53 R175H` | **14** | 주요 Hotspot |
| | `FBXW7 R465H` | **8** | - |
| | `PIK3CA E545K` | **8** | - |
| | `TP53 R248W` | **8** | - |
| **CESC** | `PIK3CA E545K` | - | 자궁경부암 주요 변이 |
| | `PIK3CA E542K` | - | 자궁경부암 주요 변이 |
| | `MAPK1 E322K` | - | - |

> ⚠️ **주의사항 (ACC 암종)**
> `ACC`에서 관측된 `SOWAHC L42L`, `CMPK2 C153C` 등은 동의 변이(Silent)이거나 노이즈일 수 있습니다. 임상적 Driver 변이로 단정하기 어려우므로 데이터 수집 특성이나 Passenger 변이 여부를 검증해야 합니다.

---

## 4. Mutation Type 분포 및 파서 검증

### 4.1 Mutation Type별 발생 건수

```
Missense-like  ████████████████████████████████████████ 178,415
Other          ████ 18,112
Nonsense       ███ 12,821
Frameshift     ██ 9,542
Indel          ▏3
```

* **Missense-like**: 178,415건 (압도적 다수)
* **Other**: 18,112건
* **Nonsense**: 12,821건
* **Frameshift**: 9,542건
* **Indel**: 3건

### 4.2 분석 및 개선 방향
1. **단순 Count 변수의 한계**: 대부분의 샘플에서 `Missense-like`가 독점적이므로, 단순히 개수를 카운트하는 피처는 구분력이 낮습니다.
2. **비율 변수화(Ratio)**: 샘플별/암종별 `missense_ratio`, `nonsense_ratio`, `frameshift_ratio`를 산출하여 상대적 비율을 피처로 활용해야 합니다.
3. **Parser 재검증**: `Indel`이 3건에 불과한 것은 실제 생물학적 결과라기보다, 변이 문자열 파싱 규칙(Parsing Rule)에서 대다수의 Indel이 `Frameshift` 또는 `Other`로 오분류되었을 가능성이 높으므로 파서 로직 검증이 권장됩니다.

---

## 5. 암종별 샘플당 변이 수 (TMB / Mutation Burden)

암종별 **Median Mutation Count**는 극심한 차이를 보이며, 이는 암종 분류 모델에서 매우 강력한 피처로 작동합니다.

### 5.1 암종별 변이 부담 비교

#### High Mutation Burden 암종 (중앙값 기준)
* **SKCM** (피부흑색종): **85.5**
* **LUSC** (폐편평세포암): **57.0**
* **BLCA** (방광암): **54.0**
* **STES** (위식도암): **41.0**
* **LUAD** (폐선암): **40.5**
* **HNSC** (두경부암): **31.0**
* **DLBC** (림프종): **30.0**
* **LIHC** (간암): **25.0**
* **ACC** (부신피질암): **24.0**
* **COAD** (대장암): **23.0**

#### Low Mutation Burden 암종 (중앙값 기준)
* **BRCA** (유방암): **9.0**
* **LGG** (저등급 신경교종): **8.0**
* **PRAD** (전립선암): **6.0**
* **THCA** (갑상선암): **4.0**
* **PCPG** (크롬친화세포종): **4.0**
* **LAML** (급성골수성백혈병): **3.0**
* **THYM** (흉선암): **2.0**

### 5.2 치우침(Skewness) 처리 전략

일부 암종(예: UCEC)은 극단적인 Hypermutated 샘플 존재로 인해 평균과 중앙값의 격차가 매우 큽니다.

| 암종 (Subclass) | 평균 (Mean) | 중앙값 (Median) | 특이사항 |
| :--- | :---: | :---: | :--- |
| **UCEC** | **134.0** | **18.0** | 극단적 우측 스큐 (Hypermutated Sample 다수) |
| **SKCM** | **133.0** | **85.5** | 전반적으로 높은 변이율 |
| **PAAD** | **36.0** | **14.0** | 우측 스큐 |
| **BRCA** | **17.0** | **9.0** | 우측 스큐 |

#### 피처 변환 방안
* Logistic Regression 등 선형 모델 적용 시 Outlier 영향을 줄이기 위한 **로그 변환** 필수:
  $$ 	ext{Transformed Feature} = \log(1 + 	ext{total\_mutation\_count}) $$
* Tree 기반 모델(LightGBM, XGBoost)에서도 과도한 Split 분할 방지를 위해 Raw Count와 Log 변환 피처의 성능을 비교 검증해야 합니다.

---

## 6. 유전자 공동 변이 (Co-occurrence) 분석

특정 유전자 조합이 함께 변이되는 패턴은 단일 유전자 피처보다 강력한 암종 특이 신호가 됩니다.

### 6.1 암종별 대표 공동 변이 조합

```
[ 뇌종양 계열 (LGG / GBMLGG) ]      [ 대장암 (COAD) ]             [ 유방암 (BRCA) ]
  IDH1 + TP53 + ATRX                 APC + TP53                    PIK3CA + TP53
  └─ 강한 3중 시너지                  └─ 주요 개시 변이 조합         └─ 대표 드라이버 조합
```

* **BRCA**: `PIK3CA + TP53` (50회), `PIK3CA + CDH1` (35회), `PIK3CA + MAP3K1` (27회)
* **COAD**: `APC + TP53` (86회), `APC + SYNE1` (43회), `APC + PIK3CA` (28회)
* **GBMLGG / LGG**: `IDH1 + TP53` (117회 / 107회), `IDH1 + ATRX` (91회 / 79회), `TP53 + ATRX` (83회 / 76회)
* **HNSC**: `TP53 + CDKN2A` (40회), `TP53 + PCLO` (34회), `TP53 + SYNE1` (29회)
* **KIPAN / KIRC**: `VHL + MTOR`, `VHL + KMT2D`, `VHL + RYR1`
* **LAML**: `NPM1 + IDH1`, `NPM1 + PTPN11`, `IDH2 + RUNX1`

### 6.2 향후 지표 산출 공식 (Co-occurrence Refinement)

단순 관측 횟수(Co-count)는 독립적으로 빈도가 높은 유전자(`TP53` 등)에 의한 착시가 발생하므로, 아래의 통계 지표를 추가 산출해야 합니다.

1. **Jaccard Similarity Index**:
   $$ J(A, B) = rac{|A \cap B|}{|A \cup B|} $$

2. **Lift (상호 연관성)**:
   $$ 	ext{Lift}(A, B) = rac{P(A \cap B)}{P(A) 	imes P(B)} $$
   * $	ext{Lift} > 1$: 두 유전자가 독립적일 때보다 훨씬 높은 확률로 동시 발생 (강한 상호작용).

---

## 7. 향후 피처 엔지니어링 및 모델링 액션 플랜

```
1. 수치형 피처 (Numeric)
   ├── log1p_total_mutation_count (로그 변환 TMB)
   ├── mutated_gene_count (변이 유전자 수)
   └── missense / nonsense / frameshift ratio (변이 타입 비율)

2. 범주형 / 바이너리 피처 (Binary / Categorical)
   ├── Specific Hotspot Features (e.g., PIK3CA_H1047R, BRAF_V600E)
   └── Subclass Normalized Mutation Rate (암종 내 변이 비율)

3. 상호작용 피처 (Interaction)
   └── High Lift Gene Pair Combo (Lift > 1.5 이상 유전자 쌍 Binary 피처)
```

1. **변이율 정규화**: 암종별 샘플 수 차이를 보정한 `mutated_sample_count / subclass_sample_count` 산출.
2. **핫스팟 피처화**: `PIK3CA_H1047R`, `BRAF_V600E` 등 반복 등장 세부 변이의 파생 변수화.
3. **TMB 변환**: `log1p(total_mutation_count)` 및 `mutation_count_bin` 생성.
4. **Co-occurrence 지표화**: 단순 Co-count를 넘어 `Lift` 및 `Jaccard Similarity` 계산 후 높은 신호를 보이는 조합 피처 선택.