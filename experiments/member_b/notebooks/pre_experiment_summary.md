# Member B Pre-Experiment Summary

## 목적

유전체 변이 기반 다중분류 문제에서, 공용 benchmark 기준으로 `Macro F1`를 유지하거나 개선할 수 있는 전처리 방향을 찾는 것이 목표였다.
비교 기준은 항상 `team_baseline_binary`였고, split, seed, fold, 모델 파라미터는 공용 benchmark 규칙을 따랐다.

## 데이터와 기본 해석

- 데이터 형태:
  - TCGA/MAF 원본 long table이 아니라, 이미 `sample x gene` wide mutation matrix 형태
  - gene이 컬럼으로 펼쳐져 있고, 각 셀 값은 `WT` 또는 변이 token
- 클래스 구조:
  - 총 26개 클래스
  - 최소 클래스 `DLBC = 38`
  - 최대 클래스 `BRCA = 786`
  - 클래스 불균형 비율 약 `20.68x`
- QA 결과:
  - token 정규화 이슈는 샘플 점검 범위에서 크지 않았음
  - train/test drift 존재
  - test의 평균 mutation gene count가 train보다 훨씬 높았음
  - 일부 gene prevalence도 train/test 간 차이가 컸음

해석:
- `Macro F1` 기준에서는 소수 클래스 성능이 중요하다.
- train/test drift가 존재하므로 복잡한 요약 feature보다 단순한 gene-level binary 표현이 더 안정적일 가능성이 높다.

## 기준선

### Baseline

- `team_baseline_binary`
- 의미:
  - 각 gene에서 `WT = 0`, 변이 존재 = `1`
- benchmark 기준 대표 점수:
  - full benchmark logistic 기준 `0.344689`

### Baseline 재현

- `member_b_binary_only`
- 결과:
  - baseline과 동일 성능
- 해석:
  - baseline 재현 성공

## 실험 흐름과 결과

### 1. Sample / Gene Summary 계열

실험 이유:
- EDA에서 샘플별 변이 수와 token 다양성이 보여서, sample-level 통계가 보조 feature가 될 수 있는지 확인

실험:
- `member_b_binary_plus_stats`
- `member_b_binary_type_stats`

결과:
- baseline 대비 큰 폭 하락

해석:
- 현재 방식의 sample/gene statistics, mutation type count는 noise 가능성이 높음

### 2. Feature Group Ablation

실험 이유:
- 어떤 정보가 실제로 성능에 기여하는지 분해해서 확인

실험 그룹:
- `binary_mutation`
- `rare_mutation`
- `original_mutation`

결과:
- `binary_mutation` 제거 시 성능 하락
- `rare_mutation` 제거 시 성능 하락
- `original_mutation` 제거 시 성능 개선

해석:
- `binary_mutation`은 핵심
- `rare_mutation`도 보조적으로 의미 있음
- `original_mutation`은 현재 encoding 방식에서는 noise 가능성

### 3. Filtering / Selection 계열

실험 이유:
- 불필요한 feature 제거 또는 supervised selection으로 baseline 개선 가능성 확인

실험:
- constant feature 제거
- duplicate feature 제거
- variance threshold
- minimum frequency filtering
- chi-square feature selection

결과:
- `constant + duplicate 제거`
  - baseline과 동일 성능
  - feature 수 감소
- `minimum frequency >= 2`
  - fast 모드에서는 baseline보다 높아 보였음
  - full benchmark에서는 baseline보다 아주 소폭 낮음
  - baseline에 매우 근접한 경량화 후보
- `minimum frequency >= 3`, `>= 5`
  - `>= 2`보다 더 약함
- `chi2 feature selection`
  - fast 모드에서는 강한 개선처럼 보였음
  - full benchmark에서는 baseline보다 낮음

해석:
- 정리형 filtering은 무손실 경량화에는 의미가 있음
- aggressive selection은 full benchmark에서 일반화 이득으로 이어지지 않음

### 4. Rare / Weighting / Drift 대응 아이디어

실험 이유:
- rare mutation이 중요하다는 신호가 있어, 희귀 변이를 더 잘 반영할 수 있는지 확인

실험:
- frequency weighted binary
- cancer specificity score
- gene group aggregation
- correlation filtering

결과:
- 모두 baseline보다 낮거나 동일

해석:
- 희귀 변이 자체는 의미가 있지만, 현재 weighting / aggregation 방식은 baseline을 넘지 못함
- correlation filtering은 성능 개선은 없고 경량화 보조 수준

### 5. Sample Complexity / Rare-Common Composition / Interaction

실험 이유:
- 샘플 수준 요약이나 상호작용 feature가 추가 정보를 줄 수 있는지 확인

실험:
- sample complexity features
- rare/common composition
- gene interaction features

결과:
- 모두 baseline보다 크게 낮음

해석:
- sample-level 요약이나 제한된 interaction feature는 현재 데이터 표현에서 정보 손실이 큼

### 6. Biological Prior 기반 요약

실험 이유:
- driver gene, oncogene/TSG, pathway, known pair 등 생물학적 prior를 요약 feature로 추가하면 도움이 되는지 확인

실험:
- baseline + driver gene features
- baseline + oncogene/TSG features
- baseline + pathway features
- baseline + pathway damage breadth
- baseline + known gene pair features

결과:
- 모두 baseline보다 크게 낮음

해석:
- 현재 wide mutation matrix에서는 biological prior를 작은 요약 feature로 압축하는 방식이 원래 gene-level binary signal을 대체하거나 보강하지 못함

## 최종 판단

### 성능 기준 1순위

- `team_baseline_binary`

### 보조 후보

- `constant + duplicate 제거`
  - 성능 손실 없이 feature 수를 줄이는 정리형 전처리
- `minimum frequency >= 2`
  - benchmark 기준 baseline보다 아주 소폭 낮지만, 경량화 후보로는 의미 있음

### 채택하지 않은 방향

- sample/gene statistics
- mutation type count
- original mutation token
- frequency weighting
- cancer specificity
- gene group aggregation
- sample complexity
- rare/common composition
- gene interaction
- chi-square feature selection
- biological prior 기반 요약 feature

## 최종 결론

- full benchmark 기준에서 가장 안정적이고 높은 성능을 유지한 전처리는 `team_baseline_binary`였다.
- 여러 추가 전처리와 feature engineering을 시도했지만, baseline을 일관되게 넘는 방향은 확인되지 않았다.
- 현재 데이터에서는 복잡한 요약 feature보다 `gene-level binary representation`이 가장 강하게 작동했다.
- 제출 점수 방어 관점에서는 baseline binary 유지가 가장 보수적이고 안전한 전략이다.
