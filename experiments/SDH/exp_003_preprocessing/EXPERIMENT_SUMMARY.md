# exp_003 실험 요약 및 후속 연구

## 1. 목적

공용 `run_preprocessing_benchmark()`의 모델, 5-Fold, 평가 조건을 고정하고
유전체 문자열 전처리 10종만 비교했다. 주요 질문은 mutation burden, 변이 유형,
유전자 빈도 필터 및 hotspot 피처가 기본 WT 이진화보다 유효한지 확인하는 것이다.

## 2. 실험 조건

- 데이터: train 6,201행, 유전자 피처 4,384개
- 검증: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 모델: 공용 벤치마크 Logistic Regression
- 주 지표: 전체 OOF Macro F1
- 누수 방지:
  - 전처리 객체를 fold마다 복제하고 train 부분에서만 `fit`
  - 유전자 빈도 기준과 hotspot 목록을 fold train에서만 결정
  - validation에는 학습된 규칙으로 `transform`만 적용

현재 공용 Logistic Regression 설정에는 `C`가 명시되어 있지 않으므로 sklearn
기본값 `C=1.0`이 적용된다. 아래 결과는 이 조건에서의 전처리 비교 결과다.

## 3. seed 42 결과

| 순위 | Case | 전처리 | OOF Macro F1 | Accuracy | baseline 대비 |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | 06 | 두 burden + 변이 유형별 개수 | **0.37803** | 0.37413 | **+0.03350** |
| 2 | 10 | 두 burden + hotspot top 50 | **0.37226** | 0.36736 | **+0.02773** |
| 3 | 09 | 두 burden + 최소 변이 빈도 10 | **0.36605** | 0.36091 | **+0.02153** |
| 4 | 03 | gene burden | 0.36238 | 0.35962 | +0.01785 |
| 5 | 05 | 두 burden | 0.36239 | 0.36075 | +0.01787 |
| 6 | 08 | 두 burden + 최소 변이 빈도 5 | 0.36121 | 0.35994 | +0.01669 |
| 7 | 07 | 두 burden + 최소 변이 빈도 3 | 0.36118 | 0.35994 | +0.01665 |
| 8 | 04 | token burden | 0.36094 | 0.35865 | +0.01641 |
| 9 | 01 | WT 이진화 | 0.34452 | 0.34285 | 기준 |
| 10 | 02 | 상수열 제거 | 0.34439 | 0.34269 | -0.00013 |

case 05의 seed 42 값은 3-seed confirmation 결과 안의 seed 42 기록이다.

## 4. 해석

### 변이 유형별 개수가 가장 유망하다

synonymous, missense, nonsense, frameshift, other의 샘플별 `log1p` 개수를 추가한
case 06이 baseline보다 Macro F1 0.03350 높았다. 단순히 변이 존재 여부만 사용하는
것보다 변이의 성격을 요약한 정보가 암종 구분에 유효하다는 가설을 지지한다.

### hotspot 피처도 강한 개선을 보였다

fold train에서 가장 자주 나온 mutation token 50개를 이진 피처로 추가한 case 10이
두 번째로 높았다. 다만 특정 seed 또는 빈도가 높은 일부 클래스에 의존할 수 있어
반복 검증과 클래스별 F1 확인이 필요하다.

### 빈도 필터는 임계값 10이 가장 좋았다

최소 빈도 3과 5는 비슷했지만 10에서 성능이 더 높았다. 희귀 유전자 피처를 줄이는
것이 현재 선형 모델의 분산을 낮췄을 가능성이 있다. 10보다 큰 임계값은 후속 탐색
대상이다.

### burden 두 개의 중복성이 크다

gene burden과 token burden은 각각 baseline을 개선했지만 두 피처를 함께 사용해도
gene burden 단독보다 뚜렷하게 좋아지지 않았다. 한 셀에 여러 mutation token이
기록된 정보의 추가 기여가 제한적일 수 있다.

### 상수열 제거만으로는 효과가 없다

fold별 상수열 제거는 피처 수를 줄였지만 점수는 사실상 동일했다. 계산량 절감
목적에는 사용할 수 있지만 독립적인 성능 개선 전처리로 보기는 어렵다.

## 5. 완료된 반복 검증

case 05 두 burden의 3-seed 결과:

| Seed | OOF Macro F1 | Accuracy |
| ---: | ---: | ---: |
| 42 | 0.36239 | 0.36075 |
| 52 | 0.35463 | 0.35462 |
| 62 | 0.35740 | 0.35607 |
| 평균 | **0.35814 ± 0.00394** | **0.35715** |

세 seed 모두 baseline의 기존 3-seed 평균 0.33738보다 높아 burden 개선은 한 seed에만
국한되지 않는 것으로 판단한다.

## 6. 바로 실행할 후속 연구

새 실험으로 분리하지 않고 exp_003에서 두 단계로 검증한다.

### 6.1 LR 3-seed confirmation

다음 세 후보를 seed 42/52/62로 반복 검증한다.

1. `case_06_mutation_types`
2. `case_10_hotspot_top50`
3. `case_09_min_count_10`

채택 기준:

- 3-seed 평균 OOF Macro F1이 case 05와 baseline보다 높다.
- 세 seed에서 개선 방향이 일관된다.
- 표준편차가 과도하게 증가하지 않는다.
- 클래스별 F1 개선이 소수 클래스의 급등에만 의존하지 않는다.

### 6.2 LightGBM 2차 검증

LR에서 선별된 세 후보와 WT 이진화 baseline을 공용 LightGBM의 동일 seed 42,
동일 5-Fold로 비교한다.

1. `case_01_wt_binary`: LightGBM 기준점
2. `case_06_mutation_types`
3. `case_10_hotspot_top50`
4. `case_09_min_count_10`

LightGBM seed 42에서 세 후보가 모두 baseline보다 0.005 이상 높아 자동으로
3-seed confirmation 대상으로 선택됐다.

| 순위 | 전처리 | 3-seed Macro F1 | 표준편차 | Accuracy | baseline 대비 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `case_06_mutation_types` | **0.37260** | 0.00425 | 0.37456 | **+0.08269** |
| 2 | `case_10_hotspot_top50` | 0.35932 | **0.00168** | 0.36215 | +0.06940 |
| 3 | `case_09_min_count_10` | 0.35387 | 0.00205 | 0.35924 | +0.06395 |
| 4 | `case_01_wt_binary` | 0.28992 | 0.00080 | 0.30361 | 기준 |

`case_06_mutation_types`의 seed별 Macro F1:

| Seed | Macro F1 | Accuracy |
| ---: | ---: | ---: |
| 42 | 0.37385 | 0.37623 |
| 52 | 0.36787 | 0.36930 |
| 62 | 0.37609 | 0.37816 |

변이 유형 피처는 모든 seed에서 가장 높았고 LR seed 42에서도 0.37803으로
1위였다. 따라서 모델 종류와 seed가 달라져도 개선 방향이 유지되는 확정 전처리
1순위로 판단한다.

현재 모델 비교에서는 LR case 06이 LightGBM case 06보다 근소하게 높다. LR을
주력 모델로 유지하고 LightGBM은 비선형 검증 및 향후 앙상블 다양성 후보로 둔다.

## 7. 반복 검증 후 연구 후보

상위 후보의 안정성이 확인됐으므로 exp_004에서 다음 조합을 진행한다.

1. 변이 유형 + 최소 빈도 10
2. 변이 유형 + hotspot top 50
3. 변이 유형 + 최소 빈도 10 + hotspot
4. 최소 빈도 임계값 10/15/20/30 비교
5. hotspot 개수 20/50/100 비교
6. 변이 유형별 burden의 클래스별 기여 및 계수 분석

조합 실험에서도 공용 벤치마크 조건을 유지한다. 전처리가 확정되기 전에는 모델
하이퍼파라미터 튜닝이나 soft voting을 섞지 않는다.

## 8. 최종 결론

- 확정 전처리 1순위: 변이 유형별 `log1p` 개수를 포함한 case 06
- 차선 후보: hotspot top 50
- 빈도 필터 후보: 최소 변이 빈도 10
- 주력 1차 모델: Logistic Regression
- 2차 검증 및 다양성 모델: LightGBM
- 다음 단계: case 06을 기준으로 hotspot 크기와 빈도 임계값을 조합한 exp_004
