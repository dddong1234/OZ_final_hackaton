# SDH exp-002 전처리 Baseline 결과

## 결론

공용 전처리 벤치마크의 기준 점수는 다음과 같다.

| 지표 | 결과 |
|---|---:|
| **OOF Macro F1** | **0.34452** |
| OOF Accuracy | 0.34285 |
| Fold Macro F1 평균 ± 표준편차 | 0.34322 ± 0.00887 |
| Fold Accuracy 평균 ± 표준편차 | 0.34285 ± 0.01101 |
| 전체 실행 시간 | 52.35초 |

이후 전처리 실험은 동일한 공용 Logistic Regression과 동일한 fold에서 실행하고, **OOF Macro F1 0.34452**를 1차 비교 기준으로 사용한다.

### 추가 검증 요약

| 모델 | CV seeds | OOF Macro F1 | OOF Accuracy | 실행 시간 |
|---|---|---:|---:|---:|
| Logistic Regression | 42 | 0.34452 | 0.34285 | 52.35초 |
| Logistic Regression | 42, 52, 62 | **0.33738 ± 0.00625** | 0.33640 ± 0.00573 | 152.66초 |
| LightGBM | 42 | 0.29058 | 0.30463 | 135.44초 |

- Logistic Regression의 seed별 OOF Macro F1은 0.34452, 0.33295, 0.33466이었다.
- 현재 WT/변이 이진 전처리에서는 Logistic Regression이 LightGBM보다 OOF Macro F1 기준 0.05394 높았다.
- 이후 전처리의 안정적인 비교 기준은 Logistic Regression 3-seed 평균 **0.33738 ± 0.00625**로 사용한다.
- LightGBM 결과는 비선형 모델에서 현재 이진 전처리의 기준점으로 사용한다.

---

## 실험 정보

| 항목 | 값 |
|---|---|
| 실험 ID | `sdh-exp-002-baseline` |
| 실험 목적 | 팀 공용 전처리 비교 기준점 확립 |
| 학습 데이터 | `train.csv` |
| 데이터 크기 | 6,201행 |
| 타깃 | `SUBCLASS` |
| 클래스 수 | 26 |
| 입력 feature 수 | 4,384 |
| 주 평가 지표 | 전체 OOF 예측의 Macro F1 |
| 보조 평가 지표 | OOF Accuracy |

---

## 전처리

`common.starter_preprocess`의 기본 전처리를 사용했다.

```text
결측값 → WT
WT → 0
WT가 아닌 변이 문자열 → 1
```

이번 baseline에서는 다음 처리를 적용하지 않았다.

- 전체가 `WT`인 유전자 제거
- 변이 빈도 기반 feature selection
- 구체적인 변이 문자열 또는 변이 유형 분리
- TMB 및 샘플별 변이 수 파생 feature
- 스케일링
- 차원 축소

모든 4,384개 유전자 컬럼이 각 fold에서 동일하게 사용됐다.

---

## 검증 조건

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

- 각 행은 validation에 정확히 한 번 포함된다.
- 5개 fold의 예측을 원래 행 순서에 맞춰 결합해 OOF 점수를 계산한다.
- 전처리 Pipeline은 각 fold마다 clone되며 `fit`은 fold 학습 부분에만 적용한다.
- 최종 비교 점수는 fold 점수의 최댓값이나 단순 평균이 아니라 전체 OOF Macro F1이다.

---

## 모델

```python
LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
```

명시하지 않은 주요 기본값:

```text
penalty = "l2"
C = 1.0
fit_intercept = True
tol = 1e-4
```

전처리 비교 중에는 위 모델 파라미터를 변경하지 않는다. 모델 파라미터를 변경한 결과는 전처리 실험이 아닌 별도 모델 실험으로 취급한다.

---

## Fold별 결과

| Fold | Train | Validation | Features | Accuracy | Macro F1 | 시간 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,960 | 1,241 | 4,384 | 0.35294 | 0.34554 | 10.32초 |
| 2 | 4,961 | 1,240 | 4,384 | 0.32661 | 0.32894 | 11.26초 |
| 3 | 4,961 | 1,240 | 4,384 | 0.34194 | 0.35267 | 9.79초 |
| 4 | 4,961 | 1,240 | 4,384 | 0.33952 | 0.34190 | 10.03초 |
| 5 | 4,961 | 1,240 | 4,384 | 0.35323 | 0.34704 | 10.91초 |
| **평균** |  |  | **4,384** | **0.34285** | **0.34322** |  |
| **표본 표준편차** |  |  |  | **0.01101** | **0.00887** |  |

Fold Macro F1의 최소값은 0.32894, 최대값은 0.35267이며 변동폭은 0.02373이다.

---

## 점수 해석

### OOF 점수와 fold 평균의 차이

- `0.34452`: 전체 6,201개 OOF 예측을 한 번에 평가한 Macro F1
- `0.34322`: 5개 fold에서 각각 계산한 Macro F1의 단순 평균

두 값은 계산 방식이 다르므로 정확히 일치하지 않는다. 팀의 전처리 순위 비교에는 전체 데이터 기준인 **OOF Macro F1**을 사용한다.

### 표준편차

이번 기본 실행의 CV seed는 42 하나뿐이므로 seed 간 OOF 표준편차는 계산할 수 없다. `± 0.00887`은 seed 간 편차가 아니라 **5개 fold 점수의 표본 표준편차**다.

최종 후보의 split 안정성을 확인할 때는 다음 seed로 반복 5-fold를 실행한다.

```text
42, 52, 62
```

---

## 재현 방법

프로젝트 루트에서 JupyterLab을 실행한 뒤 다음 노트북의 셀을 순서대로 실행한다.

```text
experiments/SDH/exp_002_baseline/experiment.ipynb
```

핵심 호출:

```python
from common.preprocessing_benchmark import run_preprocessing_benchmark
from common.starter_preprocess import make_baseline_preprocessor

preprocessor = make_baseline_preprocessor()

baseline_result = run_preprocessing_benchmark(
    train,
    preprocessor,
    experiment_id="sdh-exp-002-baseline",
    preprocessing_name="WT/variant binary",
)
```

경량 재현 정보와 상세 fold 결과는 같은 폴더의 `metrics.json`에 저장된다.

---

## 후속 전처리 비교 규칙

1. sklearn Transformer 또는 전처리 Pipeline만 변경한다.
2. fold, seed, 모델 및 평가지표는 변경하지 않는다.
3. 1차 비교는 OOF Macro F1 `0.34452`를 기준으로 한다.
4. 성능이 개선된 전처리만 동일 fold의 LightGBM으로 2차 검증한다.
5. 점수가 비슷한 최종 후보는 seed 42, 52, 62의 반복 5-fold로 확인한다.
6. OOF 예측 및 확률 파일은 Git에 커밋하지 않는다.

---

## 관련 파일

```text
common/starter_preprocess.py
common/preprocessing_benchmark.py
experiments/SDH/exp_002_baseline/experiment.ipynb
experiments/SDH/exp_002_baseline/results/metrics.json
experiments/SDH/exp_002_baseline/results/metrics_lightgbm_seed42.json
experiments/SDH/exp_002_baseline/results/metrics_logistic_seeds_42_52_62.json
```
