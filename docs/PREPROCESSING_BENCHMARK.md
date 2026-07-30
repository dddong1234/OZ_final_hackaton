# 공용 전처리 벤치마크

팀원이 전처리만 구현하고 동일한 모델과 fold에서 성능을 비교하기 위한 공용 인터페이스다.

## 고정 조건

### 1차 벤치마크

- 검증: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 모델: `LogisticRegression`
- 모델 설정:
  - `solver="lbfgs"`
  - `max_iter=1000`
  - `class_weight="balanced"`
  - `random_state=42`
- 주 지표: 전체 OOF 예측의 Macro F1
- 보조 지표: OOF Accuracy

### 2차 검증

- fold: 1차 벤치마크와 동일
- 모델: 고정 파라미터의 `LGBMClassifier`
- 목적: 1차에서 선별된 전처리의 비선형 효과 확인

### 최종 후보 확인

`confirmation=True`로 실행하면 동일한 5-fold 검증을 seed `42`, `52`, `62`에서 반복한다.

## 노트북 사용법

### 기존 공용 WT 이진 전처리 평가

```python
import pandas as pd

from common.preprocessing_benchmark import run_preprocessing_benchmark
from common.starter_preprocess import fit, transform

train = pd.read_csv("data/raw/train.csv", low_memory=False)

result = run_preprocessing_benchmark(
    train,
    fit,
    transform,
    experiment_id="sdh-exp-002-wt-binary",
    preprocessing_name="WT/variant binary",
)

display(result.summary_frame())
display(result.fold_metrics)
```

### 노트북에서 전처리 직접 구현

```python
import numpy as np


def fit_preprocessing(train_fold, target_column, id_column):
    feature_columns = [
        column
        for column in train_fold.columns
        if column not in {target_column, id_column}
    ]
    active_columns = [
        column
        for column in feature_columns
        if train_fold[column].fillna("WT").ne("WT").any()
    ]
    return {"feature_columns": active_columns}


def transform_preprocessing(dataframe, state, target_column, id_column):
    del target_column, id_column
    mutation = (
        dataframe[state["feature_columns"]]
        .fillna("WT")
        .ne("WT")
        .astype("float32")
    )
    mutation["log1p_TMB"] = np.log1p(mutation.sum(axis=1))
    return mutation
```

평가:

```python
result = run_preprocessing_benchmark(
    train,
    fit_preprocessing,
    transform_preprocessing,
    experiment_id="sdh-exp-003-active-gene-tmb",
    preprocessing_name="active genes + log1p(TMB)",
)

display(result.summary_frame())
```

`fit_preprocessing`에는 매 fold의 학습 부분만 전달된다. feature selection, scaler, encoding처럼 학습이 필요한 기준은 반드시 이 함수 안에서 계산하고 state로 반환한다.

## LightGBM 2차 검증

1차 Logistic Regression에서 선별된 전처리만 실행한다.

```python
lgbm_result = run_preprocessing_benchmark(
    train,
    fit_preprocessing,
    transform_preprocessing,
    experiment_id="sdh-exp-003-active-gene-tmb",
    preprocessing_name="active genes + log1p(TMB)",
    model="lightgbm",
)

display(lgbm_result.summary_frame())
```

## 반복 5-fold 확인

점수가 비슷한 최종 후보에만 사용한다. 총 15번 학습하므로 기본 벤치마크보다 오래 걸린다.

```python
confirmed = run_preprocessing_benchmark(
    train,
    fit_preprocessing,
    transform_preprocessing,
    experiment_id="sdh-exp-003-active-gene-tmb",
    preprocessing_name="active genes + log1p(TMB)",
    confirmation=True,
)

display(confirmed.summary_frame())
display(confirmed.run_metrics)
```

## 결과 객체

- `result.summary`: 실험, 검증, 모델, 주요 지표 요약
- `result.run_metrics`: CV seed별 OOF 지표
- `result.fold_metrics`: fold별 지표, feature 수, 실행 시간
- `result.oof_predictions`: ID, 실제 타깃, seed, fold, 예측값, 클래스별 확률
- `result.summary_frame()`: 노트북 표시용 한 행 요약
- `result.save_metrics(path)`: 확률 파일을 제외한 경량 `metrics.json` 저장

예시:

```python
result.save_metrics(
    "experiments/SDH/exp_003_active_gene_tmb/results/metrics.json"
)
```

OOF 확률은 Git에 커밋하지 않는다.

단일 seed 기본 실행에서는 seed 간 표준편차가 정의되지 않으므로
`oof_f1_macro_std`와 `oof_accuracy_std`는 `null`이다. 이때 실행 로그에는
전체 OOF 점수와 fold별 점수의 평균 ± 표본 표준편차를 구분해서 표시한다.
`confirmation=True`에서는 세 seed의 OOF 점수 평균 ± 표본 표준편차도 표시한다.

## 비교 규칙

1. 전처리 외에 모델, fold, seed, 지표를 변경하지 않는다.
2. 1차 순위는 `oof_f1_macro_mean`으로 비교한다.
3. 단일 fold 점수나 가장 좋은 fold 점수로 순위를 정하지 않는다.
4. 최종 후보는 `confirmation=True`로 평균과 표준편차를 확인한다.
5. 전처리 결과의 행 순서를 변경하지 않는다.
6. 좋은 전처리는 같은 fold의 LightGBM으로 2차 검증한다.
