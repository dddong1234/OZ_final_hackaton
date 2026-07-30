# 공용 전처리 벤치마크

팀원이 sklearn Transformer 또는 전처리 `Pipeline`만 구현하면 동일한 모델과 fold에서 성능을 비교할 수 있는 공용 인터페이스다.

## 고정 평가 조건

### 1차 벤치마크

- 검증: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 모델: 고정된 `LogisticRegression`
- 주 지표: 전체 OOF 예측의 Macro F1
- 보조 지표: OOF Accuracy

```python
LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
```

### 2차 검증

- fold: 1차 벤치마크와 동일
- 모델: 고정된 `LGBMClassifier`
- 목적: 1차에서 선별된 전처리의 비선형 효과 확인

### 최종 후보 확인

`confirmation=True`로 실행하면 동일한 5-fold 검증을 seed `42`, `52`, `62`에서 반복한다.

---

## 역할 구분

팀원은 sklearn 호환 전처리 객체만 작성한다.

```text
팀원 구현
└── Transformer 또는 preprocessing Pipeline

공용 벤치마크
├── ID와 SUBCLASS 제외
├── StratifiedKFold-5 생성
├── fold마다 preprocessor clone
├── Pipeline(preprocessor, fixed model) 구성
├── fold train에서만 fit
├── validation predict
├── OOF 예측과 클래스별 확률 정렬
└── Macro F1, Accuracy 및 실행 시간 기록
```

공용 벤치마크 내부 구조:

```python
Pipeline(
    [
        ("preprocessing", clone(preprocessor)),
        ("model", fixed_benchmark_model),
    ]
)
```

따라서 validation 데이터에는 `transform`만 적용되며 전처리 학습 누수가 방지된다.

---

## 기본 baseline 실행

공용 baseline은 결측값을 `WT`로 처리하고 `WT=0`, 변이=1로 변환한다.

```python
import pandas as pd

from common.preprocessing_benchmark import run_preprocessing_benchmark
from common.starter_preprocess import make_baseline_preprocessor

train = pd.read_csv("data/raw/train.csv", low_memory=False)
preprocessor = make_baseline_preprocessor()

result = run_preprocessing_benchmark(
    train,
    preprocessor,
    experiment_id="common-wt-binary-baseline",
    preprocessing_name="WT/variant binary",
)

display(result.summary_frame())
display(result.fold_metrics)
```

---

## 기존 sklearn 전처리 조합

sklearn에서 제공하는 Transformer는 별도 wrapper 없이 `Pipeline`으로 조합한다.

### WT 이진화 + 상수 컬럼 제거

```python
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from common.starter_preprocess import WTBinaryEncoder


preprocessor = Pipeline(
    [
        ("wt_binary", WTBinaryEncoder()),
        ("remove_constant", VarianceThreshold(threshold=0.0)),
    ]
)
```

### WT 이진화 + chi-square feature selection

```python
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2
from sklearn.pipeline import Pipeline

from common.starter_preprocess import WTBinaryEncoder


preprocessor = Pipeline(
    [
        ("wt_binary", WTBinaryEncoder()),
        ("remove_constant", VarianceThreshold(threshold=0.0)),
        ("select_k_best", SelectKBest(score_func=chi2, k=1000)),
    ]
)
```

`SelectKBest`의 `fit(X, y)`에는 fold train의 `X`, `y`만 전달된다.

### WT 이진화 + TruncatedSVD

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from common.starter_preprocess import WTBinaryEncoder


preprocessor = Pipeline(
    [
        ("wt_binary", WTBinaryEncoder()),
        ("remove_constant", VarianceThreshold(threshold=0.0)),
        (
            "svd",
            TruncatedSVD(
                n_components=200,
                random_state=42,
            ),
        ),
    ]
)
```

희소 데이터의 차원 축소에는 중심화를 요구하는 PCA보다 `TruncatedSVD`가 적합하다.

### WT 이진화 + sparse scaling

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.starter_preprocess import WTBinaryEncoder


preprocessor = Pipeline(
    [
        ("wt_binary", WTBinaryEncoder()),
        ("scaler", StandardScaler(with_mean=False)),
    ]
)
```

희소 행렬에서는 `StandardScaler(with_mean=False)`를 사용한다.

### 원본 변이 문자열 One-hot encoding

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


preprocessor = Pipeline(
    [
        (
            "one_hot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=5,
                dtype=np.float32,
            ),
        ),
    ]
)
```

유전자별 변이 문자열 종류가 많으므로 `min_frequency` 없이 전체 One-hot encoding을 적용하면 차원이 크게 증가할 수 있다.

`OrdinalEncoder`는 변이 문자열에 실제로 존재하지 않는 순서 관계를 부여하므로 기본 비교에서는 권장하지 않는다.

---

## 커스텀 Transformer 구현

sklearn에 없는 전처리는 `BaseEstimator`, `TransformerMixin`을 상속해 구현한다.

규칙:

1. 실험 파라미터는 `__init__` 인자로 받는다.
2. `__init__`에서는 인자를 같은 이름의 속성으로 저장만 한다.
3. 데이터에서 학습한 속성은 이름 끝에 `_`를 붙인다.
4. `fit`은 반드시 `self`를 반환한다.
5. `transform`은 학습된 속성을 사용하고 새로운 기준을 학습하지 않는다.
6. 출력은 2차원 DataFrame, ndarray 또는 sparse matrix여야 한다.

### 변이 빈도 선택 + `log1p(TMB)`

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MutationFrequencyTMB(
    TransformerMixin,
    BaseEstimator,
):
    def __init__(
        self,
        *,
        min_mutation_count=5,
        add_tmb=True,
    ):
        self.min_mutation_count = min_mutation_count
        self.add_tmb = add_tmb

    def fit(self, X, y=None):
        del y

        if not isinstance(X, pd.DataFrame):
            raise TypeError("pandas DataFrame 입력이 필요합니다.")

        mutation_mask = X.fillna("WT").ne("WT")
        mutation_counts = mutation_mask.sum(axis=0)

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.selected_feature_names_ = np.asarray(
            mutation_counts[
                mutation_counts >= self.min_mutation_count
            ].index,
            dtype=object,
        )

        return self

    def transform(self, X):
        mutation_mask = (
            X.reindex(columns=self.feature_names_in_)
            .fillna("WT")
            .ne("WT")
        )

        features = (
            mutation_mask
            .reindex(columns=self.selected_feature_names_)
            .astype("float32")
        )

        if self.add_tmb:
            features["log1p_TMB"] = np.log1p(
                mutation_mask.sum(axis=1)
            ).astype("float32")

        return features
```

사용:

```python
preprocessor = MutationFrequencyTMB(
    min_mutation_count=5,
    add_tmb=True,
)

result = run_preprocessing_benchmark(
    train,
    preprocessor,
    experiment_id="sdh-exp-003-frequency-tmb",
    preprocessing_name="mutation count >= 5 + log1p(TMB)",
)

display(result.summary_frame())
display(result.fold_metrics)
```

커스텀 Transformer도 fold마다 `sklearn.base.clone`으로 새 객체가 생성된다.

---

## 여러 전처리 단계 조합

커스텀 Transformer와 sklearn Transformer를 한 Pipeline에 조합할 수 있다.

```python
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline


preprocessor = Pipeline(
    [
        (
            "mutation_features",
            MutationFrequencyTMB(
                min_mutation_count=5,
                add_tmb=True,
            ),
        ),
        (
            "select_k_best",
            SelectKBest(
                score_func=chi2,
                k=1000,
            ),
        ),
    ]
)
```

각 단계는 앞 단계의 출력을 받아 순서대로 `fit_transform` 또는 `transform`된다.

---

## LightGBM 2차 검증

1차 Logistic Regression에서 선별된 전처리 객체를 그대로 전달한다.

```python
lgbm_result = run_preprocessing_benchmark(
    train,
    preprocessor,
    experiment_id="sdh-exp-003-frequency-tmb",
    preprocessing_name="mutation count >= 5 + log1p(TMB)",
    model="lightgbm",
)

display(lgbm_result.summary_frame())
```

---

## 반복 5-fold 확인

점수가 비슷한 최종 후보에만 사용한다. 총 15번 학습한다.

```python
confirmed = run_preprocessing_benchmark(
    train,
    preprocessor,
    experiment_id="sdh-exp-003-frequency-tmb",
    preprocessing_name="mutation count >= 5 + log1p(TMB)",
    confirmation=True,
)

display(confirmed.summary_frame())
display(confirmed.run_metrics)
```

---

## 결과 객체

- `result.summary`: 실험, 전처리, 검증, 모델, 주요 지표
- `result.run_metrics`: CV seed별 OOF 지표
- `result.fold_metrics`: fold별 지표, feature 수, 실행 시간
- `result.oof_predictions`: ID, 실제 타깃, seed, fold, 예측값, 클래스별 확률
- `result.summary_frame()`: 노트북 표시용 한 행 요약
- `result.save_metrics(path)`: OOF 확률을 제외한 경량 JSON 저장

```python
result.save_metrics(
    "experiments/SDH/exp_003_frequency_tmb/results/metrics.json"
)
```

OOF 예측과 확률 파일은 Git에 커밋하지 않는다.

단일 seed에서는 seed 간 표준편차가 정의되지 않으므로 `oof_f1_macro_std`와 `oof_accuracy_std`는 `null`이다. 이때 fold별 평균과 표본 표준편차를 함께 확인한다.

---

## 비교 규칙

1. 전처리 Pipeline 외에 모델, fold, seed, 지표를 변경하지 않는다.
2. 1차 순위는 `oof_f1_macro_mean`으로 비교한다.
3. 단일 fold 점수나 가장 좋은 fold 점수로 순위를 정하지 않는다.
4. 최종 후보는 `confirmation=True`로 seed 간 평균과 표준편차를 확인한다.
5. Transformer의 `fit`에서 validation 또는 전체 train을 참조하지 않는다.
6. 좋은 전처리는 같은 fold의 LightGBM으로 2차 검증한다.
