from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import Any


class WTBinaryEncoder(TransformerMixin, BaseEstimator):
    """Encode WT as 0 and every recorded mutation as 1."""

    def __init__(
        self,
        *,
        fill_value: str = "WT",
        dtype: str = "float32",
    ) -> None:
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> WTBinaryEncoder:
        del y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WTBinaryEncoder는 pandas DataFrame 입력이 필요합니다.")
        if X.columns.duplicated().any():
            raise ValueError("입력 feature에 중복 컬럼이 있습니다.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("transform 전에 fit을 실행해야 합니다.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WTBinaryEncoder는 pandas DataFrame 입력이 필요합니다.")
        features = X.reindex(columns=self.feature_names_in_).fillna(
            self.fill_value
        )
        return features.ne(self.fill_value).astype(self.dtype)

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        del input_features
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("get_feature_names_out 전에 fit을 실행해야 합니다.")
        return self.feature_names_in_.copy()


def make_baseline_preprocessor() -> Pipeline:
    """Return the canonical sklearn preprocessing Pipeline."""

    return Pipeline(
        [
            ("wt_binary", WTBinaryEncoder()),
        ]
    )


def fit(
    train_df: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> dict[str, list[str]]:
    feature_columns = [
        column
        for column in train_df.columns
        if column not in {target_column, id_column}
    ]
    return {"feature_columns": feature_columns}


def transform(
    dataframe: pd.DataFrame,
    state: dict[str, list[str]],
    target_column: str,
    id_column: str,
) -> pd.DataFrame:
    del target_column, id_column
    features = dataframe.reindex(columns=state["feature_columns"]).fillna("WT")
    return features.ne("WT").astype("int8")
