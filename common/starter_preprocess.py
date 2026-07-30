from __future__ import annotations

import pandas as pd


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
