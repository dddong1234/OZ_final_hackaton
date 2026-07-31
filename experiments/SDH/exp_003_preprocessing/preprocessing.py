"""Leakage-safe preprocessing candidates for the exp_003 benchmark."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


_AMINO_ACID_CHANGE = re.compile(r"^[A-Z*](?:\d+)[A-Z*]$")


def _tokens(value: Any, fill_value: str) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == fill_value:
        return []
    return text.split()


def _mutation_type(token: str) -> str:
    lowered = token.lower()
    if "fs" in lowered:
        return "frameshift"
    if "*" in token:
        return "nonsense"
    if _AMINO_ACID_CHANGE.fullmatch(token):
        return "synonymous" if token[0] == token[-1] else "missense"
    return "other"


class MutationFeatureTransformer(TransformerMixin, BaseEstimator):
    """Encode gene mutations and optionally append row-level mutation features.

    All learned choices, including retained genes and hotspot tokens, are
    determined in ``fit``. The public benchmark clones and fits this transformer
    independently inside each training fold.
    """

    def __init__(
        self,
        *,
        min_gene_count: int = 0,
        include_gene_burden: bool = False,
        include_token_burden: bool = False,
        include_mutation_type_counts: bool = False,
        hotspot_count: int = 0,
        fill_value: str = "WT",
        dtype: str = "float32",
    ) -> None:
        self.min_gene_count = min_gene_count
        self.include_gene_burden = include_gene_burden
        self.include_token_burden = include_token_burden
        self.include_mutation_type_counts = include_mutation_type_counts
        self.hotspot_count = hotspot_count
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> MutationFeatureTransformer:
        del y
        self._validate_input(X)
        if self.min_gene_count < 0:
            raise ValueError("min_gene_count는 0 이상이어야 합니다.")
        if self.hotspot_count < 0:
            raise ValueError("hotspot_count는 0 이상이어야 합니다.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        normalized = X.reindex(columns=self.feature_names_in_).fillna(
            self.fill_value
        )
        mutation_counts = normalized.ne(self.fill_value).sum(axis=0)
        threshold = max(1, self.min_gene_count)
        if self.min_gene_count == 0:
            self.selected_gene_names_ = self.feature_names_in_.copy()
        else:
            self.selected_gene_names_ = np.asarray(
                mutation_counts[mutation_counts >= threshold].index,
                dtype=object,
            )
        if len(self.selected_gene_names_) == 0:
            raise ValueError("조건을 만족하는 유전자 피처가 없습니다.")

        token_counts: Counter[str] = Counter()
        if self.hotspot_count:
            for value in normalized.to_numpy(dtype=object).ravel():
                token_counts.update(_tokens(value, self.fill_value))
        ordered_tokens = sorted(
            token_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        self.hotspot_tokens_ = np.asarray(
            [token for token, _ in ordered_tokens[: self.hotspot_count]],
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("transform 전에 fit을 실행해야 합니다.")
        self._validate_input(X)
        normalized = X.reindex(columns=self.feature_names_in_).fillna(
            self.fill_value
        )
        output = normalized.reindex(
            columns=self.selected_gene_names_
        ).ne(self.fill_value).astype(self.dtype)

        needs_tokens = (
            self.include_token_burden
            or self.include_mutation_type_counts
            or len(self.hotspot_tokens_) > 0
        )
        gene_count = normalized.ne(self.fill_value).sum(axis=1).to_numpy()
        if self.include_gene_burden:
            output["log1p_gene_burden"] = np.log1p(gene_count).astype(
                self.dtype
            )

        if needs_tokens:
            token_rows = [
                [
                    token
                    for value in row
                    for token in _tokens(value, self.fill_value)
                ]
                for row in normalized.to_numpy(dtype=object)
            ]
            if self.include_token_burden:
                output["log1p_token_burden"] = np.log1p(
                    np.fromiter(
                        (len(tokens) for tokens in token_rows),
                        dtype=np.int32,
                        count=len(token_rows),
                    )
                ).astype(self.dtype)

            if self.include_mutation_type_counts:
                type_names = (
                    "synonymous",
                    "missense",
                    "nonsense",
                    "frameshift",
                    "other",
                )
                type_matrix = np.zeros(
                    (len(token_rows), len(type_names)),
                    dtype=self.dtype,
                )
                type_positions = {
                    name: position for position, name in enumerate(type_names)
                }
                for row_position, tokens in enumerate(token_rows):
                    counts = Counter(_mutation_type(token) for token in tokens)
                    for name, count in counts.items():
                        type_matrix[
                            row_position, type_positions[name]
                        ] = np.log1p(count)
                for position, name in enumerate(type_names):
                    output[f"log1p_{name}_count"] = type_matrix[:, position]

            for hotspot in self.hotspot_tokens_:
                output[f"hotspot__{hotspot}"] = np.fromiter(
                    (hotspot in tokens for tokens in token_rows),
                    dtype=np.int8,
                    count=len(token_rows),
                ).astype(self.dtype)

        return output

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MutationFeatureTransformer는 pandas DataFrame 입력이 필요합니다."
            )
        if X.columns.duplicated().any():
            raise ValueError("입력 feature에 중복 컬럼이 있습니다.")


def make_preprocessing_candidates() -> dict[str, MutationFeatureTransformer]:
    """Return the ten exp_003 preprocessing cases in execution order."""

    common_burdens = {
        "min_gene_count": 1,
        "include_gene_burden": True,
        "include_token_burden": True,
    }
    return {
        "case_01_wt_binary": MutationFeatureTransformer(),
        "case_02_remove_constant": MutationFeatureTransformer(
            min_gene_count=1
        ),
        "case_03_gene_burden": MutationFeatureTransformer(
            min_gene_count=1,
            include_gene_burden=True,
        ),
        "case_04_token_burden": MutationFeatureTransformer(
            min_gene_count=1,
            include_token_burden=True,
        ),
        "case_05_both_burdens": MutationFeatureTransformer(
            **common_burdens
        ),
        "case_06_mutation_types": MutationFeatureTransformer(
            **common_burdens,
            include_mutation_type_counts=True,
        ),
        "case_07_min_count_3": MutationFeatureTransformer(
            **{**common_burdens, "min_gene_count": 3}
        ),
        "case_08_min_count_5": MutationFeatureTransformer(
            **{**common_burdens, "min_gene_count": 5}
        ),
        "case_09_min_count_10": MutationFeatureTransformer(
            **{**common_burdens, "min_gene_count": 10}
        ),
        "case_10_hotspot_top50": MutationFeatureTransformer(
            **common_burdens,
            hotspot_count=50,
        ),
    }
