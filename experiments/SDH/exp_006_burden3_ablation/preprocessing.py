"""Third-burden feature candidate built on SDH's best preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from experiments.SDH.exp_003_preprocessing.preprocessing import (
    MutationFeatureTransformer,
    _tokens,
)


class Burden3Transformer(TransformerMixin, BaseEstimator):
    """Append the number of genes containing multiple mutation tokens.

    The base block is SDH's exp_004/005 winner:
    gene binary + two burdens + mutation types + fold-train hotspot 50.
    """

    def __init__(
        self,
        *,
        include_multi_mutated_gene_burden: bool = False,
        hotspot_count: int = 50,
        fill_value: str = "WT",
        dtype: str = "float32",
    ) -> None:
        self.include_multi_mutated_gene_burden = (
            include_multi_mutated_gene_burden
        )
        self.hotspot_count = hotspot_count
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y: Any = None) -> Burden3Transformer:
        self.base_transformer_ = MutationFeatureTransformer(
            min_gene_count=1,
            include_gene_burden=True,
            include_token_burden=True,
            include_mutation_type_counts=True,
            hotspot_count=self.hotspot_count,
            fill_value=self.fill_value,
            dtype=self.dtype,
        )
        self.base_transformer_.fit(X, y)
        self.feature_names_in_ = self.base_transformer_.feature_names_in_
        self.n_features_in_ = self.base_transformer_.n_features_in_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "base_transformer_"):
            raise RuntimeError("transform 전에 fit을 실행해야 합니다.")

        output = self.base_transformer_.transform(X)
        if not self.include_multi_mutated_gene_burden:
            return output

        normalized = X.reindex(columns=self.feature_names_in_).fillna(
            self.fill_value
        )
        multi_mutated_gene_count = np.fromiter(
            (
                sum(
                    len(_tokens(value, self.fill_value)) >= 2
                    for value in row
                )
                for row in normalized.to_numpy(dtype=object)
            ),
            dtype=np.int32,
            count=len(normalized),
        )
        output["log1p_multi_mutated_gene_burden"] = np.log1p(
            multi_mutated_gene_count
        ).astype(self.dtype)
        return output


def make_candidates() -> dict[str, Burden3Transformer]:
    """Return the reference and one-variable burden3 candidate."""

    return {
        "case_01_two_burdens_reference": Burden3Transformer(
            include_multi_mutated_gene_burden=False
        ),
        "case_02_three_burdens": Burden3Transformer(
            include_multi_mutated_gene_burden=True
        ),
    }
