"""Leakage-safe combinations of previously promising mutation features."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from experiments.SDH.exp_003_preprocessing.preprocessing import (
    MutationFeatureTransformer,
    _tokens,
)
from experiments.moon.exp_006_train_only_eda.variant_features import (
    TRUNCATING_TYPES,
    event_table,
    normalise_values,
)


class CombinedMutationTransformer(TransformerMixin, BaseEstimator):
    """Combine SDH summary/hotspot features with fold-fitted functional blocks."""

    def __init__(
        self,
        *,
        include_multi_mutated_gene_burden: bool = False,
        include_truncating_gene_flags: bool = False,
        recurrent_missense_min_count: int = 0,
        exclude_hotspot_from_recurrent: bool = False,
        hotspot_count: int = 50,
        min_gene_count: int = 1,
        fill_value: str = "WT",
        dtype: str = "float32",
    ) -> None:
        self.include_multi_mutated_gene_burden = (
            include_multi_mutated_gene_burden
        )
        self.include_truncating_gene_flags = include_truncating_gene_flags
        self.recurrent_missense_min_count = recurrent_missense_min_count
        self.exclude_hotspot_from_recurrent = exclude_hotspot_from_recurrent
        self.hotspot_count = hotspot_count
        self.min_gene_count = min_gene_count
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "CombinedMutationTransformer":
        if self.recurrent_missense_min_count not in (0,) and (
            self.recurrent_missense_min_count < 2
        ):
            raise ValueError("recurrent missense 최소 빈도는 0 또는 2 이상이어야 합니다.")

        self.base_transformer_ = MutationFeatureTransformer(
            min_gene_count=self.min_gene_count,
            include_gene_burden=True,
            include_token_burden=True,
            include_mutation_type_counts=True,
            hotspot_count=self.hotspot_count,
            fill_value=self.fill_value,
            dtype=self.dtype,
        ).fit(X, y)
        self.feature_names_in_ = self.base_transformer_.feature_names_in_
        self.n_features_in_ = self.base_transformer_.n_features_in_

        values = normalise_values(X, list(self.feature_names_in_))
        events = event_table(values)

        if self.include_truncating_gene_flags:
            self.truncating_genes_ = np.asarray(
                sorted(
                    events.loc[
                        events["event_type"].isin(TRUNCATING_TYPES),
                        "gene",
                    ].unique()
                ),
                dtype=object,
            )
        else:
            self.truncating_genes_ = np.asarray([], dtype=object)

        recurrent_pairs: list[tuple[str, str]] = []
        if self.recurrent_missense_min_count:
            missense = events.loc[
                events["event_type"].eq("MISSENSE"),
                ["gene", "event"],
            ]
            counts = (
                missense.value_counts()
                .rename("count")
                .reset_index()
                .sort_values(
                    ["count", "gene", "event"],
                    ascending=[False, True, True],
                )
            )
            selected = counts.loc[
                counts["count"].ge(self.recurrent_missense_min_count),
                ["gene", "event"],
            ]
            recurrent_pairs = list(
                selected.itertuples(index=False, name=None)
            )
            if self.exclude_hotspot_from_recurrent:
                hotspot_tokens = set(self.base_transformer_.hotspot_tokens_)
                recurrent_pairs = [
                    pair for pair in recurrent_pairs
                    if pair[1] not in hotspot_tokens
                ]
        self.recurrent_pairs_ = tuple(recurrent_pairs)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "base_transformer_"):
            raise RuntimeError("transform 전에 fit을 실행해야 합니다.")

        output = self.base_transformer_.transform(X)
        normalized = X.reindex(columns=self.feature_names_in_).fillna(
            self.fill_value
        )

        if self.include_multi_mutated_gene_burden:
            multi_count = np.fromiter(
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
                multi_count
            ).astype(self.dtype)

        if len(self.truncating_genes_) or self.recurrent_pairs_:
            values = normalise_values(X, list(self.feature_names_in_))
            events = event_table(values)
            extra_features: dict[str, np.ndarray] = {}

            if len(self.truncating_genes_):
                truncating = events.loc[
                    events["event_type"].isin(TRUNCATING_TYPES),
                    ["sample", "gene"],
                ].drop_duplicates()
                observed = set(
                    zip(truncating["sample"], truncating["gene"])
                )
                for gene in self.truncating_genes_:
                    extra_features[f"truncating__{gene}"] = np.fromiter(
                        ((index, gene) in observed for index in values.index),
                        dtype=np.int8,
                        count=len(values),
                    ).astype(self.dtype)

            if self.recurrent_pairs_:
                observed_pairs = set(
                    zip(events["sample"], events["gene"], events["event"])
                )
                for gene, event in self.recurrent_pairs_:
                    extra_features[
                        f"recurrent_missense__{gene}__{event}"
                    ] = np.fromiter(
                        (
                            (index, gene, event) in observed_pairs
                            for index in values.index
                        ),
                        dtype=np.int8,
                        count=len(values),
                    ).astype(self.dtype)

            if extra_features:
                output = pd.concat(
                    [
                        output,
                        pd.DataFrame(extra_features, index=output.index),
                    ],
                    axis=1,
                )
        return output


def make_candidates() -> dict[str, CombinedMutationTransformer]:
    """Return named cases; the notebook evaluates each case explicitly."""

    return {
        "case_01_burden2_types_hotspot50": CombinedMutationTransformer(),
        "case_02_burden3_types_hotspot50": CombinedMutationTransformer(
            include_multi_mutated_gene_burden=True,
        ),
        "case_03_burden3_types_hotspot100": CombinedMutationTransformer(
            include_multi_mutated_gene_burden=True,
            hotspot_count=100,
        ),
        "case_04_burden3_types_hotspot50_truncating": (
            CombinedMutationTransformer(
                include_multi_mutated_gene_burden=True,
                include_truncating_gene_flags=True,
            )
        ),
        "case_05_burden3_types_recurrent5": CombinedMutationTransformer(
            include_multi_mutated_gene_burden=True,
            hotspot_count=0,
            recurrent_missense_min_count=5,
        ),
        "case_06_burden3_types_hotspot50_recurrent_complement": (
            CombinedMutationTransformer(
                include_multi_mutated_gene_burden=True,
                recurrent_missense_min_count=5,
                exclude_hotspot_from_recurrent=True,
            )
        ),
        "case_07_burden3_types_hotspot50_functional_full": (
            CombinedMutationTransformer(
                include_multi_mutated_gene_burden=True,
                include_truncating_gene_flags=True,
                recurrent_missense_min_count=5,
                exclude_hotspot_from_recurrent=True,
            )
        ),
        "case_08_burden3_types_hotspot50_min10": (
            CombinedMutationTransformer(
                include_multi_mutated_gene_burden=True,
                min_gene_count=10,
            )
        ),
    }
