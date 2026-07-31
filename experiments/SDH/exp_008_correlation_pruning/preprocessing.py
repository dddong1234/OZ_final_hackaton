"""Correlation pruning layered on the exp_007 functional feature set."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone


FUNCTIONAL_PREFIXES = ("truncating__", "recurrent_missense__")


class CorrelationPrunedTransformer(TransformerMixin, BaseEstimator):
    """Fit a feature builder, then learn removable columns on fold-train only."""

    def __init__(
        self,
        base_transformer: Any,
        *,
        threshold: float | None = None,
        exact_duplicates_only: bool = False,
        block_aware: bool = False,
    ) -> None:
        self.base_transformer = base_transformer
        self.threshold = threshold
        self.exact_duplicates_only = exact_duplicates_only
        self.block_aware = block_aware

    def fit(self, X: pd.DataFrame, y: Any = None):
        if self.threshold is not None and not 0 < self.threshold <= 1:
            raise ValueError("threshold는 0보다 크고 1 이하여야 합니다.")
        self.base_transformer_ = clone(self.base_transformer)
        transformed = self.base_transformer_.fit_transform(X, y)
        if not isinstance(transformed, pd.DataFrame):
            transformed = pd.DataFrame(transformed, index=X.index)

        all_columns = list(transformed.columns)
        functional = [
            column for column in all_columns
            if str(column).startswith(FUNCTIONAL_PREFIXES)
        ]
        dropped: set[str] = set()

        if self.exact_duplicates_only and functional:
            duplicate_mask = transformed[functional].T.duplicated(keep="first")
            dropped.update(duplicate_mask.index[duplicate_mask].tolist())

        if self.threshold is not None and functional:
            remaining = [column for column in functional if column not in dropped]
            correlation = transformed[remaining].corr().abs().to_numpy()
            for later in range(1, len(remaining)):
                if remaining[later] in dropped:
                    continue
                earlier_values = correlation[:later, later]
                earlier_kept = np.asarray(
                    [remaining[i] not in dropped for i in range(later)]
                )
                if np.any((earlier_values >= self.threshold) & earlier_kept):
                    dropped.add(remaining[later])

            if self.block_aware:
                for column in remaining:
                    if column in dropped:
                        continue
                    gene, event = self._gene_event(column)
                    comparison_columns = []
                    if gene in transformed.columns:
                        comparison_columns.append(gene)
                    hotspot = f"hotspot__{event}" if event else None
                    if hotspot in transformed.columns:
                        comparison_columns.append(hotspot)
                    if comparison_columns:
                        correlations = transformed[comparison_columns].corrwith(
                            transformed[column]
                        ).abs()
                        if correlations.ge(self.threshold).any():
                            dropped.add(column)

        self.dropped_columns_ = tuple(
            column for column in all_columns if column in dropped
        )
        self.kept_columns_ = tuple(
            column for column in all_columns if column not in dropped
        )
        self.n_features_before_ = len(all_columns)
        self.n_features_after_ = len(self.kept_columns_)
        return self

    @staticmethod
    def _gene_event(column: str) -> tuple[str | None, str | None]:
        if column.startswith("truncating__"):
            return column.removeprefix("truncating__"), None
        match = re.fullmatch(r"recurrent_missense__(.+?)__(.+)", column)
        return match.groups() if match else (None, None)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "kept_columns_"):
            raise RuntimeError("transform 전에 fit을 실행해야 합니다.")
        transformed = self.base_transformer_.transform(X)
        return transformed.reindex(columns=self.kept_columns_, fill_value=0)


def make_pruning_candidates(base_transformer: Any) -> dict[str, Any]:
    return {
        "case_01_functional_full_reference": clone(base_transformer),
        "case_02_exact_duplicate_drop": CorrelationPrunedTransformer(
            base_transformer, exact_duplicates_only=True
        ),
        "case_03_corr_099": CorrelationPrunedTransformer(
            base_transformer, threshold=0.99
        ),
        "case_04_corr_095": CorrelationPrunedTransformer(
            base_transformer, threshold=0.95
        ),
        "case_05_corr_090": CorrelationPrunedTransformer(
            base_transformer, threshold=0.90
        ),
        "case_06_corr_095_block_aware": CorrelationPrunedTransformer(
            base_transformer, threshold=0.95, block_aware=True
        ),
    }
