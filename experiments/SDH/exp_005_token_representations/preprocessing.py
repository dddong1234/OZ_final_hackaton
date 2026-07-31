"""Fold-safe hashed mutation-token representations for exp_005."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher

from experiments.SDH.exp_003_preprocessing.preprocessing import (
    MutationFeatureTransformer,
    _mutation_type,
    _tokens,
)


_CODON_PATTERN = re.compile(r"^([A-Z*])(\d+)")
_VALID_REPRESENTATIONS = {"exact", "codon", "gene_type"}


class HashedMutationAppender(TransformerMixin, BaseEstimator):
    """Append fixed-dimensional mutation-token blocks to the exp_003 winner.

    Hashing does not learn a vocabulary from validation or test data. Every
    mutation feature is computed independently inside a row.
    """

    def __init__(
        self,
        *,
        representations: tuple[str, ...] = ("exact",),
        hash_dim: int = 8192,
        fill_value: str = "WT",
        dtype: str = "float32",
    ) -> None:
        self.representations = representations
        self.hash_dim = hash_dim
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> HashedMutationAppender:
        self._validate_input(X)
        unknown = sorted(set(self.representations) - _VALID_REPRESENTATIONS)
        if unknown:
            raise ValueError(f"지원하지 않는 representation: {unknown}")
        if not self.representations:
            raise ValueError("representations는 하나 이상이어야 합니다.")
        if self.hash_dim <= 0:
            raise ValueError("hash_dim은 양수여야 합니다.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.base_transformer_ = MutationFeatureTransformer(
            min_gene_count=1,
            include_gene_burden=True,
            include_token_burden=True,
            include_mutation_type_counts=True,
            fill_value=self.fill_value,
            dtype=self.dtype,
        )
        self.base_transformer_.fit(X, y)
        self.hashers_ = {
            representation: FeatureHasher(
                n_features=self.hash_dim,
                input_type="string",
                alternate_sign=False,
                dtype=np.dtype(self.dtype),
            )
            for representation in self.representations
        }
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        if not hasattr(self, "base_transformer_"):
            raise RuntimeError("transform 전에 fit을 실행해야 합니다.")
        self._validate_input(X)
        normalized = X.reindex(columns=self.feature_names_in_).fillna(
            self.fill_value
        )
        base = sparse.csr_matrix(
            self.base_transformer_.transform(normalized).to_numpy(
                dtype=self.dtype
            )
        )
        row_features = self._row_features(normalized)
        hashed_blocks = [
            self.hashers_[representation].transform(
                [
                    features[representation]
                    for features in row_features
                ]
            )
            for representation in self.representations
        ]
        return sparse.hstack(
            [base, *hashed_blocks],
            format="csr",
            dtype=np.dtype(self.dtype),
        )

    def _row_features(
        self,
        X: pd.DataFrame,
    ) -> list[dict[str, list[str]]]:
        columns = [str(column) for column in self.feature_names_in_]
        rows: list[dict[str, list[str]]] = []
        for row in X.to_numpy(dtype=object):
            features = {
                representation: []
                for representation in self.representations
            }
            for gene, value in zip(columns, row, strict=True):
                for token in _tokens(value, self.fill_value):
                    if "exact" in features:
                        features["exact"].append(
                            f"exact::{gene}::{token}"
                        )
                    if "codon" in features:
                        match = _CODON_PATTERN.match(token)
                        if match:
                            source_amino_acid, position = match.groups()
                            features["codon"].append(
                                f"codon::{gene}::"
                                f"{source_amino_acid}{position}"
                            )
                    if "gene_type" in features:
                        features["gene_type"].append(
                            f"type::{gene}::{_mutation_type(token)}"
                        )
            rows.append(features)
        return rows

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "HashedMutationAppender는 pandas DataFrame 입력이 필요합니다."
            )
        if X.columns.duplicated().any():
            raise ValueError("입력 feature에 중복 컬럼이 있습니다.")


def make_token_candidates() -> dict[str, Any]:
    """Return ten preprocessing-only candidates for exp_005."""

    reference = {
        "min_gene_count": 1,
        "include_gene_burden": True,
        "include_token_burden": True,
        "include_mutation_type_counts": True,
    }
    return {
        "case_01_types_reference": MutationFeatureTransformer(**reference),
        "case_02_types_hotspot50": MutationFeatureTransformer(
            **reference,
            hotspot_count=50,
        ),
        "case_03_exact_hash4096": HashedMutationAppender(
            representations=("exact",),
            hash_dim=4096,
        ),
        "case_04_exact_hash8192": HashedMutationAppender(
            representations=("exact",),
            hash_dim=8192,
        ),
        "case_05_exact_hash16384": HashedMutationAppender(
            representations=("exact",),
            hash_dim=16384,
        ),
        "case_06_codon_hash4096": HashedMutationAppender(
            representations=("codon",),
            hash_dim=4096,
        ),
        "case_07_codon_hash8192": HashedMutationAppender(
            representations=("codon",),
            hash_dim=8192,
        ),
        "case_08_codon_hash16384": HashedMutationAppender(
            representations=("codon",),
            hash_dim=16384,
        ),
        "case_09_gene_type_hash4096": HashedMutationAppender(
            representations=("gene_type",),
            hash_dim=4096,
        ),
        "case_10_all_hash4096": HashedMutationAppender(
            representations=("exact", "codon", "gene_type"),
            hash_dim=4096,
        ),
    }
