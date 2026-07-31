"""Preprocessing combinations derived from the exp_003 winner."""

from __future__ import annotations

from experiments.SDH.exp_003_preprocessing.preprocessing import (
    MutationFeatureTransformer,
)


def _candidate(
    *,
    min_gene_count: int = 1,
    hotspot_count: int = 0,
) -> MutationFeatureTransformer:
    return MutationFeatureTransformer(
        min_gene_count=min_gene_count,
        include_gene_burden=True,
        include_token_burden=True,
        include_mutation_type_counts=True,
        hotspot_count=hotspot_count,
    )


def make_combination_candidates() -> dict[str, MutationFeatureTransformer]:
    """Return exp_004 candidates in benchmark execution order."""

    return {
        "case_01_types_reference": _candidate(),
        "case_02_types_min10": _candidate(min_gene_count=10),
        "case_03_types_min15": _candidate(min_gene_count=15),
        "case_04_types_min20": _candidate(min_gene_count=20),
        "case_05_types_min30": _candidate(min_gene_count=30),
        "case_06_types_hotspot20": _candidate(hotspot_count=20),
        "case_07_types_hotspot50": _candidate(hotspot_count=50),
        "case_08_types_hotspot100": _candidate(hotspot_count=100),
        "case_09_types_min10_hotspot50": _candidate(
            min_gene_count=10,
            hotspot_count=50,
        ),
        "case_10_types_min20_hotspot50": _candidate(
            min_gene_count=20,
            hotspot_count=50,
        ),
    }
