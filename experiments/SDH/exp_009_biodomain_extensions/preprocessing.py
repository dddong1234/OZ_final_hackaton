"""Train-only A/S extensions for the SDH exp_007 functional-full features."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from experiments.SDH.exp_007_fe_combinations.preprocessing import (
    CombinedMutationTransformer,
)
from experiments.moon.exp_006_train_only_eda.variant_features import (
    event_table,
    normalise_values,
)


AA = tuple("ACDEFGHIKLMNPQRSTVWY")
POSITION_BINS = ((1, 50), (51, 100), (101, 250), (251, 500), (501, 1000), (1001, None))
SUBSTITUTION = re.compile(r"^([A-Z])(-?\d+)([A-Z])$")


def _event_frame(X: pd.DataFrame) -> pd.DataFrame:
    values = normalise_values(X, list(X.columns))
    events = event_table(values)
    if events.empty:
        return pd.DataFrame(
            columns=["sample", "gene", "event", "event_type", "ref", "position", "alt"]
        )
    extracted = events["event"].str.extract(SUBSTITUTION)
    events = events.copy()
    events["ref"] = extracted[0]
    events["position"] = pd.to_numeric(extracted[1], errors="coerce")
    events["alt"] = extracted[2]
    return events


def _count_by(
    events: pd.DataFrame,
    index: pd.Index,
    column: str,
    keys: tuple[str, ...],
    prefix: str,
) -> pd.DataFrame:
    columns = [f"{prefix}{key}" for key in keys]
    if events.empty:
        return pd.DataFrame(0, index=index, columns=columns, dtype="float32")
    counts = pd.crosstab(events["sample"], events[column]).reindex(index=index, fill_value=0)
    return pd.DataFrame(
        {
            f"{prefix}{key}": counts[key].astype("float32") if key in counts else 0.0
            for key in keys
        },
        index=index,
    )


class ProteinNotationExtensionTransformer(BaseEstimator, TransformerMixin):
    """Append A/S blocks to the fold-fitted exp_007 functional-full base."""

    def __init__(
        self,
        *,
        include_a_ref_alt: bool = False,
        include_a_pair: bool = False,
        include_a_position: bool = False,
        include_s_count_topology: bool = False,
        include_s_distribution: bool = False,
    ) -> None:
        self.include_a_ref_alt = include_a_ref_alt
        self.include_a_pair = include_a_pair
        self.include_a_position = include_a_position
        self.include_s_count_topology = include_s_count_topology
        self.include_s_distribution = include_s_distribution

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.base_ = CombinedMutationTransformer(
            include_multi_mutated_gene_burden=True,
            include_truncating_gene_flags=True,
            recurrent_missense_min_count=5,
            exclude_hotspot_from_recurrent=True,
        ).fit(X, y)
        frame = self._extra_frame(X)
        self.output_columns_ = frame.columns[frame.nunique(dropna=False).gt(1)].tolist()
        self.n_features_in_ = len(self.base_.transform(X).columns) + len(self.output_columns_)
        return self

    def _extra_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        events = _event_frame(X)
        blocks: list[pd.DataFrame] = []
        substitutions = events.dropna(subset=["ref", "alt", "position"])

        if self.include_a_ref_alt and not substitutions.empty:
            blocks.append(_count_by(substitutions, X.index, "ref", AA, "A__ref_"))
            blocks.append(_count_by(substitutions, X.index, "alt", AA, "A__alt_"))
        if self.include_a_pair:
            pairs = tuple(f"{ref}>{alt}" for ref in AA for alt in AA if ref != alt)
            substitutions = substitutions.copy()
            substitutions["pair"] = substitutions["ref"] + ">" + substitutions["alt"]
            blocks.append(_count_by(substitutions, X.index, "pair", pairs, "A__pair_"))
        if self.include_a_position:
            position_block: dict[str, pd.Series] = {}
            for lower, upper in POSITION_BINS:
                suffix = f"{lower}_{upper}" if upper is not None else f"{lower}_plus"
                mask = substitutions["position"].ge(lower)
                if upper is not None:
                    mask &= substitutions["position"].le(upper)
                position_block[f"A__position_{suffix}_count"] = (
                    mask.groupby(substitutions["sample"]).sum().reindex(X.index, fill_value=0)
                )
            blocks.append(pd.DataFrame(position_block, index=X.index, dtype="float32"))

        if self.include_s_count_topology or self.include_s_distribution:
            if events.empty:
                empty_index = pd.MultiIndex.from_arrays(
                    [[], []], names=["sample", "gene"]
                )
                per_gene = pd.DataFrame(
                    columns=["event_count", "type_count"], index=empty_index
                )
            else:
                per_gene = events.groupby(["sample", "gene"], observed=True).agg(
                    event_count=("event", "size"), type_count=("event_type", "nunique")
                )
        if self.include_s_count_topology:
            structure: dict[str, pd.Series] = {}
            for label, mask in (
                ("one_event_gene_count", per_gene.event_count.eq(1)),
                ("two_event_gene_count", per_gene.event_count.eq(2)),
                ("three_or_more_event_gene_count", per_gene.event_count.ge(3)),
                ("multi_notation_type_gene_count", per_gene.type_count.ge(2)),
            ):
                structure[f"S__{label}"] = mask.groupby(level="sample").sum().reindex(X.index, fill_value=0)
            structure["S__max_events_in_one_gene"] = per_gene.event_count.groupby(level="sample").max().reindex(X.index, fill_value=0)
            blocks.append(pd.DataFrame(structure, index=X.index, dtype="float32"))
        if self.include_s_distribution:
            type_counts = pd.crosstab(events["sample"], events["event_type"]).reindex(X.index, fill_value=0)
            total = type_counts.sum(axis=1)
            proportions = type_counts.div(total.replace(0, np.nan), axis=0).fillna(0)
            blocks.append(pd.DataFrame({
                "S__notation_type_diversity": type_counts.gt(0).sum(axis=1),
                "S__notation_type_entropy": -(proportions.where(proportions.gt(0), 1.0) * np.log(proportions.where(proportions.gt(0), 1.0))).sum(axis=1),
                "S__dominant_notation_type_share": proportions.max(axis=1),
            }, index=X.index, dtype="float32"))
        return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "base_"):
            raise RuntimeError("transform 전에 fit을 실행해야 합니다.")
        base = self.base_.transform(X)
        extra = self._extra_frame(X).reindex(columns=self.output_columns_, fill_value=0)
        return pd.concat([base, extra], axis=1)


def make_candidates() -> dict[str, ProteinNotationExtensionTransformer]:
    return {
        "case_01_functional_full_reference": ProteinNotationExtensionTransformer(),
        "case_02_plus_A_all": ProteinNotationExtensionTransformer(
            include_a_ref_alt=True, include_a_pair=True, include_a_position=True
        ),
        "case_03_plus_S_all": ProteinNotationExtensionTransformer(
            include_s_count_topology=True, include_s_distribution=True
        ),
        "case_04_plus_A_plus_S": ProteinNotationExtensionTransformer(
            include_a_ref_alt=True, include_a_pair=True, include_a_position=True,
            include_s_count_topology=True, include_s_distribution=True,
        ),
        "case_05_plus_A_ref_alt": ProteinNotationExtensionTransformer(
            include_a_ref_alt=True
        ),
        "case_06_plus_A_pair": ProteinNotationExtensionTransformer(
            include_a_pair=True
        ),
        "case_07_plus_A_position": ProteinNotationExtensionTransformer(
            include_a_position=True
        ),
        "case_08_plus_S_count_topology": ProteinNotationExtensionTransformer(
            include_s_count_topology=True
        ),
        "case_09_plus_S_distribution": ProteinNotationExtensionTransformer(
            include_s_distribution=True
        ),
    }
