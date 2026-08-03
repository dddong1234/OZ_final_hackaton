"""Sequential champion-inspired FE additions for SDH exp010.

The candidates are cumulative:
functional full + A pair raw -> A pair log1p -> +S -> +train-only contrast
-> +train-only exact top-4.  Exact mutation names are never hard-coded; both
label-dependent blocks are learned inside each training fold.
"""

from __future__ import annotations

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
CONTRAST_PAIRS = (("KIRC", "KIPAN"), ("LGG", "GBMLGG"))


def _events(X: pd.DataFrame) -> pd.DataFrame:
    values = normalise_values(X, list(X.columns))
    return event_table(values)


def _count_by(
    events: pd.DataFrame,
    index: pd.Index,
    column: str,
    keys: tuple[str, ...],
    prefix: str,
    *,
    log1p: bool = False,
) -> pd.DataFrame:
    names = [f"{prefix}{key}" for key in keys]
    if events.empty:
        return pd.DataFrame(0.0, index=index, columns=names, dtype="float32")
    counts = pd.crosstab(events["sample"], events[column]).reindex(
        index=index, fill_value=0
    )
    output = pd.DataFrame(
        {
            f"{prefix}{key}": counts[key].astype("float32")
            if key in counts
            else 0.0
            for key in keys
        },
        index=index,
    )
    if log1p:
        output = np.log1p(output).astype("float32")
    return output


class ChampionAblationTransformer(BaseEstimator, TransformerMixin):
    """Fold-safe cumulative ablations of the champion report's FE blocks."""

    def __init__(
        self,
        *,
        include_a_pair: bool = False,
        log1p_a_pair: bool = False,
        include_s: bool = False,
        include_contrast: bool = False,
        include_exact_top4: bool = False,
        contrast_top_k: int = 5,
        exact_top_k: int = 4,
        exact_min_support: int = 10,
        exact_min_concentration: float = 0.60,
    ) -> None:
        self.include_a_pair = include_a_pair
        self.log1p_a_pair = log1p_a_pair
        self.include_s = include_s
        self.include_contrast = include_contrast
        self.include_exact_top4 = include_exact_top4
        self.contrast_top_k = contrast_top_k
        self.exact_top_k = exact_top_k
        self.exact_min_support = exact_min_support
        self.exact_min_concentration = exact_min_concentration

    def fit(self, X: pd.DataFrame, y: Any = None):
        if self.log1p_a_pair and not self.include_a_pair:
            raise ValueError("log1p_a_pair는 include_a_pair와 함께 사용해야 합니다.")

        self.base_ = CombinedMutationTransformer(
            include_multi_mutated_gene_burden=True,
            include_truncating_gene_flags=True,
            recurrent_missense_min_count=5,
            exclude_hotspot_from_recurrent=True,
        ).fit(X, y)
        train_events = _events(X)
        self.contrast_features_ = self._fit_contrast(train_events, X, y)
        self.exact_events_ = self._fit_exact_events(train_events, X, y)

        extra = self._extra_frame(X)
        self.output_columns_ = extra.columns[
            extra.nunique(dropna=False).gt(1)
        ].tolist()
        self.n_features_in_ = (
            len(self.base_.transform(X).columns) + len(self.output_columns_)
        )
        return self

    def _fit_contrast(
        self,
        events: pd.DataFrame,
        X: pd.DataFrame,
        y: Any,
    ) -> dict[str, tuple[tuple[str, float], ...]]:
        if not self.include_contrast:
            return {}
        if y is None:
            raise ValueError("contrast 피처에는 fold-train labels가 필요합니다.")

        labels = pd.Series(np.asarray(y), index=X.index)
        presence = events[["sample", "gene"]].drop_duplicates()
        selected: dict[str, tuple[tuple[str, float], ...]] = {}
        for left, right in CONTRAST_PAIRS:
            left_index = labels.index[labels.eq(left)]
            right_index = labels.index[labels.eq(right)]
            if len(left_index) == 0 or len(right_index) == 0:
                continue

            left_counts = (
                presence[presence["sample"].isin(left_index)]
                .groupby("gene")["sample"]
                .nunique()
            )
            right_counts = (
                presence[presence["sample"].isin(right_index)]
                .groupby("gene")["sample"]
                .nunique()
            )
            counts = pd.concat(
                [left_counts.rename("left"), right_counts.rename("right")],
                axis=1,
            ).fillna(0)
            counts["support"] = counts["left"] + counts["right"]
            counts["contrast"] = (
                counts["left"] / len(left_index)
                - counts["right"] / len(right_index)
            )
            candidates = counts[counts["support"].ge(10)].copy()
            if candidates.empty:
                continue
            candidates["score"] = candidates["support"] * candidates[
                "contrast"
            ].abs()
            candidates = candidates.sort_values(
                ["score", "support"], ascending=[False, False]
            )
            candidates = candidates.head(self.contrast_top_k)
            selected[f"{left}_vs_{right}"] = tuple(
                (gene, float(np.sign(row["contrast"])))
                for gene, row in candidates.iterrows()
                if row["contrast"] != 0
            )
        return selected

    def _fit_exact_events(
        self,
        events: pd.DataFrame,
        X: pd.DataFrame,
        y: Any,
    ) -> tuple[tuple[str, str], ...]:
        if not self.include_exact_top4:
            return ()
        if y is None:
            raise ValueError("exact top-4 피처에는 fold-train labels가 필요합니다.")
        if events.empty:
            return ()

        labels = pd.Series(np.asarray(y), index=X.index, name="label")
        labeled = events.merge(
            labels, left_on="sample", right_index=True, how="inner"
        )
        labeled = labeled[labeled["event_type"].eq("MISSENSE")]
        if labeled.empty:
            return ()
        grouped = labeled.groupby(["gene", "event", "label"]).size()
        support = grouped.groupby(level=["gene", "event"]).sum()
        concentration = grouped.groupby(level=["gene", "event"]).max() / support
        eligible = support[
            support.ge(self.exact_min_support)
            & concentration.ge(self.exact_min_concentration)
        ]
        if eligible.empty:
            return ()
        ranking = pd.DataFrame(
            {
                "support": eligible,
                "concentration": concentration.loc[eligible.index],
            }
        )
        ranking["score"] = ranking["support"] * ranking["concentration"]
        ranking = ranking.sort_values(
            ["score", "support", "concentration"],
            ascending=[False, False, False],
        )
        return tuple(ranking.head(self.exact_top_k).index.tolist())

    def _extra_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        events = _events(X)
        blocks: list[pd.DataFrame] = []

        if self.include_a_pair:
            pairs = tuple(
                f"{ref}>{alt}"
                for ref in AA
                for alt in AA
                if ref != alt
            )
            substitutions = events.copy()
            parsed = substitutions["event"].str.extract(
                r"^([A-Z])(-?\d+)([A-Z])$"
            )
            substitutions["ref"] = parsed[0]
            substitutions["alt"] = parsed[2]
            substitutions = substitutions.dropna(subset=["ref", "alt"])
            substitutions["pair"] = (
                substitutions["ref"] + ">" + substitutions["alt"]
            )
            blocks.append(
                _count_by(
                    substitutions,
                    X.index,
                    "pair",
                    pairs,
                    "A__pair_",
                    log1p=self.log1p_a_pair,
                )
            )

        if self.include_s:
            if events.empty:
                empty_index = pd.MultiIndex.from_arrays(
                    [[], []], names=["sample", "gene"]
                )
                per_gene = pd.DataFrame(
                    columns=["event_count", "type_count"], index=empty_index
                )
            else:
                per_gene = events.groupby(["sample", "gene"], observed=True).agg(
                    event_count=("event", "size"),
                    type_count=("event_type", "nunique"),
                )
            structure: dict[str, pd.Series] = {}
            for label, mask in (
                ("one_event_gene_count", per_gene.event_count.eq(1)),
                ("two_event_gene_count", per_gene.event_count.eq(2)),
                (
                    "three_or_more_event_gene_count",
                    per_gene.event_count.ge(3),
                ),
                ("multi_notation_type_gene_count", per_gene.type_count.ge(2)),
            ):
                structure[f"S__{label}"] = (
                    mask.groupby(level="sample")
                    .sum()
                    .reindex(X.index, fill_value=0)
                )
            structure["S__max_events_in_one_gene"] = (
                per_gene.event_count.groupby(level="sample")
                .max()
                .reindex(X.index, fill_value=0)
            )
            type_counts = pd.crosstab(events["sample"], events["event_type"]).reindex(
                X.index, fill_value=0
            )
            total = type_counts.sum(axis=1)
            proportions = type_counts.div(total.replace(0, np.nan), axis=0).fillna(0)
            structure["S__notation_type_diversity"] = type_counts.gt(0).sum(axis=1)
            positive = proportions.where(proportions.gt(0), 1.0)
            structure["S__notation_type_entropy"] = -(
                positive * np.log(positive)
            ).sum(axis=1)
            structure["S__dominant_notation_type_share"] = proportions.max(axis=1)
            blocks.append(pd.DataFrame(structure, index=X.index, dtype="float32"))

        if self.include_contrast:
            presence = events[["sample", "gene"]].drop_duplicates()
            for name, selected in self.contrast_features_.items():
                genes = tuple(gene for gene, _ in selected)
                signs = np.asarray([sign for _, sign in selected], dtype="float32")
                if not genes:
                    blocks.append(
                        pd.DataFrame(
                            0.0,
                            index=X.index,
                            columns=[f"C__{name}__count", f"C__{name}__score"],
                            dtype="float32",
                        )
                    )
                    continue
                matrix = pd.crosstab(presence["sample"], presence["gene"]).reindex(
                    index=X.index, columns=genes, fill_value=0
                )
                blocks.append(
                    pd.DataFrame(
                        {
                            f"C__{name}__count": matrix.sum(axis=1),
                            f"C__{name}__score": matrix.to_numpy(dtype="float32")
                            @ signs,
                        },
                        index=X.index,
                        dtype="float32",
                    )
                )

        if self.include_exact_top4:
            observed = set(zip(events["sample"], events["gene"], events["event"]))
            exact = {
                f"D__exact_{gene}_{event}": np.fromiter(
                    ((index, gene, event) in observed for index in X.index),
                    dtype=np.int8,
                    count=len(X),
                ).astype("float32")
                for gene, event in self.exact_events_
            }
            if exact:
                blocks.append(pd.DataFrame(exact, index=X.index))

        return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "base_"):
            raise RuntimeError("transform 전에 fit을 실행해야 합니다.")
        base = self.base_.transform(X)
        extra = self._extra_frame(X).reindex(
            columns=self.output_columns_, fill_value=0
        )
        return pd.concat([base, extra], axis=1)


def make_candidates() -> dict[str, ChampionAblationTransformer]:
    """Return cumulative cases; execute each case in notebook order."""

    return {
        "case_01_exp009_pair_raw": ChampionAblationTransformer(
            include_a_pair=True
        ),
        "case_02_plus_pair_log1p": ChampionAblationTransformer(
            include_a_pair=True,
            log1p_a_pair=True,
        ),
        "case_03_plus_S": ChampionAblationTransformer(
            include_a_pair=True,
            log1p_a_pair=True,
            include_s=True,
        ),
        "case_04_plus_train_contrast": ChampionAblationTransformer(
            include_a_pair=True,
            log1p_a_pair=True,
            include_s=True,
            include_contrast=True,
        ),
        "case_05_plus_train_exact_top4": ChampionAblationTransformer(
            include_a_pair=True,
            log1p_a_pair=True,
            include_s=True,
            include_contrast=True,
            include_exact_top4=True,
        ),
    }
