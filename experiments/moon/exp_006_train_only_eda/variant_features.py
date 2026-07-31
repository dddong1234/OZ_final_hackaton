"""Leak-safe features for the competition's compact protein mutation strings.

This module makes no clinical pathogenicity claim and imports no external gene,
pathway, hotspot, or patient data.  It converts only the supplied per-row
strings into deterministic syntax classes.  Any learned event dictionary is
created by ``fit`` and must therefore be fitted on a CV training split only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


WT = "WT"
MISSING = "MISSING"
EVENT_TYPES = (
    "MISSENSE",
    "SYNONYMOUS",
    "NONSENSE",
    "FRAMESHIFT",
    "SPLICE",
    "INFRAME_INDEL",
    "OTHER",
)
TRUNCATING_TYPES = frozenset({"NONSENSE", "FRAMESHIFT", "SPLICE"})
_SUBSTITUTION = re.compile(r"([A-Z*])(-?\d+)([A-Z*])$")
_SPLICE = re.compile(r"SPLICE|IVS|[+-]\d+")
_INFRAME_INDEL = re.compile(r"DEL|INS|DUP")
_X_STOP = re.compile(r"[A-Z*]-?\d+X$")


def normalize_cell(value: object) -> tuple[str, ...]:
    """Normalize and de-duplicate events from one source cell.

    The source uses whitespace for multiple events.  Accepting common delimiter
    variants makes this row-wise operation robust; it does not learn from data.
    """

    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == WT:
        return ()
    text = re.sub(r"[;,|]+", " ", text)
    events = [item.removeprefix("P.") for item in text.split() if item]
    return tuple(dict.fromkeys(events))


def classify_event(event: str) -> str:
    """Classify notation syntax, not pathogenicity or treatment relevance."""

    event = event.upper().removeprefix("P.")
    if "FS" in event:
        return "FRAMESHIFT"
    if _SPLICE.search(event):
        return "SPLICE"
    if _INFRAME_INDEL.search(event):
        return "INFRAME_INDEL"
    if "*" in event or _X_STOP.fullmatch(event):
        return "NONSENSE"
    match = _SUBSTITUTION.fullmatch(event)
    if match:
        return "SYNONYMOUS" if match.group(1) == match.group(3) else "MISSENSE"
    return "OTHER"


def normalise_values(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = dataframe.reindex(columns=columns).astype("string")
    values = values.apply(lambda series: series.str.strip().str.upper())
    return values.replace(r"^$", pd.NA, regex=True).fillna(MISSING)


def event_table(values: pd.DataFrame) -> pd.DataFrame:
    """Return one unique sample-gene-event row for non-WT, non-missing cells."""

    eligible = values.ne(WT) & values.ne(MISSING)
    row_pos, col_pos = np.nonzero(eligible.to_numpy(dtype=bool))
    if not len(row_pos):
        return pd.DataFrame(columns=["sample", "gene", "event", "event_type"])
    raw = values.to_numpy(dtype=object)
    long = pd.DataFrame(
        {
            "sample": values.index.take(row_pos),
            "gene": values.columns.take(col_pos),
            "raw": raw[row_pos, col_pos],
        }
    )
    long["event"] = long["raw"].map(normalize_cell)
    events = long.explode("event").drop(columns="raw").dropna(subset="event")
    events = events.drop_duplicates(subset=["sample", "gene", "event"])
    events["event_type"] = events["event"].map(classify_event)
    return events.reset_index(drop=True)


def _presence(
    events: pd.DataFrame,
    sample_index: pd.Index,
    keys: Iterable[str],
    key_column: str,
    prefix: str,
    allowed_types: frozenset[str] | None = None,
) -> pd.DataFrame:
    selected = events if allowed_types is None else events[events.event_type.isin(allowed_types)]
    keys = list(keys)
    if not keys:
        return pd.DataFrame(index=sample_index)
    if selected.empty:
        matrix = pd.DataFrame(0, index=sample_index, columns=keys, dtype="int8")
    else:
        matrix = pd.crosstab(selected["sample"], selected[key_column]).reindex(
            index=sample_index, columns=keys, fill_value=0
        )
        matrix = matrix.gt(0).astype("int8")
    matrix.columns = [f"{prefix}{key}" for key in matrix.columns]
    return matrix


def summary_features(
    values: pd.DataFrame,
    events: pd.DataFrame,
    recurrent_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Create row-internal burden and functional-class summaries."""

    index = values.index
    result = pd.DataFrame(index=index)
    result["summary__mutated_gene_count"] = (
        values.ne(WT) & values.ne(MISSING)
    ).sum(axis=1).astype("int16")
    result["summary__event_count"] = events.groupby("sample").size().reindex(index, fill_value=0).astype("int16")
    multi = events.groupby(["sample", "gene"]).size().gt(1)
    result["summary__multi_event_gene_count"] = multi.groupby("sample").sum().reindex(index, fill_value=0).astype("int16")
    for event_type in EVENT_TYPES:
        counts = events.loc[events.event_type.eq(event_type)].groupby("sample").size()
        result[f"summary__{event_type.lower()}_event_count"] = counts.reindex(index, fill_value=0).astype("int16")
    truncating = events.loc[events.event_type.isin(TRUNCATING_TYPES), ["sample", "gene"]].drop_duplicates()
    result["summary__truncating_gene_count"] = truncating.groupby("sample").size().reindex(index, fill_value=0).astype("int16")
    if recurrent_pairs:
        pair_index = pd.MultiIndex.from_tuples(recurrent_pairs, names=["gene", "event"])
        recurrent = pd.MultiIndex.from_frame(events[["gene", "event"]]).isin(pair_index)
        result["summary__recurrent_missense_event_count"] = events.loc[recurrent].groupby("sample").size().reindex(index, fill_value=0).astype("int16")
    else:
        result["summary__recurrent_missense_event_count"] = 0
    return result


class WTBinaryTransformer(BaseEstimator, TransformerMixin):
    """The common WT/non-WT reference representation, fit on schema only."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "WTBinaryTransformer":
        self.feature_columns_ = list(X.columns)
        self.output_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        values = normalise_values(X, self.feature_columns_)
        binary = ((values.ne(WT)) & (values.ne(MISSING))).astype("int8")
        return sparse.csr_matrix(binary.to_numpy(dtype=np.int8))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.output_columns_, dtype=object)


class VariantFeatureTransformer(BaseEstimator, TransformerMixin):
    """Fold-fitted functional and recurrent-event feature transformer.

    ``feature_set='gene_burden'`` adds only gene presence plus sample-level
    mutation counts.  ``functional_recurrent`` additionally adds gene-level
    truncating flags and exact recurrent missense events.  The latter dictionary
    is built only from the training data supplied to ``fit``.
    """

    def __init__(self, feature_set: str = "functional_recurrent", recurrent_min_count: int = 5):
        self.feature_set = feature_set
        self.recurrent_min_count = recurrent_min_count

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "VariantFeatureTransformer":
        if self.feature_set not in {"gene_burden", "functional_recurrent"}:
            raise ValueError("feature_set must be 'gene_burden' or 'functional_recurrent'.")
        if self.recurrent_min_count < 2:
            raise ValueError("recurrent_min_count must be at least 2.")
        self.feature_columns_ = list(X.columns)
        values = normalise_values(X, self.feature_columns_)
        events = event_table(values)
        self.active_genes_ = values.columns[(values.ne(WT) & values.ne(MISSING)).any(axis=0)].tolist()
        self.truncating_genes_ = sorted(events.loc[events.event_type.isin(TRUNCATING_TYPES), "gene"].unique())
        missense = events.loc[events.event_type.eq("MISSENSE"), ["gene", "event"]]
        counts = missense.value_counts().rename("count").reset_index()
        selected = counts.loc[counts["count"].ge(self.recurrent_min_count), ["gene", "event"]]
        self.recurrent_pairs_ = list(selected.itertuples(index=False, name=None))
        return self

    def _raw_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = normalise_values(X, self.feature_columns_)
        events = event_table(values)
        gene_presence = ((values[self.active_genes_].ne(WT)) & (values[self.active_genes_].ne(MISSING))).astype("int8")
        gene_presence.columns = [f"mutated__{gene}" for gene in gene_presence.columns]
        summary = summary_features(values, events, self.recurrent_pairs_)
        # Missingness is audited in EDA but deliberately excluded from model input:
        # it reflects data capture, not a mutation-derived biological feature.
        result = pd.concat([gene_presence, summary], axis=1)
        if self.feature_set == "functional_recurrent":
            truncating = _presence(events, values.index, self.truncating_genes_, "gene", "truncating__", TRUNCATING_TYPES)
            if self.recurrent_pairs_:
                pair_lookup = pd.DataFrame(self.recurrent_pairs_, columns=["gene", "event"])
                recurring = events.merge(pair_lookup, on=["gene", "event"], how="inner")
                recurring["pair"] = recurring["gene"] + "__" + recurring["event"]
                pair_keys = [f"{gene}__{event}" for gene, event in self.recurrent_pairs_]
                recurrent = _presence(recurring, values.index, pair_keys, "pair", "recurrent_missense__")
            else:
                recurrent = pd.DataFrame(index=values.index)
            result = pd.concat([result, truncating, recurrent], axis=1)
        # Constant features arise when a syntax type is absent in a fold.  Drop
        # them after fitting the fold, so they cannot alter the validation schema.
        if hasattr(self, "output_columns_"):
            return result.reindex(columns=self.output_columns_, fill_value=0).astype("int16")
        return result.astype("int16")

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        raw = self._raw_transform(X).reindex(columns=self.output_columns_, fill_value=0)
        return sparse.csr_matrix(raw.to_numpy(dtype=np.int16))

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params: object) -> sparse.csr_matrix:
        self.fit(X, y)
        raw = self._raw_transform(X)
        self.output_columns_ = raw.columns[raw.nunique(dropna=False).gt(1)].tolist()
        return sparse.csr_matrix(raw.reindex(columns=self.output_columns_).to_numpy(dtype=np.int16))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(getattr(self, "output_columns_", []), dtype=object)
