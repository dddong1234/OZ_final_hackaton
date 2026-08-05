"""Train-only parsing, vocabulary, and Empirical-Bayes primitives for H2-S."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from scipy import sparse


EVENT_TYPES = ("MISSENSE", "SYNONYMOUS", "NONSENSE", "FRAMESHIFT", "SPLICE", "INFRAME_INDEL", "OTHER")
TRUNCATING = frozenset({"NONSENSE", "FRAMESHIFT", "SPLICE"})
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_PAIR_INDEX = {(ref, alt): index for index, (ref, alt) in enumerate((ref, alt) for ref in AA for alt in AA if ref != alt)}
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
SPLICE_RE = re.compile(r"SPLICE|IVS|[+-]\d+")
INDEL_RE = re.compile(r"DEL|INS|DUP")


def cell_events(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == "WT":
        return ()
    return tuple(dict.fromkeys(token.removeprefix("P.") for token in re.sub(r"[;,|]+", " ", text).split() if token))


def event_type(event: str) -> str:
    if "FS" in event:
        return "FRAMESHIFT"
    if SPLICE_RE.search(event):
        return "SPLICE"
    if INDEL_RE.search(event):
        return "INFRAME_INDEL"
    if "*" in event or event.endswith("X"):
        return "NONSENSE"
    match = SUB_RE.fullmatch(event)
    if match:
        return "SYNONYMOUS" if match.group(1) == match.group(3) else "MISSENSE"
    return "OTHER"


@dataclass
class ParsedEvents:
    genes: list[str]
    events: pd.DataFrame
    row_count: int
    nan_as_mutation_count: int = 0


def parse_frame(frame: pd.DataFrame, genes: list[str]) -> ParsedEvents:
    records: list[tuple[int, int, str, str]] = []
    for gene_idx, gene in enumerate(genes):
        for row_idx, value in enumerate(frame[gene].array):
            for event in cell_events(value):
                records.append((row_idx, gene_idx, event, event_type(event)))
    events = pd.DataFrame(records, columns=["row", "gene_idx", "event", "event_type"])
    if not events.empty:
        events = events.drop_duplicates(["row", "gene_idx", "event"]).reset_index(drop=True)
        events["gene"] = events.gene_idx.map(dict(enumerate(genes)))
        events["exact_name"] = events.gene + "__" + events.event
        events["gene_type"] = events.gene + "__" + events.event_type
    else:
        events = pd.DataFrame(columns=["row", "gene_idx", "event", "event_type", "gene", "exact_name", "gene_type"])
    return ParsedEvents(genes, events, len(frame), 0)


@dataclass(frozen=True)
class EventState:
    exact_vocabulary: tuple[str, ...]
    gene_type_vocabulary: tuple[str, ...]
    classes: tuple[str, ...]
    eb_weights: np.ndarray
    eb_support: np.ndarray
    class_prior: np.ndarray


@dataclass
class EventMatrices:
    mutation: sparse.csr_matrix
    truncation: sparse.csr_matrix
    exact: sparse.csr_matrix
    gene_type: sparse.csr_matrix
    burden: np.ndarray
    event_count: np.ndarray
    type_counts: np.ndarray
    amino_pair: np.ndarray
    topology: np.ndarray


def _matrix(events: pd.DataFrame, key: str, vocabulary: tuple[str, ...], n_rows: int) -> sparse.csr_matrix:
    if events.empty or not vocabulary:
        return sparse.csr_matrix((n_rows, len(vocabulary)), dtype=np.float32)
    lookup = {name: index for index, name in enumerate(vocabulary)}
    cols = events[key].map(lookup)
    known = cols.notna().to_numpy()
    if not known.any():
        return sparse.csr_matrix((n_rows, len(vocabulary)), dtype=np.float32)
    out = sparse.coo_matrix((np.ones(int(known.sum()), dtype=np.float32), (events.loc[known, "row"], cols[known].astype(np.int32))), shape=(n_rows, len(vocabulary))).tocsr()
    out.data[:] = 1.0
    return out


def fit_event_state(parsed: ParsedEvents, labels: np.ndarray, *, min_support: int = 1, alpha: float = 1.0, shrinkage: float = 20.0) -> EventState:
    classes = tuple(sorted(np.unique(labels).tolist()))
    counts = parsed.events.gene_type.value_counts() if not parsed.events.empty else pd.Series(dtype=np.int64)
    gene_types = tuple(sorted(counts[counts.ge(min_support)].index.tolist()))
    exacts = tuple(sorted(parsed.events.exact_name.unique().tolist())) if not parsed.events.empty else ()
    matrix = _matrix(parsed.events, "gene_type", gene_types, parsed.row_count)
    support = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    priors = np.asarray([(labels == label).mean() for label in classes], dtype=np.float64)
    weights = np.zeros((len(classes), len(gene_types)), dtype=np.float32)
    for ci, label in enumerate(classes):
        mask = labels == label
        positive = np.asarray(matrix[mask].getnnz(axis=0)).ravel().astype(np.float64)
        pos_rate = (positive + alpha * priors[ci]) / (max(mask.sum(), 1) + alpha)
        negative = support - positive
        neg_rate = (negative + alpha * (1.0 - priors[ci])) / (max((~mask).sum(), 1) + alpha)
        log_odds = np.log((pos_rate + 1e-6) / (1 - pos_rate + 1e-6)) - np.log((neg_rate + 1e-6) / (1 - neg_rate + 1e-6))
        weights[ci] = np.clip(log_odds * (support / (support + shrinkage)), -4, 4)
    return EventState(exacts, gene_types, classes, weights, support.astype(np.float32), priors.astype(np.float32))


def transform_event_state(parsed: ParsedEvents, state: EventState) -> EventMatrices:
    n_rows, n_genes = parsed.row_count, len(parsed.genes)
    events = parsed.events
    if events.empty:
        mutation = sparse.csr_matrix((n_rows, n_genes), dtype=np.float32)
        truncation = mutation.copy()
    else:
        mutated = events[["row", "gene_idx"]].drop_duplicates()
        mutation = sparse.coo_matrix((np.ones(len(mutated), dtype=np.float32), (mutated.row, mutated.gene_idx)), shape=(n_rows, n_genes)).tocsr()
        trunc = events.loc[events.event_type.isin(TRUNCATING), ["row", "gene_idx"]].drop_duplicates()
        truncation = sparse.coo_matrix((np.ones(len(trunc), dtype=np.float32), (trunc.row, trunc.gene_idx)), shape=(n_rows, n_genes)).tocsr()
    burden = np.zeros((n_rows, 3), dtype=np.float32)
    burden[:, 0] = np.asarray(mutation.getnnz(axis=1)).ravel()
    event_count = np.bincount(events.row.to_numpy(dtype=np.int32), minlength=n_rows).astype(np.float32).reshape(-1, 1) if not events.empty else np.zeros((n_rows, 1), dtype=np.float32)
    burden[:, 1] = event_count.ravel()
    type_counts = np.zeros((n_rows, len(EVENT_TYPES)), dtype=np.float32)
    amino_pair = np.zeros((n_rows, 380), dtype=np.float32)
    topology = np.zeros((n_rows, 8), dtype=np.float32)
    if not events.empty:
        type_lookup = {name: i for i, name in enumerate(EVENT_TYPES)}
        for row, typ, event in events[["row", "event_type", "event"]].itertuples(index=False):
            type_counts[int(row), type_lookup.get(typ, type_lookup["OTHER"])] += 1
            match = SUB_RE.fullmatch(event)
            if match and match.group(1) in AA and match.group(3) in AA and match.group(1) != match.group(3):
                amino_pair[int(row), AA_PAIR_INDEX[(match.group(1), match.group(3))]] += 1
        by_gene = events.groupby(["row", "gene_idx"]).agg(event_count=("event", "size"), type_count=("event_type", "nunique"))
        burden[:, 2] = by_gene.event_count.gt(1).groupby(level=0).sum().reindex(range(n_rows), fill_value=0).to_numpy()
        masks = (by_gene.event_count.eq(1), by_gene.event_count.eq(2), by_gene.event_count.ge(3), by_gene.type_count.ge(2))
        for col, mask in enumerate(masks):
            topology[:, col] = mask.groupby(level=0).sum().reindex(range(n_rows), fill_value=0).to_numpy()
        topology[:, 4] = by_gene.event_count.groupby(level=0).max().reindex(range(n_rows), fill_value=0).to_numpy()
        proportions = type_counts / np.maximum(type_counts.sum(axis=1, keepdims=True), 1.0)
        topology[:, 5] = (type_counts > 0).sum(axis=1)
        safe = np.where(proportions > 0, proportions, 1.0)
        topology[:, 6] = -(safe * np.log(safe)).sum(axis=1)
        topology[:, 7] = proportions.max(axis=1)
    return EventMatrices(mutation, truncation, _matrix(events, "exact_name", state.exact_vocabulary, n_rows), _matrix(events, "gene_type", state.gene_type_vocabulary, n_rows), burden, event_count, type_counts, np.log1p(amino_pair), topology)


def assert_fold_contract(outer_train: np.ndarray, fit_rows: np.ndarray, outer_valid: np.ndarray) -> dict[str, bool]:
    train_set, fit_set, valid_set = set(outer_train.tolist()), set(fit_rows.tolist()), set(outer_valid.tolist())
    return {"leakage_check": fit_set.issubset(train_set) and not bool(train_set & valid_set) and not bool(fit_set & valid_set), "outer_validation_used_for_fit": bool(fit_set & valid_set)}
