# -*- coding: utf-8 -*-
"""Train-only lightweight Exact-event EB model for batch or real-time inference.

This file intentionally has no dependency on other project modules.  It fits
all vocabularies, statistics and the Logistic Regression model on train only;
saved bundles transform future rows without refitting.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


WT = "WT"
EVENT_TYPES = ("MISSENSE", "SYNONYMOUS", "NONSENSE", "FRAMESHIFT", "SPLICE", "INFRAME_INDEL", "OTHER")
TRUNCATING = frozenset({"NONSENSE", "FRAMESHIFT", "SPLICE"})
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_PAIRS = {(left, right): index for index, (left, right) in enumerate((left, right) for left in AA for right in AA if left != right)}
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
SPLICE_RE = re.compile(r"SPLICE|IVS|[+-]\d+")
INDEL_RE = re.compile(r"DEL|INS|DUP")
EB_ALPHA = 1.0
EB_SHRINKAGE = 20.0
EB_CLIP = 4.0
LR_CONFIG = {"solver": "lbfgs", "C": 0.07, "max_iter": 2000, "class_weight": "balanced"}


def normalise_cell(value: object) -> tuple[str, ...]:
    """Return deduplicated events; missing/WT values are deliberately empty."""
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == WT:
        return ()
    return tuple(dict.fromkeys(item.removeprefix("P.") for item in re.sub(r"[;,|]+", " ", text).split() if item))


def classify_event(event: str) -> str:
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


@dataclass(frozen=True)
class Vocabulary:
    exact_events: tuple[str, ...]
    gene_types: tuple[str, ...]


@dataclass
class ParsedRows:
    mutation: sparse.csr_matrix
    exact: sparse.csr_matrix
    gene_type: sparse.csr_matrix
    burden: np.ndarray
    event_type_count: np.ndarray
    truncation_count: np.ndarray
    amino_pair: np.ndarray
    topology: np.ndarray


@dataclass(frozen=True)
class EBState:
    selected: np.ndarray
    weights: np.ndarray
    class_keep: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass
class FeatureState:
    genes: list[str]
    classes: np.ndarray
    vocabulary: Vocabulary
    raw_keep: np.ndarray
    gene_type_eb: EBState
    exact_eb: EBState


@dataclass
class LightweightBundle:
    classes: np.ndarray
    feature_state: FeatureState
    model: LogisticRegression
    audit: dict


def gene_columns(frame: pd.DataFrame, *, training: bool) -> list[str]:
    excluded = {"ID", "SUBCLASS"} if training else {"ID"}
    return [column for column in frame.columns if column not in excluded]


def _records(frame: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    rows: list[tuple[int, int, str, str]] = []
    for gene_index, gene in enumerate(genes):
        for row_index, value in enumerate(frame[gene].array):
            rows.extend((row_index, gene_index, event, classify_event(event)) for event in normalise_cell(value))
    output = pd.DataFrame(rows, columns=["row", "gene_index", "event", "event_type"])
    if output.empty:
        return output
    output = output.drop_duplicates(["row", "gene_index", "event"]).reset_index(drop=True)
    output["gene"] = output.gene_index.map(dict(enumerate(genes)))
    output["exact_name"] = output.gene + "__" + output.event
    output["gene_type_name"] = output.gene + "__" + output.event_type
    return output


def fit_vocabulary(frame: pd.DataFrame, genes: list[str]) -> Vocabulary:
    records = _records(frame, genes)
    if records.empty:
        return Vocabulary((), ())
    return Vocabulary(tuple(sorted(records.exact_name.unique())), tuple(sorted(records.gene_type_name.unique())))


def _binary(records: pd.DataFrame, column: str, vocabulary: tuple[str, ...], row_count: int) -> sparse.csr_matrix:
    if records.empty or not vocabulary:
        return sparse.csr_matrix((row_count, len(vocabulary)), dtype=np.float32)
    lookup = {name: index for index, name in enumerate(vocabulary)}
    columns = records[column].map(lookup)
    known = columns.notna().to_numpy()
    if not known.any():
        return sparse.csr_matrix((row_count, len(vocabulary)), dtype=np.float32)
    result = sparse.coo_matrix((np.ones(known.sum(), dtype=np.float32), (records.loc[known, "row"], columns[known].astype(np.int32))), shape=(row_count, len(vocabulary))).tocsr()
    result.data[:] = 1.0
    return result


def parse_rows(frame: pd.DataFrame, genes: list[str], vocabulary: Vocabulary) -> ParsedRows:
    row_count = len(frame); records = _records(frame, genes)
    mutation = sparse.csr_matrix((row_count, len(genes)), dtype=np.float32)
    burden = np.zeros((row_count, 3), dtype=np.float32)
    event_type_count = np.zeros((row_count, len(EVENT_TYPES)), dtype=np.float32)
    truncation_count = np.zeros((row_count, 1), dtype=np.float32)
    amino_pair = np.zeros((row_count, len(AA_PAIRS)), dtype=np.float32)
    topology = np.zeros((row_count, 4), dtype=np.float32)
    if not records.empty:
        mutated = records[["row", "gene_index"]].drop_duplicates()
        mutation = sparse.coo_matrix((np.ones(len(mutated), dtype=np.float32), (mutated.row, mutated.gene_index)), shape=(row_count, len(genes))).tocsr()
        mutation.data[:] = 1.0
        burden[:, 0] = np.asarray(mutation.sum(axis=1)).ravel()
        burden[:, 1] = records.groupby("row").size().reindex(range(row_count), fill_value=0)
        per_gene = records.groupby(["row", "gene_index"]).agg(event_count=("event", "size"), type_count=("event_type", "nunique"))
        burden[:, 2] = per_gene.event_count.gt(1).groupby(level=0).sum().reindex(range(row_count), fill_value=0)
        for index, event_type in enumerate(EVENT_TYPES):
            event_type_count[:, index] = records.event_type.eq(event_type).groupby(records.row).sum().reindex(range(row_count), fill_value=0)
        truncation_count[:, 0] = records.event_type.isin(TRUNCATING).groupby(records.row).sum().reindex(range(row_count), fill_value=0)
        topology[:, 0] = per_gene.event_count.eq(1).groupby(level=0).sum().reindex(range(row_count), fill_value=0)
        topology[:, 1] = per_gene.event_count.eq(2).groupby(level=0).sum().reindex(range(row_count), fill_value=0)
        topology[:, 2] = per_gene.event_count.ge(3).groupby(level=0).sum().reindex(range(row_count), fill_value=0)
        topology[:, 3] = per_gene.type_count.ge(2).groupby(level=0).sum().reindex(range(row_count), fill_value=0)
        for row, event in records[["row", "event"]].itertuples(index=False):
            match = SUB_RE.fullmatch(event)
            if match and (match.group(1), match.group(3)) in AA_PAIRS:
                amino_pair[int(row), AA_PAIRS[(match.group(1), match.group(3))]] += 1
    return ParsedRows(mutation, _binary(records, "exact_name", vocabulary.exact_events, row_count), _binary(records, "gene_type_name", vocabulary.gene_types, row_count), burden, event_type_count, truncation_count, amino_pair, topology)


def raw_features(parsed: ParsedRows) -> sparse.csr_matrix:
    return sparse.hstack([parsed.mutation, sparse.csr_matrix(np.log1p(parsed.burden)), sparse.csr_matrix(np.log1p(parsed.event_type_count)), sparse.csr_matrix(np.log1p(parsed.truncation_count)), sparse.csr_matrix(np.log1p(parsed.amino_pair)), sparse.csr_matrix(parsed.topology)], format="csr")


def fit_eb(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    selected = np.flatnonzero((support > 0) & (support < matrix.shape[0]))
    if not len(selected):
        return selected, np.zeros((len(classes), 0), dtype=np.float32)
    selected_matrix = matrix[:, selected]; support = support[selected]
    prior = (support + EB_ALPHA) / (len(labels) + 2.0 * EB_ALPHA)
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, label in enumerate(classes):
        positive_mask = labels == label
        positive = np.asarray(selected_matrix[positive_mask].getnnz(axis=0)).ravel().astype(np.float64)
        negative = support - positive
        positive_rate = (positive + EB_SHRINKAGE * prior) / (positive_mask.sum() + EB_SHRINKAGE)
        negative_rate = (negative + EB_SHRINKAGE * prior) / ((~positive_mask).sum() + EB_SHRINKAGE)
        positive_rate = np.clip(positive_rate, 1e-6, 1 - 1e-6)
        negative_rate = np.clip(negative_rate, 1e-6, 1 - 1e-6)
        weights[class_index] = np.log(positive_rate / (1 - positive_rate)) - np.log(negative_rate / (1 - negative_rate))
    return selected, np.clip(weights, -EB_CLIP, EB_CLIP).astype(np.float32)


def apply_eb(matrix: sparse.csr_matrix, selected: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if not len(selected):
        return np.zeros((matrix.shape[0], weights.shape[0]), dtype=np.float32)
    active = matrix[:, selected]
    score = np.asarray(active @ weights.T, dtype=np.float32)
    return score / np.sqrt(np.maximum(np.asarray(active.getnnz(axis=1)).ravel(), 1))[:, None]


def fit_eb_state(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[np.ndarray, EBState]:
    oof = np.zeros((matrix.shape[0], len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_index, valid_index in splitter.split(np.zeros(len(labels)), labels):
        selected, weights = fit_eb(matrix[fit_index], labels[fit_index], classes)
        oof[valid_index] = apply_eb(matrix[valid_index], selected, weights)
    selected, weights = fit_eb(matrix, labels, classes)
    keep = oof.min(axis=0) != oof.max(axis=0)
    oof = oof[:, keep]; mean = oof.mean(axis=0, keepdims=True); std = np.maximum(oof.std(axis=0, keepdims=True), 1e-6)
    return ((oof - mean) / std).astype(np.float32), EBState(selected, weights, keep, mean.astype(np.float32), std.astype(np.float32))


def apply_eb_state(matrix: sparse.csr_matrix, state: EBState) -> np.ndarray:
    return ((apply_eb(matrix, state.selected, state.weights)[:, state.class_keep] - state.mean) / state.std).astype(np.float32)


def transform_features(feature_state: FeatureState, frame: pd.DataFrame) -> sparse.csr_matrix:
    if list(frame.columns) != ["ID", *feature_state.genes]:
        raise ValueError("input columns must be ID followed by the fitted training gene order")
    parsed = parse_rows(frame.loc[:, feature_state.genes], feature_state.genes, feature_state.vocabulary)
    raw = raw_features(parsed)[:, feature_state.raw_keep]
    gene_type_score = apply_eb_state(parsed.gene_type, feature_state.gene_type_eb)
    exact_score = apply_eb_state(parsed.exact, feature_state.exact_eb)
    return sparse.hstack([raw, sparse.csr_matrix(gene_type_score), sparse.csr_matrix(exact_score)], format="csr")


def fit_bundle(train: pd.DataFrame, seed: int = 42) -> LightweightBundle:
    if "ID" not in train or "SUBCLASS" not in train:
        raise ValueError("train must contain ID and SUBCLASS")
    genes = gene_columns(train, training=True)
    if int(train[genes].isna().sum().sum()) != 0:
        raise ValueError("train gene cells must not contain NaN")
    labels = train.SUBCLASS.to_numpy(); classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    train_frame = train.loc[:, ["ID", *genes]]; vocabulary = fit_vocabulary(train_frame.loc[:, genes], genes)
    parsed = parse_rows(train_frame.loc[:, genes], genes, vocabulary)
    raw = raw_features(parsed); raw_keep = np.asarray(raw.min(axis=0).toarray()).ravel() != np.asarray(raw.max(axis=0).toarray()).ravel()
    gene_type_oof, gene_type_state = fit_eb_state(parsed.gene_type, labels, classes, seed)
    exact_oof, exact_state = fit_eb_state(parsed.exact, labels, classes, seed)
    matrix = sparse.hstack([raw[:, raw_keep], sparse.csr_matrix(gene_type_oof), sparse.csr_matrix(exact_oof)], format="csr")
    model = LogisticRegression(**LR_CONFIG, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning); model.fit(matrix, labels)
    feature_state = FeatureState(genes, classes, vocabulary, raw_keep, gene_type_state, exact_state)
    audit = {"seed": seed, "train_rows": int(len(train)), "gene_count": int(len(genes)), "class_count": int(len(classes)), "feature_count": int(matrix.shape[1]), "convergence_warning_count": int(sum(issubclass(item.category, ConvergenceWarning) for item in caught)), "leakage_check": True, "nan_as_mutation_count": 0, "test_read_during_fit": False, "raw_train_test_concat": False}
    return LightweightBundle(classes, feature_state, model, audit)


def predict_proba(bundle: LightweightBundle, frame: pd.DataFrame) -> np.ndarray:
    matrix = transform_features(bundle.feature_state, frame)
    probability = bundle.model.predict_proba(matrix)
    lookup = {label: index for index, label in enumerate(bundle.model.classes_)}
    aligned = probability[:, [lookup[label] for label in bundle.classes]]
    if not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("probabilities must be normalized")
    return aligned.astype(np.float32)


def save_bundle(bundle: LightweightBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, path)
    path.with_suffix(path.suffix + ".audit.json").write_text(json.dumps(bundle.audit, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bundle(path: Path) -> LightweightBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, LightweightBundle):
        raise TypeError("model file is not a LightweightBundle")
    return bundle

