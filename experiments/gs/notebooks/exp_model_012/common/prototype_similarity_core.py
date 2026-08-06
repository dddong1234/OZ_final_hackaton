"""Fold-train-only mutation profile prototypes for the H0 complement screen."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

EVENT_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
SPLICE_RE = re.compile(r"SPLICE|IVS|[+-]\d+")
INDEL_RE = re.compile(r"DEL|INS|DUP")


@dataclass(frozen=True)
class PrototypeArtifacts:
    vocabulary: dict[str, int]
    idf: np.ndarray
    prototypes: sparse.csr_matrix
    priors: np.ndarray
    classes: np.ndarray


def normalize_cell(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == "WT":
        return ()
    return tuple(dict.fromkeys(token.removeprefix("P.") for token in re.sub(r"[;,|]+", " ", text).split() if token))


def functional_type(event: str) -> str:
    if "FS" in event:
        return "FRAMESHIFT"
    if SPLICE_RE.search(event):
        return "SPLICE"
    if INDEL_RE.search(event):
        return "INFRAME_INDEL"
    if "*" in event or event.endswith("X"):
        return "NONSENSE"
    match = EVENT_RE.fullmatch(event)
    if match:
        return "SYNONYMOUS" if match.group(1) == match.group(3) else "MISSENSE"
    return "OTHER"


def parse_event_tokens(frame: pd.DataFrame, genes: list[str]) -> list[list[str]]:
    rows: list[list[str]] = [[] for _ in range(len(frame))]
    for gene in genes:
        for row_index, value in enumerate(frame[gene].array):
            for event in normalize_cell(value):
                rows[row_index].extend((f"{gene}__{functional_type(event)}", f"{gene}__{event}"))
    return [list(dict.fromkeys(tokens)) for tokens in rows]


def _matrix(token_rows: list[list[str]], vocabulary: dict[str, int]) -> sparse.csr_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row, tokens in enumerate(token_rows):
        for token in tokens:
            column = vocabulary.get(token)
            if column is not None:
                row_indices.append(row)
                column_indices.append(column)
    return sparse.coo_matrix(
        (np.ones(len(row_indices), dtype=np.float32), (row_indices, column_indices)),
        shape=(len(token_rows), len(vocabulary)),
        dtype=np.float32,
    ).tocsr()


def fit_train_only_prototype(
    frame: pd.DataFrame,
    labels: np.ndarray,
    genes: list[str],
    classes: np.ndarray,
) -> PrototypeArtifacts:
    token_rows = parse_event_tokens(frame, genes)
    vocabulary = {token: index for index, token in enumerate(sorted({token for row in token_rows for token in row}))}
    if not vocabulary:
        raise ValueError("fold-train has no event tokens")
    matrix = _matrix(token_rows, vocabulary)
    document_frequency = np.asarray(matrix.getnnz(axis=0)).ravel()
    idf = (np.log((1.0 + len(token_rows)) / (1.0 + document_frequency)) + 1.0).astype(np.float32)
    weighted = matrix.multiply(idf).tocsr()
    weighted = normalize(weighted, norm="l2", axis=1, copy=False)
    centroid_rows = []
    priors = []
    for label in classes:
        member = weighted[np.asarray(labels) == label]
        if member.shape[0]:
            centroid_rows.append(sparse.csr_matrix(member.mean(axis=0)))
            priors.append(member.shape[0] / weighted.shape[0])
        else:
            centroid_rows.append(sparse.csr_matrix((1, weighted.shape[1]), dtype=np.float32))
            priors.append(0.0)
    prototypes = normalize(sparse.vstack(centroid_rows, format="csr"), norm="l2", axis=1, copy=False)
    return PrototypeArtifacts(vocabulary, idf, prototypes, np.asarray(priors, dtype=np.float32), np.asarray(classes, dtype=object))


def predict_prototype(frame: pd.DataFrame, genes: list[str], artifacts: PrototypeArtifacts) -> np.ndarray:
    matrix = _matrix(parse_event_tokens(frame, genes), artifacts.vocabulary)
    weighted = normalize(matrix.multiply(artifacts.idf).tocsr(), norm="l2", axis=1, copy=False)
    similarity = (weighted @ artifacts.prototypes.T).toarray().astype(np.float32, copy=False)
    log_prior = np.log(np.maximum(artifacts.priors, 1e-9))[None, :]
    logits = similarity + log_prior
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits, dtype=np.float32)
    probability /= probability.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    return probability.astype(np.float32)
