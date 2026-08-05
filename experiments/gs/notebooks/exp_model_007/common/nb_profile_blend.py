"""Fold-safe gene×event-type profile matrix for Complement NB."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse

WT = "WT"
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")


def _events(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == WT:
        return ()
    return tuple(dict.fromkeys(item.removeprefix("P.") for item in re.sub(r"[;,|]+", " ", text).split() if item))


def _event_type(event: str) -> str:
    if "FS" in event:
        return "FRAMESHIFT"
    if re.search(r"SPLICE|IVS|[+-]\d+", event):
        return "SPLICE"
    if re.search(r"DEL|INS|DUP", event):
        return "INFRAME_INDEL"
    if "*" in event or event.endswith("X"):
        return "NONSENSE"
    match = SUB_RE.fullmatch(event)
    if match:
        return "SYNONYMOUS" if match.group(1) == match.group(3) else "MISSENSE"
    return "OTHER"


def build_gene_type_matrix(frame: pd.DataFrame, genes: list[str], vocabulary: tuple[str, ...] | None) -> tuple[sparse.csr_matrix, tuple[str, ...]]:
    rows: list[int] = []
    names: list[str] = []
    for gene in genes:
        for row, value in enumerate(frame[gene].array):
            events = _events(value)
            names.extend(f"{gene}__{_event_type(event)}" for event in events)
            rows.extend([row] * len(events))
    if vocabulary is None:
        vocabulary = tuple(sorted(set(names)))
    lookup = {name: index for index, name in enumerate(vocabulary)}
    cols = [lookup.get(name, -1) for name in names]
    keep = np.asarray(cols) >= 0
    if not keep.any():
        return sparse.csr_matrix((len(frame), len(vocabulary)), dtype=np.float32), vocabulary
    matrix = sparse.coo_matrix((np.ones(int(keep.sum()), dtype=np.float32), (np.asarray(rows)[keep], np.asarray(cols)[keep])), shape=(len(frame), len(vocabulary))).tocsr()
    matrix.data[:] = 1.0
    return matrix, vocabulary


def fixed_blend(h0_probability: np.ndarray, nb_probability: np.ndarray) -> np.ndarray:
    blended = .75 * np.asarray(h0_probability, dtype=np.float64) + .25 * np.asarray(nb_probability, dtype=np.float64)
    return (blended / blended.sum(axis=1, keepdims=True)).astype(np.float32)
