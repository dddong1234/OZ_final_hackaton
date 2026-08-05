"""Train-only raw-versus-canonical mutation-profile audit helpers."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

WT = "WT"
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")


def split_events(value: object) -> tuple[str, ...]:
    """Return deterministic event tokens; missing/Wild Type/blank remain event-free."""
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == WT:
        return ()
    return tuple(
        dict.fromkeys(token.removeprefix("P.") for token in re.sub(r"[;,|]+", " ", text).split() if token)
    )


def event_type(event: str) -> str:
    """Use H0-compatible coarse functional mutation taxonomy."""
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


def _raw_text(value: object) -> str:
    return str(value).strip()


def build_profiles(
    frame: pd.DataFrame, genes: list[str], show_progress: bool = False
) -> tuple[dict[str, list[str]], dict[str, int | bool | dict[str, int]]]:
    """Build raw, canonical-event, and H0-style gene-type row fingerprints."""
    raw_rows = [[] for _ in range(len(frame))]
    canonical_rows = [[] for _ in range(len(frame))]
    type_rows = [[] for _ in range(len(frame))]
    raw_segment_count = 0
    parsed_event_count = 0
    wt_cell_count = 0
    nan_cell_count = 0
    multi_event_cell_count = 0
    type_counter: Counter[str] = Counter()

    iterator = tqdm(genes, desc="canonical profile parsing", unit="gene", disable=not show_progress)
    for gene in iterator:
        values = frame[gene].array
        for row, value in enumerate(values):
            if pd.isna(value):
                nan_cell_count += 1
                continue
            text = _raw_text(value)
            if not text or text.upper() == WT:
                wt_cell_count += 1
                continue
            events = split_events(value)
            raw_segment_count += len(events)
            parsed_event_count += len(events)
            if len(events) > 1:
                multi_event_cell_count += 1
            raw_rows[row].append(f"{gene}={text}")
            for event in events:
                canonical_rows[row].append(f"{gene}={event}")
                kind = event_type(event)
                type_rows[row].append(f"{gene}__{kind}")
                type_counter[kind] += 1

    profiles = {
        "raw": ["|".join(items) for items in raw_rows],
        "canonical_event": ["|".join(items) for items in canonical_rows],
        "gene_type": ["|".join(sorted(set(items))) for items in type_rows],
    }
    audit: dict[str, int | bool | dict[str, int]] = {
        "raw_segment_count": raw_segment_count,
        "parsed_event_count": parsed_event_count,
        "wt_cell_count": wt_cell_count,
        "nan_cell_count": nan_cell_count,
        "multi_event_cell_count": multi_event_cell_count,
        "event_type_counts": dict(sorted(type_counter.items())),
        "segment_conservation": raw_segment_count == parsed_event_count,
    }
    return profiles, audit


def purity_summary(profiles: list[str], labels: np.ndarray, profile_kind: str) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    table = pd.DataFrame({"profile": profiles, "label": labels})
    grouped = table.groupby("profile", sort=False).label.agg(["size", lambda series: series.value_counts().max(), "nunique"])
    grouped.columns = ["support", "majority_count", "label_count"]
    grouped["purity"] = grouped["majority_count"] / grouped["support"]
    detail = grouped.reset_index()
    weighted = float(np.average(detail.purity, weights=detail.support)) if len(detail) else 0.0
    summary: dict[str, float | int | str] = {
        "profile_kind": profile_kind,
        "unique_profiles": int(len(detail)),
        "duplicate_rows": int(detail.loc[detail.support.gt(1), "support"].sum()) if len(detail) else 0,
        "weighted_purity": weighted,
        "conflict_profiles": int(detail.label_count.gt(1).sum()),
    }
    return detail, summary


def canonicalization_disagreement(raw: list[str], canonical: list[str], labels: np.ndarray) -> pd.DataFrame:
    """Summarize whether canonicalization merges distinct raw profiles."""
    table = pd.DataFrame({"raw_profile": raw, "canonical_profile": canonical, "label": labels})
    grouped = table.groupby("canonical_profile", sort=False).agg(
        row_count=("label", "size"),
        raw_profile_count=("raw_profile", "nunique"),
        label_count=("label", "nunique"),
    ).reset_index()
    grouped["raw_merged"] = grouped.raw_profile_count.gt(1)
    return grouped
