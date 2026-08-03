"""Train-only mutation grammar recovery; never interprets NaN/WT as events."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


_SPLIT = re.compile(r"\s*(?:;|\||/|,)\s*")
_AA = re.compile(r"^([A-Z*])([0-9]+)([A-Z*])$")
_POS = re.compile(r"(?:[A-Z*])?([0-9]+)(?:[A-Z*])?")


@dataclass(frozen=True)
class CanonicalEvent:
    gene: str
    raw: str
    canonical_type: str
    legacy_type: str
    position: int | None


def _legacy(kind: str) -> str:
    if kind in {"FRAMESHIFT_INS", "FRAMESHIFT_DEL"}:
        return "FRAMESHIFT"
    if kind in {"INFRAME_INS", "INFRAME_DEL", "DUPLICATION", "DELINS_COMPLEX"}:
        return "INFRAME_DEL"
    if kind in {"STOP_LOSS", "START_LOSS"}:
        return "NONSENSE"
    return kind if kind in {"MISSENSE", "SYNONYMOUS", "NONSENSE", "SPLICE"} else "OTHER"


def _classify(raw: str) -> tuple[str, int | None]:
    value = raw.upper().replace(" ", "")
    position = int(_POS.search(value).group(1)) if _POS.search(value) else None
    if re.search(r"SPLICE|IVS|[+-][0-9]+", value): return "SPLICE", position
    if "START" in value and ("LOSS" in value or "LOST" in value): return "START_LOSS", position
    if "STOP" in value and ("LOSS" in value or "LOST" in value): return "STOP_LOSS", position
    if re.search(r"FS|FRAMESHIFT", value): return ("FRAMESHIFT_INS" if "INS" in value else "FRAMESHIFT_DEL"), position
    if "DELINS" in value or "COMPLEX" in value: return "DELINS_COMPLEX", position
    if "DUP" in value: return "DUPLICATION", position
    if "INS" in value: return "INFRAME_INS", position
    if "DEL" in value: return "INFRAME_DEL", position
    match = _AA.match(value)
    if match:
        ref, _, alt = match.groups()
        if alt == "*": return "NONSENSE", position
        if ref == alt: return "SYNONYMOUS", position
        return "MISSENSE", position
    if re.match(r"^[A-Z*]?[0-9]+$", value): return "OTHER_VALID", position
    return "UNKNOWN", position


def parse_cell(gene: str, value: object) -> list[CanonicalEvent]:
    if not isinstance(value, str) or not value.strip() or value.strip().upper() in {"WT", "NAN"}:
        return []
    unique: list[str] = []
    for segment in _SPLIT.split(value.strip()):
        segment = segment.strip()
        if segment and segment.upper() not in {"WT", "NAN"} and segment not in unique:
            unique.append(segment)
    events = []
    for segment in unique:
        kind, position = _classify(segment)
        events.append(CanonicalEvent(gene, segment, kind, _legacy(kind), position))
    return events


def parse_frame(frame: pd.DataFrame) -> list[list[CanonicalEvent]]:
    return [[event for gene, value in zip(frame.columns, row) for event in parse_cell(gene, value)] for row in frame.itertuples(index=False, name=None)]


def event_tokens(events: list[list[CanonicalEvent]], mode: str) -> list[set[str]]:
    if mode not in {"legacy", "canonical"}: raise ValueError("mode must be legacy or canonical")
    return [{f"{event.gene}__{event.legacy_type if mode == 'legacy' else event.canonical_type}" for event in row} for row in events]


def audit_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw_segments: list[str] = []; delimiters = Counter(); parsed: list[CanonicalEvent] = []
    wt_cells = nan_cells = multi_cells = 0
    for row in frame.itertuples(index=False, name=None):
        for gene, value in zip(frame.columns, row):
            if not isinstance(value, str): nan_cells += 1; continue
            if not value.strip() or value.strip().upper() == "WT": wt_cells += 1; continue
            delimiters.update(re.findall(r";|\||/|,", value))
            parts = [part.strip() for part in _SPLIT.split(value) if part.strip() and part.strip().upper() not in {"WT", "NAN"}]
            raw_segments.extend(parts); multi_cells += int(len(parts) > 1); parsed.extend(parse_cell(gene, value))
    type_table = pd.DataFrame(Counter(event.canonical_type for event in parsed).items(), columns=["canonical_type", "count"]).sort_values("count", ascending=False)
    unknown = pd.DataFrame(Counter(event.raw for event in parsed if event.canonical_type == "UNKNOWN").items(), columns=["raw_pattern", "count"]).sort_values("count", ascending=False)
    contract = {"raw_segment_count": len(raw_segments), "parsed_event_count": len(parsed), "unknown_event_count": int((type_table.canonical_type.eq('UNKNOWN') * type_table['count']).sum()) if not type_table.empty else 0, "wt_cell_count": wt_cells, "nan_cell_count": nan_cells, "multi_event_cell_count": multi_cells, "delimiter_counts": dict(delimiters), "segment_conservation": len(raw_segments) == len(parsed)}
    return type_table, unknown, contract
