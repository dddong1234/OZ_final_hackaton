"""Train-only raw-versus-normalized mutation profile auditing."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _raw_cell(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _normalized_cell(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == "WT":
        return ()
    tokens = [token.removeprefix("P.") for token in re.sub(r"[;,|]+", " ", text).split() if token]
    return tuple(sorted(set(tokens)))


def raw_profile(frame: pd.DataFrame, genes: list[str]) -> list[str]:
    """Gene-order-preserving raw text fingerprint; no labels or test statistics."""
    output = [[] for _ in range(len(frame))]
    for gene in genes:
        values = frame[gene]
        active = values.notna() & values.astype(str).str.strip().str.upper().ne("WT") & values.astype(str).str.strip().ne("")
        for row, value in values[active].items():
            output[int(row)].append(f"{gene}={_raw_cell(value)}")
    return ["|".join(value) for value in output]


def normalized_profile(frame: pd.DataFrame, genes: list[str]) -> list[str]:
    """Canonical profile after case/prefix/delimiter normalization only."""
    output = [[] for _ in range(len(frame))]
    for gene in genes:
        values = frame[gene]
        active = values.notna() & values.astype(str).str.strip().str.upper().ne("WT") & values.astype(str).str.strip().ne("")
        for row, value in values[active].items():
            tokens = _normalized_cell(value)
            if tokens:
                output[int(row)].append(f"{gene}={' '.join(tokens)}")
    return ["|".join(value) for value in output]


def profile_purity(profile: list[str], labels: np.ndarray) -> pd.DataFrame:
    table = pd.DataFrame({"profile": profile, "label": labels})
    grouped = table.groupby("profile", sort=False).label.agg(["size", lambda values: values.value_counts().max(), "nunique"])
    grouped.columns = ["support", "majority_count", "label_count"]
    grouped["purity"] = grouped.majority_count / grouped.support
    return grouped.reset_index()


def purity_summary(profile: list[str], labels: np.ndarray, profile_kind: str) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    details = profile_purity(profile, labels)
    weighted = float(np.average(details.purity, weights=details.support)) if len(details) else 0.0
    summary = {
        "profile_kind": profile_kind,
        "unique_profiles": int(len(details)),
        "duplicate_rows": int(details.loc[details.support.gt(1), "support"].sum() if len(details) else 0),
        "weighted_purity": weighted,
        "conflict_profiles": int(details.label_count.gt(1).sum()),
    }
    return details, summary
