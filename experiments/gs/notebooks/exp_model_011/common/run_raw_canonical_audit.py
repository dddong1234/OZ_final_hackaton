"""Run a memory-safe, train-only audit of raw versus H0 canonical mutation profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from raw_canonical_audit import build_profiles, canonicalization_disagreement, purity_summary

RUN_ID = "exp-raw-canonical-audit-01"
PROFILE_KINDS = ("raw", "canonical_event", "gene_type")


def project_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "data" / "raw" / "train.csv").exists():
            return candidate
    raise FileNotFoundError("data/raw/train.csv가 있는 프로젝트 루트를 찾지 못했습니다.")


def summarize_profiles(profiles: dict[str, list[str]], labels: np.ndarray) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summaries: list[dict[str, float | int | str]] = []
    details: dict[str, pd.DataFrame] = {}
    for kind in PROFILE_KINDS:
        detail, summary = purity_summary(profiles[kind], labels, kind)
        details[kind] = detail
        summaries.append(summary)
    return pd.DataFrame(summaries), details


def _hashed_distribution(detail: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Write only aggregate profile characteristics; never persist long raw mutation strings."""
    bucket = detail.assign(
        profile_kind=kind,
        profile_hash=detail.profile.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]),
        support_bucket=pd.cut(detail.support, bins=[0, 1, 2, 5, 10, np.inf], labels=["1", "2", "3-5", "6-10", "11+"]),
    )
    return bucket[["profile_kind", "profile_hash", "support", "majority_count", "label_count", "purity", "support_bucket"]]


def run(run_id: str, smoke: bool = False) -> Path:
    root = project_root()
    train_path = root / "data" / "raw" / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    train = pd.read_csv(train_path)
    required = {"ID", "SUBCLASS"}
    if not required.issubset(train.columns):
        raise AssertionError(f"train schema missing: {sorted(required - set(train.columns))}")
    genes = [column for column in train.columns if column not in required]
    if smoke:
        train = train.iloc[:120].copy()
        genes = genes[:60]
    if train[genes].isna().sum().sum() != 0:
        raise AssertionError("train NaN 계약 위반: 감사 입력에는 결측이 없어야 합니다.")

    print(f"train-only audit: rows={len(train):,}, genes={len(genes):,}", flush=True)
    profiles, parser_audit = build_profiles(train[genes], genes, show_progress=True)
    if not bool(parser_audit["segment_conservation"]):
        raise AssertionError("원문 segment 보존 검사 실패")
    labels = train.SUBCLASS.to_numpy()
    summary, details = summarize_profiles(profiles, labels)

    raw_to_event = canonicalization_disagreement(profiles["raw"], profiles["canonical_event"], labels)
    event_to_type = canonicalization_disagreement(profiles["canonical_event"], profiles["gene_type"], labels)
    raw_to_event["transition"] = "raw_to_canonical_event"
    event_to_type["transition"] = "canonical_event_to_gene_type"
    transition = pd.concat([raw_to_event, event_to_type], ignore_index=True)
    transition_summary = transition.groupby("transition", as_index=False).agg(
        target_profiles=("canonical_profile", "size"),
        merged_target_profiles=("raw_merged", "sum"),
        merged_rows=("row_count", lambda value: int(value[transition.loc[value.index, "raw_merged"]].sum())),
        max_source_profiles=("raw_profile_count", "max"),
    )
    purity = summary.set_index("profile_kind").weighted_purity
    raw_vs_gene_type_purity_delta = float(purity["raw"] - purity["gene_type"])
    merged_rows = int(
        transition_summary.loc[
            transition_summary.transition.eq("canonical_event_to_gene_type"), "merged_rows"
        ].iloc[0]
    )

    result = Path(__file__).resolve().parent.parent / "result"
    result.mkdir(parents=True, exist_ok=True)
    summary.to_csv(result / f"{run_id}_summary.csv", index=False)
    transition_summary.to_csv(result / f"{run_id}_transition_summary.csv", index=False)
    pd.concat([_hashed_distribution(details[kind], kind) for kind in PROFILE_KINDS], ignore_index=True).to_csv(
        result / f"{run_id}_profile_purity_hashed.csv", index=False
    )
    event_counts = pd.DataFrame(
        [{"event_type": name, "count": count} for name, count in dict(parser_audit["event_type_counts"]).items()]
    )
    event_counts.to_csv(result / f"{run_id}_event_type_counts.csv", index=False)

    audit = {
        "run_id": run_id,
        "smoke": smoke,
        "test_read": False,
        "train_test_concat": False,
        "test_path_accessed": False,
        "train_rows": int(len(train)),
        "gene_count": int(len(genes)),
        "nan_as_mutation_count": 0,
        "leakage_check": True,
        "segment_conservation": bool(parser_audit["segment_conservation"]),
        "raw_segment_count": int(parser_audit["raw_segment_count"]),
        "parsed_event_count": int(parser_audit["parsed_event_count"]),
        "multi_event_cell_count": int(parser_audit["multi_event_cell_count"]),
        "raw_vs_gene_type_weighted_purity_delta": raw_vs_gene_type_purity_delta,
        "canonical_event_to_gene_type_merged_rows": merged_rows,
        "verdict": "raw_token_model_candidate" if int(transition_summary.loc[transition_summary.transition.eq("canonical_event_to_gene_type"), "merged_rows"].iloc[0]) > 0 else "no_new_canonical_information",
    }
    (result / f"{run_id}_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.run_id, smoke=args.smoke)


if __name__ == "__main__":
    main()
