"""CLI: audit raw and normalized profiles from train.csv only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from profile_audit import normalized_profile, purity_summary, raw_profile


def project_root() -> Path:
    for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv를 가진 프로젝트 루트를 찾지 못했습니다.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-raw-profile-purity-audit-01")
    args = parser.parse_args()
    root = project_root()
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN 계약 위반")
    outputs = []
    result = Path(__file__).parent.parent / "result"
    result.mkdir(exist_ok=True)
    for kind, profiles in (("raw", raw_profile(train[genes], genes)), ("normalized", normalized_profile(train[genes], genes))):
        detail, summary = purity_summary(profiles, train.SUBCLASS.to_numpy(), kind)
        detail.to_csv(result / f"{args.run_id}_{kind}_profile_purity.csv", index=False)
        outputs.append(summary)
    pd.DataFrame(outputs).to_csv(result / f"{args.run_id}_summary.csv", index=False)
    (result / f"{args.run_id}_audit.json").write_text(json.dumps({"test_read": False, "train_rows": len(train), "gene_count": len(genes)}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
