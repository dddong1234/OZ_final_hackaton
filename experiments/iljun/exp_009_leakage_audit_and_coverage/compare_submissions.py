"""두 제출 CSV가 완전히 동일한지 대조한다.

리팩터링이 예측을 바꾸지 않았음을 증명하는 용도.  ID 순서·행 수·클래스 분포까지
함께 확인하고, 다른 행이 있으면 앞부분을 보여준다.

    .venv/bin/python .../compare_submissions.py A.csv B.csv [--out artifacts/equivalence.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def _record(out: Path | None, payload: dict) -> None:
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out}")


def main(argv: list[str]) -> int:
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if len(argv) != 3:
        print(__doc__)
        return 2
    left_path, right_path = Path(argv[1]), Path(argv[2])
    left, right = pd.read_csv(left_path), pd.read_csv(right_path)

    print(f"A = {left_path}  {left.shape}")
    print(f"B = {right_path}  {right.shape}")

    if list(left.columns) != list(right.columns):
        print(f"✗ 컬럼이 다릅니다: {list(left.columns)} vs {list(right.columns)}")
        return 1
    if len(left) != len(right):
        print(f"✗ 행 수가 다릅니다: {len(left)} vs {len(right)}")
        return 1
    if not left["ID"].equals(right["ID"]):
        print("✗ ID 순서가 다릅니다")
        return 1

    differing = left["SUBCLASS"] != right["SUBCLASS"]
    n_diff = int(differing.sum())
    if n_diff == 0:
        print(f"\n✅ 완전 일치 — {len(left)}행 전부 동일한 예측")
        print(f"   클래스 {left.SUBCLASS.nunique()}종, 최다 {left.SUBCLASS.value_counts().idxmax()}"
              f" ({left.SUBCLASS.value_counts().max()}건)")
        _record(out, {
            "identical": True,
            "rows": len(left),
            "differing_rows": 0,
            "a": str(left_path),
            "b": str(right_path),
            "distinct_classes": int(left.SUBCLASS.nunique()),
        })
        return 0

    print(f"\n✗ {n_diff}행 불일치 ({n_diff / len(left):.3%})")
    sample = pd.DataFrame({
        "ID": left.loc[differing, "ID"],
        "A": left.loc[differing, "SUBCLASS"],
        "B": right.loc[differing, "SUBCLASS"],
    })
    print(sample.head(20).to_string(index=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
