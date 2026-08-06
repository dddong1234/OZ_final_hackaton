"""두 제출 CSV의 재현 일치율과 audit JSON 차이를 비교한다.

사용법:
    python compare_submissions.py 내가돌린것.csv 경수님것.csv
audit JSON(<csv>.audit.json)이 옆에 있으면 핵심 필드도 함께 대조한다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

KEYS = [
    "exact_vocabulary_size", "final_feature_count", "structured_feature_count",
    "gene_type_eb_feature_count", "exact_eb_feature_count",
    "specialist_pairs", "selective_non_eb_test_rows",
    "convergence_warning_count", "leakage_check", "nan_as_mutation_count",
]


def load_audit(csv_path: Path) -> dict:
    path = csv_path.with_suffix(".audit.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    per_seed = data.get("per_seed_audits")
    return per_seed[0] if per_seed else data


def main() -> None:
    mine, theirs = Path(sys.argv[1]), Path(sys.argv[2])
    a = pd.read_csv(mine).set_index("ID").SUBCLASS
    b = pd.read_csv(theirs).set_index("ID").SUBCLASS
    if not a.index.equals(b.index):
        common = a.index.intersection(b.index)
        print(f"[warn] ID 집합이 다름: {len(a)} vs {len(b)}, 공통 {len(common)}행만 비교")
        a, b = a.loc[common], b.loc[common]

    same = int((a == b).sum())
    total = len(a)
    print(f"일치율: {same}/{total} = {same / total:.4%}")
    if same < total:
        diff = pd.DataFrame({"mine": a[a != b], "theirs": b[a != b]})
        print("\n불일치 행 (최대 20개):")
        print(diff.head(20).to_string())
        print("\n불일치 클래스 조합 상위:")
        print(diff.groupby(["mine", "theirs"]).size().sort_values(ascending=False).head(10).to_string())

    audit_a, audit_b = load_audit(mine), load_audit(theirs)
    if audit_a and audit_b:
        print("\naudit 대조:")
        for key in KEYS:
            va, vb = audit_a.get(key), audit_b.get(key)
            mark = "OK " if va == vb else "DIFF"
            print(f"  [{mark}] {key}: {va!r} vs {vb!r}")
    else:
        print("\n[info] audit JSON이 한쪽에 없어 상수 대조는 건너뜀")

    env = audit_a.get("environment") or {}
    if env:
        print("\n내 환경:", {k: env[k] for k in ("python", "numpy", "pandas", "scipy", "scikit_learn", "lightgbm") if k in env})


if __name__ == "__main__":
    main()
