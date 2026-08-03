"""Event Ontology 후보 블록들의 train→test 전이율 측정.

    .venv/bin/python experiments/iljun/exp_009_leakage_audit_and_coverage/run_ontology_coverage.py

배경
  임경수님 'Event Ontology 실험 정리'가 세 블록을 제안했다.
    gene x position          (+0.003030, seed42)
    gene x 50aa-bin          (+0.001317, seed42, 미검출)
    gene x functype x bin    (+0.004032, seed42, 최고)

  이 증분은 전부 CV 에서 잰 값이다.  CV 는 train 을 쪼갠 것이라 fold-train 과
  fold-valid 의 어휘가 상당히 겹치지만, LB 는 그렇지 않다 — exp_009 측정에서
  test 변이의 94.5%(출현 기준)가 train 에 아예 없었다.

  따라서 "CV 에서 +0.004" 와 "LB 에서 +0.004" 는 블록마다 다르게 갈린다.
  이 스크립트는 각 블록 정의가 test 를 얼마나 덮는지를 직접 센다.  제출 슬롯을
  쓰기 전에 어느 후보가 살아남을지 미리 가르는 것이 목적이다.

측정 방식
  train 에서 블록의 컬럼 어휘를 만들고, test 이벤트가 그 어휘로 표현되는 비율을
  (1) 출현 횟수 기준, (2) 그 컬럼이 하나라도 켜지는 환자 비율 로 잰다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_submission as rs  # noqa: E402

POSITION_RE = re.compile(r"(\d+)")


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def annotate(events: pd.DataFrame) -> pd.DataFrame:
    """이벤트에 위치(첫 번째 정수)와 50aa 구간을 붙인다."""
    frame = events.copy()
    frame["pos"] = pd.to_numeric(
        frame.event.str.extract(POSITION_RE, expand=False), errors="coerce"
    )
    frame["bin50"] = (frame.pos // 50).astype("Int64")
    return frame


# 블록 정의 — 임경수님 페이지의 세 후보 + 대조군 2개
BLOCKS = {
    "gene (G 블록, 대조군)": lambda f: f.gene,
    "gene x exact event (08 의 R 블록 수준)": lambda f: f.gene + "__" + f.event,
    "gene x position": lambda f: f.gene + "__" + f.pos.astype("string"),
    "gene x 50aa-bin": lambda f: f.gene + "__b" + f.bin50.astype("string"),
    "gene x functype x 50aa-bin (signature)": lambda f: (
        f.gene + "__" + f.event_type + "__b" + f.bin50.astype("string")
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "ontology_coverage.json")
    parser.add_argument("--min-counts", type=int, nargs="+", default=[1, 3, 5, 10])
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    data_dir = args.data_dir or root / "data" / "raw"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    genes = [c for c in train.columns if c not in (rs.CONFIG.id_col, rs.CONFIG.target_col)]

    train_cache = rs.RowCache.build(train[genes], genes, show_progress=False)
    test_cache = rs.RowCache.build(test[genes], genes, show_progress=False,
                                   vocabulary=train_cache.event_names)
    train_events = annotate(train_cache.events)
    test_events = annotate(test_cache.events)
    n_test = len(test)

    # min_count 를 함께 쓴다.  원어휘 전체(min_count=1)로만 재면 오해를 부른다 —
    # 실제 실험은 희소 열을 걸러내므로(예: gene x position 은 214k 어휘에서
    # 2.5k 열만 남았다), 살아남은 열의 커버리지가 진짜 운용값이다.
    report = {}
    for name, key in BLOCKS.items():
        train_keys = key(train_events).dropna()
        test_keys = key(test_events)
        valid = test_keys.notna()
        occurrences = int(valid.sum())
        counts = train_keys.value_counts()

        levels = {}
        for min_count in args.min_counts:
            vocabulary = set(counts[counts >= min_count].index)
            covered = test_keys[valid].isin(vocabulary)
            patients_hit = int(test_events.loc[valid][covered.to_numpy()].row.nunique())
            levels[str(min_count)] = {
                "columns": len(vocabulary),
                "test_occurrences_covered": int(covered.sum()),
                "occurrence_coverage": int(covered.sum()) / occurrences if occurrences else 0.0,
                "test_patients_with_any_hit": patients_hit,
                "patient_hit_rate": patients_hit / n_test,
            }
        report[name] = {"test_occurrences": occurrences, "by_min_count": levels}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"test {n_test:,}명 · train {len(train):,}명\n")
    print(f"{'블록 정의':38s}{'min_cnt':>8s}{'열 수':>10s}{'출현 커버':>10s}{'환자 도달':>10s}")
    print("-" * 78)
    for name, entry in report.items():
        for index, (min_count, level) in enumerate(entry["by_min_count"].items()):
            print(f"{name if index == 0 else '':36s}{min_count:>8s}{level['columns']:>10,}"
                  f"{level['occurrence_coverage']:>10.1%}{level['patient_hit_rate']:>10.1%}")
        print()
    print("-" * 78)
    print("출현 커버 = test 변이 이벤트 중 train 어휘로 표현되는 비율")
    print("환자 도달 = 해당 블록의 열이 하나라도 켜지는 test 환자 비율")
    print("min_cnt  = train 에서 그 횟수 이상 나온 키만 열로 남긴 경우")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
