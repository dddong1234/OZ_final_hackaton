"""train 어휘가 test 변이를 얼마나 덮는가 — 표현 수준별 전이율 측정.

    .venv/bin/python experiments/iljun/exp_009_leakage_audit_and_coverage/run_vocab_coverage.py

배경
  exp-gs-002-08(팀 1위 라인)을 train/test 분리 파싱으로 리팩터하는 과정에서,
  test 이벤트의 대부분이 train 어휘에 없다는 것이 드러났다.  이 스크립트는 그
  관찰을 표현 수준별로 정량화한다.

무엇을 재는가
  같은 test 데이터를 세 가지 수준으로 표현했을 때, 각각 train 에서 학습한
  어휘/컬럼으로 얼마나 표현되는지:

    1. 정확 변이 (gene + event)   — 예: TP53__R273H
    2. recurrent missense (>=5)   — 모델이 실제로 쓰는 R 블록 230 열
    3. 유전자 단위 (G 블록)        — 예: TP53 에 변이가 있는가

왜 중요한가
  CV 는 train 내부라 어휘가 겹치지만 LB 는 처음 보는 변이가 대부분이다.
  표현 수준별 전이율은 팀이 관측해 온 CV<->LB 간격(0.08~0.14)과 블록별
  패스스루 차이가 어디서 오는지를 직접 설명한다.

출력
  artifacts/coverage.json — 아래 수치 전부
  stdout                  — 사람이 읽는 요약표
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_submission as rs  # noqa: E402  (경로 주입 후 import)


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts" / "coverage.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    data_dir = args.data_dir or root / "data" / "raw"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    genes = [c for c in train.columns if c not in (rs.CONFIG.id_col, rs.CONFIG.target_col)]

    train_cache = rs.RowCache.build(train[genes], genes, show_progress=False)
    test_cache = rs.RowCache.build(test[genes], genes, show_progress=False,
                                   vocabulary=train_cache.event_names)

    train_vocab = set(train_cache.event_names)
    test_events = test_cache.events
    test_vocab = set(test_events.pair.unique())

    # 1. 정확 변이 — 종류 기준과 출현 기준을 모두 본다.
    #    (희귀 싱글턴 때문인지, 정말 안 겹치는지를 가르기 위해)
    occ_total = len(test_events)
    occ_known = int(test_events.pair.isin(train_vocab).sum())

    # 2. 모델이 실제 쓰는 R 블록: train 에서 5회 이상 관측된 missense
    recurrent_mask = (
        np.asarray(train_cache.event_matrix.getnnz(axis=0)).ravel() >= rs.CONFIG.recurrent_min_count
    ) & train_cache.event_is_missense
    recurrent_names = {train_cache.event_names[i] for i in np.flatnonzero(recurrent_mask)}
    occ_recurrent = int(test_events.pair.isin(recurrent_names).sum())
    rows_hit = int(test_events[test_events.pair.isin(recurrent_names)].row.nunique())

    # 3. 유전자 단위 — G 블록이 쓰는 해상도
    train_genes = set(train_cache.events.gene.unique())
    test_genes = set(test_events.gene.unique())

    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "exact_event": {
            "train_vocabulary": len(train_vocab),
            "test_distinct": len(test_vocab),
            "test_distinct_covered": len(test_vocab & train_vocab),
            "test_distinct_coverage": len(test_vocab & train_vocab) / len(test_vocab),
            "test_occurrences": occ_total,
            "test_occurrences_covered": occ_known,
            "test_occurrence_coverage": occ_known / occ_total,
            "test_only_distinct": len(test_vocab - train_vocab),
        },
        "recurrent_missense_block": {
            "min_count": rs.CONFIG.recurrent_min_count,
            "columns": int(recurrent_mask.sum()),
            "test_occurrences_captured": occ_recurrent,
            "test_occurrence_coverage": occ_recurrent / occ_total,
            "test_rows_with_any_hit": rows_hit,
            "test_row_hit_rate": rows_hit / len(test),
        },
        "gene_block": {
            "train_genes_with_variant": len(train_genes),
            "test_genes_with_variant": len(test_genes),
            "test_genes_covered": len(test_genes & train_genes),
            "test_gene_coverage": len(test_genes & train_genes) / len(test_genes),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    exact, rec, gene = report["exact_event"], report["recurrent_missense_block"], report["gene_block"]
    print("표현 수준별 test 커버리지 (train 에서 학습한 어휘 기준)")
    print(f"{'수준':28s}{'커버리지':>12s}   비고")
    print("-" * 78)
    print(f"{'정확 변이 (종류 기준)':26s}{exact['test_distinct_coverage']:>11.1%}   "
          f"{exact['test_distinct_covered']:,} / {exact['test_distinct']:,}")
    print(f"{'정확 변이 (출현 기준)':26s}{exact['test_occurrence_coverage']:>11.1%}   "
          f"{exact['test_occurrences_covered']:,} / {exact['test_occurrences']:,}")
    print(f"{'recurrent missense 블록':26s}{rec['test_occurrence_coverage']:>11.2%}   "
          f"{rec['columns']} 열이 test 환자 {rec['test_row_hit_rate']:.1%} 를 건드림")
    print(f"{'유전자 단위 (G 블록)':26s}{gene['test_gene_coverage']:>11.1%}   "
          f"{gene['test_genes_covered']:,} / {gene['test_genes_with_variant']:,}")
    print("-" * 78)
    print(f"→ 정확 변이는 출현 기준으로도 {1 - exact['test_occurrence_coverage']:.1%} 가 미지의 변이다.")
    print(f"  희귀 싱글턴 문제가 아니라 표현 수준의 문제이며, 유전자 단위로 내려가면 "
          f"{gene['test_gene_coverage']:.1%} 가 전이된다.")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
