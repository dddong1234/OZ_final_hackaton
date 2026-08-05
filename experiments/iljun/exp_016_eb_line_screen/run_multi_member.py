"""EB 라인끼리 서로 보완되나 — 다중 결합 테스트.

    .venv/bin/python experiments/iljun/exp_016_eb_line_screen/run_multi_member.py

왜 하는가
  지금까지 전부 `base + 파트너 1개`만 쟀다.  p1_eb(+0.0120)와 point_process_eb(+0.0175)가
  각각 base 를 개선하는데, **둘이 서로를 보완하는지 겹치는지는 모른다.**

  클래스 분해가 서로 다른 그림을 보여준다 — 겹치는 부분(DLBC·KIRC)과 다른 부분
  (TGCT vs LAML)이 같이 있다.  겹침이 크면 하나만 쓰면 되고, 다르면 합쳐야 한다.
  이 답이 경수님께 무엇을 요청할지도 정한다.

무엇을 하나
  · 후보 간 확률 상관 행렬 — 중복도를 직접 본다
  · 2단계 fold-local 블렌드: (base + A) 를 새 base 로 두고 B 를 계약 grid 로 얹는다
  · **한계 기여** = gain(base+A+B) − gain(base+A).  이 값이 B 가 A 위에 보태는 몫이다
  · 순서 민감도를 보려고 양방향(A→B, B→A)을 모두 낸다
  · 최고 쌍 위에 세 번째를 얹어 삼중 결합까지 확인한다

  모델을 학습하지 않는다. seed 42, 계약 5점 grid.
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CONTRACT_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)
THREE_WAY = {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15}

# 모델 계열마다 하나씩 — 같은 파일 안의 변형도 접근이 다르면 따로 넣는다.
LINES = {
    "point_process_eb": ("exp-point-process-eb-01_seed42_oof_probabilities.csv",
                         "point_process_eb_"),
    "eb_enrich02": ("exp-empirical-bayes-enrichment-02_seed42_oof_probabilities.csv", "eb_"),
    "p1_eb": ("exp-all-class-evidence-ranker-01_seed42_oof_probabilities.csv", "p1_eb__"),
    "ranker": ("exp-all-class-evidence-ranker-01_seed42_oof_probabilities.csv", "ranker__"),
    "gate": ("exp-all-class-evidence-ranker-01_seed42_oof_probabilities.csv", "gate__"),
    "G0_P1_EB": ("exp-parser-recovery-g1-01_seed42_oof_probabilities.csv", "G0_P1_EB_"),
    "h2_s": ("exp-h2-evidence-shape-01_seed42_oof_probabilities.csv", "h2_s__"),
    "architecture": ("exp-intragenic-architecture-eb-01_oof_probabilities.csv", "architecture__"),
    "specialist": ("exp-auto-validated-pair-specialist-01_seed42_oof_probabilities.csv",
                   "specialist__"),
    "H0_safe3way": ("exp-safe-3way-ensemble-01_seed42_oof_probabilities.csv", "H0__"),
}


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def macro_f1(proba, classes, truth) -> float:
    return float(f1_score(truth, classes[proba.argmax(axis=1)],
                          average="macro", zero_division=0))


def pick_weight(base, extra, classes, truth, rows) -> float:
    best_weight, best = 0.0, macro_f1(base[rows], classes, truth[rows])
    for weight in CONTRACT_GRID:
        score = macro_f1((1 - weight) * base[rows] + weight * extra[rows],
                         classes, truth[rows])
        if score > best:
            best_weight, best = weight, score
    return best_weight


def foldlocal_blend(base, extra, classes, truth, folds):
    blended = np.zeros_like(base)
    weights = {}
    for fold in np.unique(folds):
        valid = folds == fold
        weight = pick_weight(base, extra, classes, truth, ~valid)
        weights[int(fold)] = weight
        blended[valid] = (1 - weight) * base[valid] + weight * extra[valid]
    return blended, weights


def load_line(root: Path, filename: str, prefix: str, classes, truth, seed: int):
    matches = list(root.glob(f"experiments/gs/**/result/{filename}"))
    if not matches:
        return None
    frame = pd.read_csv(matches[0])
    if "seed" in frame.columns:
        frame = frame[frame["seed"] == seed].reset_index(drop=True)
    if len(frame) != len(truth) or "true_class" not in frame.columns:
        return None
    if not (frame["true_class"].to_numpy() == truth).all():
        return None
    wanted = [f"{prefix}{name}" for name in classes]
    if any(column not in frame.columns for column in wanted):
        return None
    matrix = frame[wanted].to_numpy(dtype=np.float64)
    row_sum = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(row_sum > 0, row_sum, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=5, help="쌍 조합에 쓸 상위 라인 수")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    out = args.out or (Path(__file__).parent / "artifacts" / "multi_member.json")
    oof_dir = root / "experiments" / "iljun" / "results" / "contract_probabilities"

    anchor = pd.read_csv(oof_dir / f"oof_champion_auto_noexact_seed{args.seed}.csv")
    classes = np.array([c[len("prob_"):] for c in anchor.columns if c.startswith("prob_")])
    columns = [f"prob_{name}" for name in classes]
    truth = anchor["SUBCLASS"].to_numpy()
    folds = anchor["fold"].to_numpy()

    base = anchor[columns].to_numpy(dtype=np.float64) * THREE_WAY["champion"]
    for name in ("ovr", "lgbm"):
        block = pd.read_csv(oof_dir / f"oof_{name}_auto_noexact_seed{args.seed}.csv")[columns]
        base = base + block.to_numpy(dtype=np.float64) * THREE_WAY[name]
    base_score = macro_f1(base, classes, truth)

    lines, singles = {}, {}
    for label, (filename, prefix) in LINES.items():
        matrix = load_line(root, filename, prefix, classes, truth, args.seed)
        if matrix is None:
            print(f"  건너뜀 {label}", flush=True)
            continue
        lines[label] = matrix
        blended, _ = foldlocal_blend(base, matrix, classes, truth, folds)
        singles[label] = macro_f1(blended, classes, truth) - base_score

    order = sorted(singles, key=lambda label: -singles[label])
    print(f"base(three_way) {base_score:.6f} · 라인 {len(lines)}개\n")
    print("단독 추가 이득")
    for label in order:
        print(f"  {label:20s} {singles[label]:+.6f}")

    # ── 후보 간 상관 (중복도)
    print("\n후보 간 확률 상관 (상위 5개)")
    head = order[:5]
    print(f"{'':20s}" + "".join(f"{label[:11]:>12s}" for label in head))
    correlation = {}
    for left in head:
        cells = []
        for right in head:
            value = float(np.corrcoef(lines[left].ravel(), lines[right].ravel())[0, 1])
            correlation[f"{left}|{right}"] = value
            cells.append("  —" if left == right else f"{value:.3f}")
        print(f"{left[:19]:20s}" + "".join(f"{cell:>12s}" for cell in cells))

    # ── 2단계 결합: (base + A) 위에 B
    print("\n2단계 결합 — B 의 한계 기여 = gain(base+A+B) − gain(base+A)")
    pairs = []
    for first, second in permutations(order[:args.top], 2):
        stage1, _ = foldlocal_blend(base, lines[first], classes, truth, folds)
        stage2, weights = foldlocal_blend(stage1, lines[second], classes, truth, folds)
        total = macro_f1(stage2, classes, truth) - base_score
        pairs.append({
            "first": first, "second": second,
            "gain_first_only": singles[first],
            "gain_total": total,
            "marginal_of_second": total - singles[first],
            "stage2_weights": weights,
        })
    pairs.sort(key=lambda row: -row["gain_total"])
    print(f"{'A → B':44s}{'A 단독':>11s}{'A+B':>11s}{'B 한계기여':>12s}")
    for row in pairs[:10]:
        print(f"{row['first'][:20] + ' → ' + row['second'][:18]:44s}"
              f"{row['gain_first_only']:>+11.6f}{row['gain_total']:>+11.6f}"
              f"{row['marginal_of_second']:>+12.6f}")

    # ── 삼중 결합: 최고 쌍 위에 세 번째
    best = pairs[0]
    triples = []
    stage1, _ = foldlocal_blend(base, lines[best["first"]], classes, truth, folds)
    stage2, _ = foldlocal_blend(stage1, lines[best["second"]], classes, truth, folds)
    for third in order:
        if third in (best["first"], best["second"]):
            continue
        stage3, _ = foldlocal_blend(stage2, lines[third], classes, truth, folds)
        total = macro_f1(stage3, classes, truth) - base_score
        triples.append({"third": third, "gain_total": total,
                        "marginal": total - best["gain_total"]})
    triples.sort(key=lambda row: -row["gain_total"])
    print(f"\n삼중 결합 — {best['first']} → {best['second']} ({best['gain_total']:+.6f}) 위에")
    for row in triples[:5]:
        print(f"  + {row['third']:20s} 합계 {row['gain_total']:+.6f} "
              f"· 한계 {row['marginal']:+.6f}")

    report = {"seed": args.seed, "base_macro_f1": base_score, "singles": singles,
              "correlation": correlation, "pairs": pairs, "triples": triples,
              "best_pair": best,
              "note": ("seed 42 단일. 파트너 OOF 는 우리 fold 에 중첩되어 있지 않다. "
                       "2단계 모두 계약 5점 grid 를 fold-local 로 적용했다.")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
