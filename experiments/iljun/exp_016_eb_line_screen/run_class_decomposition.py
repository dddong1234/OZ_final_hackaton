"""상위 후보의 이득을 클래스별로 분해한다.

    .venv/bin/python experiments/iljun/exp_016_eb_line_screen/run_class_decomposition.py

왜 하는가
  `point_process_eb` 가 seed 42 스크리닝에서 +0.017465 로 1위였지만, 같은 실험에서
  잰 seed 잡음 폭이 최대 0.008279 였다.  39개 후보 중 최대값을 고른 것이므로 순위를
  그대로 믿을 수 없다.

  **이득이 몇 개 클래스에 몰려 있으면 잡음일 가능성이 크고, 넓게 퍼져 있으면 실재일
  가능성이 크다.**  Macro F1 은 26개 클래스 F1 의 단순 평균이므로 기여도를 정확히
  쪼갤 수 있다 — 클래스 c 의 기여 = (blend F1_c − base F1_c) / 26.

  3-seed 확인을 기다리는 동안 이 분해가 판단 근거가 된다.

무엇을 내나
  · 클래스별 F1 (base / 파트너 단독 / 블렌드) 과 기여도
  · 클래스별 복구·역손실 행 수
  · 이득이 상위 몇 개 클래스에 얼마나 몰려 있는지
  · 3-seed 가 있는 p1_eb 와의 대조
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

CONTRACT_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)
THREE_WAY = {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15}
CANDIDATES = {
    "point_process_eb": ("exp-point-process-eb-01_seed42_oof_probabilities.csv",
                         "point_process_eb_"),
    "p1_eb": ("exp-all-class-evidence-ranker-01_seed42_oof_probabilities.csv", "p1_eb__"),
    "ranker": ("exp-all-class-evidence-ranker-01_seed42_oof_probabilities.csv", "ranker__"),
}


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def macro_f1(proba, classes, truth) -> float:
    return float(f1_score(truth, classes[proba.argmax(axis=1)],
                          average="macro", zero_division=0))


def per_class_f1(predicted, classes, truth) -> np.ndarray:
    return f1_score(truth, predicted, average=None, labels=classes, zero_division=0)


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
    for fold in np.unique(folds):
        valid = folds == fold
        weight = pick_weight(base, extra, classes, truth, ~valid)
        blended[valid] = (1 - weight) * base[valid] + weight * extra[valid]
    return blended


def load_partner(root: Path, filename: str, prefix: str, classes, truth):
    matches = list(root.glob(f"experiments/gs/**/result/{filename}"))
    if not matches:
        return None
    frame = pd.read_csv(matches[0])
    if not (frame["true_class"].to_numpy() == truth).all():
        return None
    matrix = frame[[f"{prefix}{name}" for name in classes]].to_numpy(dtype=np.float64)
    row_sum = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(row_sum > 0, row_sum, 1.0)


def decompose(base, partner, classes, truth, folds) -> dict:
    blended = foldlocal_blend(base, partner, classes, truth, folds)
    base_predicted = classes[base.argmax(axis=1)]
    partner_predicted = classes[partner.argmax(axis=1)]
    blend_predicted = classes[blended.argmax(axis=1)]

    base_f1 = per_class_f1(base_predicted, classes, truth)
    partner_f1 = per_class_f1(partner_predicted, classes, truth)
    blend_f1 = per_class_f1(blend_predicted, classes, truth)

    base_ok = base_predicted == truth
    partner_ok = partner_predicted == truth
    rows = []
    for index, name in enumerate(classes):
        mask = truth == name
        rows.append({
            "class": str(name),
            "support": int(mask.sum()),
            "base_f1": float(base_f1[index]),
            "partner_f1": float(partner_f1[index]),
            "blend_f1": float(blend_f1[index]),
            "delta_f1": float(blend_f1[index] - base_f1[index]),
            "contribution": float((blend_f1[index] - base_f1[index]) / len(classes)),
            "recovered_rows": int((mask & ~base_ok & partner_ok).sum()),
            "lost_rows": int((mask & base_ok & ~partner_ok).sum()),
        })
    rows.sort(key=lambda row: -row["contribution"])
    total = sum(row["contribution"] for row in rows)
    positive = [row for row in rows if row["contribution"] > 0]
    top3 = sum(row["contribution"] for row in rows[:3])
    return {
        "base_macro_f1": macro_f1(base, classes, truth),
        "partner_solo_macro_f1": macro_f1(partner, classes, truth),
        "blend_macro_f1": macro_f1(blended, classes, truth),
        "total_gain": total,
        "classes_improved": len(positive),
        "classes_hurt": sum(1 for row in rows if row["contribution"] < 0),
        "top3_share": float(top3 / total) if total else 0.0,
        "per_class": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    out = args.out or (Path(__file__).parent / "artifacts" / "class_decomposition.json")
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

    report = {"seed": args.seed, "base": "three_way", "candidates": {}}
    for label, (filename, prefix) in CANDIDATES.items():
        partner = load_partner(root, filename, prefix, classes, truth)
        if partner is None:
            print(f"건너뜀 {label} — 파일 없음 또는 정렬 불일치", flush=True)
            continue
        report["candidates"][label] = decompose(base, partner, classes, truth, folds)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for label, result in report["candidates"].items():
        print(f"\n{'='*88}")
        print(f"{label}  단독 {result['partner_solo_macro_f1']:.6f} · "
              f"블렌드 {result['blend_macro_f1']:.6f} "
              f"(base {result['base_macro_f1']:.6f}, 이득 {result['total_gain']:+.6f})")
        print(f"  개선 {result['classes_improved']}개 클래스 · "
              f"악화 {result['classes_hurt']}개 · "
              f"상위 3개가 이득의 {result['top3_share']:.0%}")
        print(f"{'-'*88}")
        print(f"{'클래스':10s}{'표본':>6s}{'base F1':>10s}{'파트너':>9s}{'블렌드':>9s}"
              f"{'ΔF1':>10s}{'기여':>11s}{'복구':>6s}{'손실':>6s}")
        head = result["per_class"][:6]
        tail = [row for row in result["per_class"] if row["contribution"] < 0][-4:]
        for row in head + ([{"class": "…"}] if tail else []) + tail:
            if row["class"] == "…":
                print("   …")
                continue
            print(f"{row['class']:10s}{row['support']:>6d}{row['base_f1']:>10.4f}"
                  f"{row['partner_f1']:>9.4f}{row['blend_f1']:>9.4f}"
                  f"{row['delta_f1']:>+10.4f}{row['contribution']:>+11.6f}"
                  f"{row['recovered_rows']:>6d}{row['lost_rows']:>6d}")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
