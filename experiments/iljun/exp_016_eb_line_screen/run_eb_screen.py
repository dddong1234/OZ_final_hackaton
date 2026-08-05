"""경수님 EB 계열 전수 스크리닝 — 우리 앙상블에 무엇을 더할 수 있나.

    .venv/bin/python experiments/iljun/exp_016_eb_line_screen/run_eb_screen.py --base three_way

왜 하는가
  exp_015 에서 GBDT 축이 닫혔다(XGB -0.0013, CatBoost -0.0014). 같은 기준선 위에서
  부호가 갈린 것은 계열이 다른 EB 라인뿐이었다(gate +0.0070, p1_eb +0.0094).
  그런데 그때 본 것은 파일 두 개였고, 지금 main 에는 OOF 확률 파일이 22개 있다.
  전부 같은 절차로 한 번에 재서 후보를 좁힌다.

무엇을 하나
  · 파일마다 변형(variant)을 자동 탐지한다 — 26개 클래스를 모두 덮는 접두어를 찾는다.
    `gate__ACC` 처럼 이중 밑줄도, `p1_ACC` 처럼 단일 밑줄도 같은 규칙으로 잡힌다.
  · 행 정렬은 `true_class` 로 검증한다. 어긋나면 건너뛴다.
  · 우리 base 와 계약 5점 grid 로 **fold-local** 블렌드한다.
  · 계약 §4 다양성 지표를 함께 낸다.

  모델을 학습하지 않는다. 양쪽 OOF 확률만 쓴다.

한계
  · seed 42 만 쓴다. 파일 대부분이 seed 42 뿐이고, 경수님 seed 세트가 실험마다
    42/52/62/777/2024/31415 로 갈려 있어 3-seed 매칭이 되는 라인이 거의 없다.
    따라서 이 결과는 **스크리닝**이고 계약상 채택 판정 단위가 아니다.
  · 파트너 OOF 는 우리 outer fold 에 중첩되어 있지 않다(그쪽 모델 재학습 불가).
    잔여 낙관이 있으며 exp_014 와 같은 한계다.
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
THREE_WAY = {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15}   # 사전 선언된 고정 조합


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def macro_f1(proba: np.ndarray, classes: np.ndarray, truth: np.ndarray) -> float:
    return float(f1_score(truth, classes[proba.argmax(axis=1)],
                          average="macro", zero_division=0))


def detect_variants(columns, classes) -> dict[str, list[str]]:
    """26개 클래스를 모두 덮는 접두어를 찾는다. 구분자 종류를 가리지 않는다."""

    candidates: dict[str, set] = {}
    for column in columns:
        for name in classes:
            if column.endswith(name):
                prefix = column[: -len(name)]
                candidates.setdefault(prefix, set()).add(name)
    return {prefix: [f"{prefix}{name}" for name in classes]
            for prefix, covered in candidates.items()
            if len(covered) == len(classes) and prefix != ""}


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


def diversity(base, other, classes, truth) -> dict:
    base_predicted = classes[base.argmax(axis=1)]
    other_predicted = classes[other.argmax(axis=1)]
    base_ok = base_predicted == truth
    other_ok = other_predicted == truth
    oracle = np.where(base_ok, base_predicted,
                      np.where(other_ok, other_predicted, base_predicted))
    return {
        "disagreement": float((base_predicted != other_predicted).mean()),
        "recovery_rate": float((~base_ok & other_ok).sum() / max((~base_ok).sum(), 1)),
        "reverse_loss_rate": float((base_ok & ~other_ok).sum() / max(base_ok.sum(), 1)),
        "double_fault": float((~base_ok & ~other_ok).mean()),
        "oracle_macro_f1": float(f1_score(truth, oracle, average="macro", zero_division=0)),
        "probability_correlation": float(np.corrcoef(base.ravel(), other.ravel())[0, 1]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=("anchor", "three_way"), default="three_way")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--any-seed", action="store_true",
                        help="파일명 seed 가 달라도 포함한다(비교 불가 — 잡음 측정용)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    out = args.out or (Path(__file__).parent / "artifacts" / f"eb_screen_{args.base}.json")
    oof_dir = root / "experiments" / "iljun" / "results" / "contract_probabilities"

    anchor_frame = pd.read_csv(oof_dir / f"oof_champion_auto_noexact_seed{args.seed}.csv")
    classes = np.array([c[len("prob_"):] for c in anchor_frame.columns
                        if c.startswith("prob_")])
    columns = [f"prob_{name}" for name in classes]
    truth = anchor_frame["SUBCLASS"].to_numpy()
    folds = anchor_frame["fold"].to_numpy()

    base = anchor_frame[columns].to_numpy(dtype=np.float64)
    if args.base == "three_way":
        base = base * THREE_WAY["champion"]
        for name in ("ovr", "lgbm"):
            block = pd.read_csv(
                oof_dir / f"oof_{name}_auto_noexact_seed{args.seed}.csv")[columns]
            base = base + block.to_numpy(dtype=np.float64) * THREE_WAY[name]
    base_score = macro_f1(base, classes, truth)
    print(f"base={args.base} seed={args.seed} → {base_score:.6f}\n", flush=True)

    rows = []
    files = sorted(root.glob("experiments/gs/**/result/*oof_probabilit*.csv"))
    for path in files:
        # 파일명에 박힌 seed 가 우리와 다르면 같은 분할이 아니다.  행 단위 정직성은
        # 유지되지만 후보 간 비교가 성립하지 않으므로 기본적으로 제외한다.
        stamped = re.search(r"_seed(\d+)", path.name)
        if stamped and int(stamped.group(1)) != args.seed and not args.any_seed:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as error:                       # 손상/비정형 파일은 건너뛴다
            print(f"  건너뜀 {path.name} — 읽기 실패 {error}", flush=True)
            continue
        if "seed" in frame.columns:
            frame = frame[frame["seed"] == args.seed].reset_index(drop=True)
        if len(frame) != len(truth) or "true_class" not in frame.columns:
            print(f"  건너뜀 {path.name} — 행 {len(frame)} 또는 true_class 없음", flush=True)
            continue
        if not (frame["true_class"].to_numpy() == truth).all():
            print(f"  건너뜀 {path.name} — true_class 가 train 순서와 다름", flush=True)
            continue

        variants = detect_variants(frame.columns, classes)
        if not variants:
            print(f"  건너뜀 {path.name} — 26 클래스를 덮는 접두어 없음", flush=True)
            continue

        for prefix, variant_columns in sorted(variants.items()):
            matrix = frame[variant_columns].to_numpy(dtype=np.float64)
            row_sum = matrix.sum(axis=1, keepdims=True)
            matrix = matrix / np.where(row_sum > 0, row_sum, 1.0)
            blended, weights = foldlocal_blend(base, matrix, classes, truth, folds)
            score = macro_f1(blended, classes, truth)
            rows.append({
                "file": path.name,
                "variant": prefix.rstrip("_"),
                "solo_macro_f1": macro_f1(matrix, classes, truth),
                "blend_macro_f1": score,
                "gain": score - base_score,
                "weights": weights,
                "diversity": diversity(base, matrix, classes, truth),
            })

    rows.sort(key=lambda row: -row["gain"])
    report = {"base": args.base, "base_macro_f1": base_score, "seed": args.seed,
              "contract_grid": list(CONTRACT_GRID),
              "seed_matched_only": not args.any_seed,
              "candidates": rows,
              "note": ("seed 42 스크리닝이다. 파트너 OOF 는 우리 fold 에 중첩되어 있지 "
                       "않으므로 잔여 낙관이 있고, 계약상 채택 판정 단위(3-seed)가 아니다.")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    width = 96
    print("\n" + "=" * width)
    print(f"{'변형':32s}{'단독':>10s}{'블렌드':>11s}{'이득':>11s}"
          f"{'복구율':>9s}{'역손실':>9s}{'상관':>8s}")
    print("-" * width)
    for row in rows[:15]:
        d = row["diversity"]
        print(f"{row['variant'][:31]:32s}{row['solo_macro_f1']:>10.5f}"
              f"{row['blend_macro_f1']:>11.6f}{row['gain']:>+11.6f}"
              f"{d['recovery_rate']:>9.1%}{d['reverse_loss_rate']:>9.1%}"
              f"{d['probability_correlation']:>8.3f}")
    print("=" * width)
    print(f"base({args.base}) {base_score:.6f} · 후보 {len(rows)}개 · 상위 15개 표시")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
