"""팀원 간 블렌드를 계약의 fold-local 절차로 다시 잰다.

    # 경수님 OOF 를 로컬로 추출 (gitignore 되는 results/ 로)
    git fetch origin gs/exp_004
    B=origin/gs/exp_004:experiments/gs/notebooks/exp_model_004/result
    git show $B/exp-all-class-evidence-ranker-01_seed42_oof_probabilities.csv \\
      > experiments/iljun/results/gs_oof/gs_ranker_seed42.csv
    for s in 52 62; do
      git show $B/exp-selective-eb-gate-01_seed${s}_oof_probabilities.csv \\
        > experiments/iljun/results/gs_oof/gs_gate_seed${s}.csv
    done

    .venv/bin/python experiments/iljun/exp_013_contract_compliance/run_cross_member_foldlocal.py

왜 다시 재나
  exp_012 의 팀원 간 블렌드 이득(+0.011~+0.012)은 **전체 OOF 에서 가중치를 고른**
  값이다.  계약은 그것을 금지하고, 같은 저장소에서 그 방식의 편향이 +0.003290 으로
  실측됐다(`foldlocal_blend_auto.json`).  따라서 기존 숫자는 그만큼 깎아 읽어야 하며,
  채택 판단을 하려면 계약 절차로 다시 재야 한다.

  seed 도 맞춘다.  exp_012 는 seed 42 하나였고 계약은 42/52/62 를 요구한다.
  gate·p1_eb 는 세 seed 가 모두 있고 ranker 는 seed 42 만 있다.

무엇을 하나
  각 outer fold 에서 **그 fold 를 뺀 행들만으로** 계약 grid(0.05~0.30)를 골라
  해당 fold 에 적용한다.  전체 OOF 최적값도 같이 계산해 두 방식의 차이를 남긴다.
  모델을 다시 학습하지 않는다 — 양쪽 OOF 확률만 쓴다.

한계 (반드시 같이 읽을 것)
  경수님 모델을 우리 outer fold 안에서 다시 학습할 수 없으므로, 경수님의 OOF 는
  우리 fold 에 **중첩되어 있지 않다.**  outer-train 행에 대한 경수님 예측은 일부
  outer-valid 행을 학습에 포함한 모델이 낸 것이다.  그래서 이 추정치도 완전히
  편향이 없는 것은 아니고, 전체 OOF 선택보다 나을 뿐이다.  완전한 중첩 추정은
  경수님이 fold 내부 OOF 를 내줘야 가능하다.

  또한 경수님 분할이 우리와 같은지 확인할 방법이 없다(파일에 fold 열이 없다).
  팀 표준이 StratifiedKFold-5 · random_state=seed 이므로 같을 것으로 보지만
  검증되지 않은 가정이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ANCHOR_PATTERN = "oof_champion_auto_noexact_seed{seed}.csv"
CONTRACT_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)   # 계약이 명시한 파트너 쪽 가중치
PARTNER_FILES = {42: "gs_ranker_seed42.csv", 52: "gs_gate_seed52.csv", 62: "gs_gate_seed62.csv"}


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def macro_f1(proba: np.ndarray, classes: np.ndarray, truth: np.ndarray) -> float:
    return float(f1_score(truth, classes[proba.argmax(axis=1)],
                          average="macro", zero_division=0))


def load_partner(path: Path, prefix: str, classes: np.ndarray, truth: np.ndarray):
    """한 변형의 확률 행렬을 우리 클래스 순서로 맞춰 돌려준다."""

    frame = pd.read_csv(path)
    assert len(frame) == len(truth), f"행 수 불일치: {len(frame)} vs {len(truth)}"
    assert (frame["true_class"].to_numpy() == truth).all(), \
        "true_class 가 train 순서와 다릅니다 — 행 정렬을 확인하세요"
    columns = [f"{prefix}{name}" for name in classes]
    if any(column not in frame.columns for column in columns):
        return None
    matrix = frame[columns].to_numpy(dtype=np.float64)
    row_sum = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(row_sum > 0, row_sum, 1.0)


def fold_assignment(truth: np.ndarray, seed: int) -> np.ndarray:
    folds = np.zeros(len(truth), dtype=np.int16)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(truth)), truth), 1):
        folds[valid_index] = fold
    return folds


def pick_weight(anchor, partner, classes, truth, rows) -> float:
    """주어진 행 집합에서만 계약 grid 를 훑어 최적 가중치를 고른다."""

    best_weight, best_score = 0.0, macro_f1(anchor[rows], classes, truth[rows])
    for weight in CONTRACT_GRID:
        mixed = (1 - weight) * anchor[rows] + weight * partner[rows]
        score = macro_f1(mixed, classes, truth[rows])
        if score > best_score:
            best_weight, best_score = weight, score
    return best_weight


def diversity(base, partner, classes, truth) -> dict:
    """계약 §4 지표. 기준 모델이 틀린 것을 파트너가 복구하는지가 핵심이다."""

    base_predicted = classes[base.argmax(axis=1)]
    partner_predicted = classes[partner.argmax(axis=1)]
    base_ok = base_predicted == truth
    partner_ok = partner_predicted == truth
    # oracle — 둘 중 맞는 쪽을 매번 고른다고 가정한 이론적 상한(실제 점수 아님)
    oracle_predicted = np.where(base_ok, base_predicted,
                                np.where(partner_ok, partner_predicted, base_predicted))
    return {
        "disagreement": float((base_predicted != partner_predicted).mean()),
        "recovery_rate": float((~base_ok & partner_ok).sum() / max((~base_ok).sum(), 1)),
        "reverse_loss_rate": float((base_ok & ~partner_ok).sum() / max(base_ok.sum(), 1)),
        "double_fault": float((~base_ok & ~partner_ok).mean()),
        "oracle_macro_f1": float(f1_score(truth, oracle_predicted,
                                          average="macro", zero_division=0)),
        "probability_correlation": float(np.corrcoef(base.ravel(), partner.ravel())[0, 1]),
    }


def evaluate(anchor, partner, classes, truth, folds) -> dict:
    """fold-local 선택(계약)과 전체 OOF 선택(기존 방식)을 같은 데이터에서 비교."""

    foldlocal = np.zeros_like(anchor)
    choices = {}
    for fold in np.unique(folds):
        valid = folds == fold
        train = ~valid
        weight = pick_weight(anchor, partner, classes, truth, train)
        choices[int(fold)] = weight
        foldlocal[valid] = (1 - weight) * anchor[valid] + weight * partner[valid]

    whole = pick_weight(anchor, partner, classes, truth, np.ones(len(truth), dtype=bool))
    whole_blend = (1 - whole) * anchor + whole * partner

    anchor_score = macro_f1(anchor, classes, truth)
    foldlocal_score = macro_f1(foldlocal, classes, truth)
    whole_score = macro_f1(whole_blend, classes, truth)
    predicted = classes[foldlocal.argmax(axis=1)]
    return {
        "anchor_macro_f1": anchor_score,
        "partner_solo_macro_f1": macro_f1(partner, classes, truth),
        "foldlocal_macro_f1": foldlocal_score,
        "foldlocal_accuracy": float(accuracy_score(truth, predicted)),
        "foldlocal_gain": foldlocal_score - anchor_score,
        "foldlocal_weights": choices,
        "whole_oof_weight": whole,
        "whole_oof_macro_f1": whole_score,
        "whole_oof_gain": whole_score - anchor_score,
        "selection_bias": whole_score - foldlocal_score,
        "diversity": diversity(anchor, partner, classes, truth),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--prefixes", nargs="+", default=["gate__", "p1_eb__", "ranker__"])
    parser.add_argument("--base", choices=("anchor", "three_way"), default="anchor",
                        help="anchor = LR 단독(계약 §7 단위). three_way = 우리 앙상블 위에 얹기")
    parser.add_argument("--out", type=Path,
                        default=None)
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = (Path(__file__).parent / "artifacts"
                    / f"cross_member_foldlocal_{args.base}.json")

    root = find_root(Path(__file__).resolve())
    oof_dir = root / "experiments" / "iljun" / "results" / "contract_probabilities"
    partner_dir = root / "experiments" / "iljun" / "results" / "gs_oof"

    report = {"contract_grid": list(CONTRACT_GRID),
              "base": args.base,
              "anchor": "champion LR (auto contrast, exact 제거)",
              "note": ("경수님 OOF 는 우리 outer fold 에 중첩되어 있지 않다. 경수님 모델을 "
                       "다시 학습할 수 없으므로 이 추정치도 완전히 편향이 없지는 않고, "
                       "전체 OOF 선택보다 나을 뿐이다."),
              "runs": {}}

    for seed in args.seeds:
        anchor_path = oof_dir / ANCHOR_PATTERN.format(seed=seed)
        partner_path = partner_dir / PARTNER_FILES.get(seed, "")
        if not anchor_path.exists() or not partner_path.exists():
            print(f"[seed {seed}] 건너뜀 — 파일 없음", flush=True)
            continue

        anchor_frame = pd.read_csv(anchor_path)
        classes = np.array([c[len("prob_"):] for c in anchor_frame.columns
                            if c.startswith("prob_")])
        anchor = anchor_frame[[f"prob_{name}" for name in classes]].to_numpy(dtype=np.float64)
        truth = anchor_frame["SUBCLASS"].to_numpy()
        if args.base == "three_way":
            # 사전 선언된 고정 조합(탐색 아님) — 우리 앙상블을 한 덩어리로 본다.
            for name, weight in (("champion", 0.55), ("ovr", 0.30), ("lgbm", 0.15)):
                part = pd.read_csv(oof_dir / f"oof_{name}_auto_noexact_seed{seed}.csv")
                block = part[[f"prob_{c}" for c in classes]].to_numpy(dtype=np.float64)
                anchor = block * weight if name == "champion" else anchor + block * weight
        folds = fold_assignment(truth, seed)
        # ① 이 쓴 분할과 같은지 확인 — 확률 파일의 fold 열과 대조한다.
        assert (anchor_frame["fold"].to_numpy() == folds).all(), "fold 재현이 어긋납니다"

        print(f"\n[seed {seed}] anchor {macro_f1(anchor, classes, truth):.6f} "
              f"· 파트너 파일 {partner_path.name}", flush=True)
        for prefix in args.prefixes:
            partner = load_partner(partner_path, prefix, classes, truth)
            if partner is None:
                continue
            result = evaluate(anchor, partner, classes, truth, folds)
            report["runs"].setdefault(prefix, {})[seed] = result
            print(f"    {prefix:10s} 단독 {result['partner_solo_macro_f1']:.6f} · "
                  f"fold-local {result['foldlocal_macro_f1']:.6f} "
                  f"({result['foldlocal_gain']:+.6f}) · "
                  f"전체OOF {result['whole_oof_macro_f1']:.6f} "
                  f"({result['whole_oof_gain']:+.6f}, w={result['whole_oof_weight']}) · "
                  f"편향 {result['selection_bias']:+.6f}", flush=True)

    for prefix, by_seed in report["runs"].items():
        gains = [row["foldlocal_gain"] for row in by_seed.values()]
        biases = [row["selection_bias"] for row in by_seed.values()]
        report.setdefault("summary", {})[prefix] = {
            "seeds": sorted(by_seed),
            "foldlocal_macro_f1_mean": float(np.mean(
                [row["foldlocal_macro_f1"] for row in by_seed.values()])),
            "foldlocal_gain_mean": float(np.mean(gains)),
            "foldlocal_gain_std": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
            "positive_seeds": f"{sum(1 for g in gains if g > 0)}/{len(gains)}",
            "selection_bias_mean": float(np.mean(biases)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    width = 82
    print("\n" + "=" * width)
    print(f"{'파트너':12s}{'seeds':>10s}{'fold-local 평균':>18s}{'이득':>12s}"
          f"{'양수':>8s}{'선택 편향':>14s}")
    print("-" * width)
    for prefix, s in report.get("summary", {}).items():
        print(f"{prefix:12s}{str(s['seeds']):>10s}{s['foldlocal_macro_f1_mean']:>18.6f}"
              f"{s['foldlocal_gain_mean']:>+12.6f}{s['positive_seeds']:>8s}"
              f"{s['selection_bias_mean']:>+14.6f}")
    print("=" * width)
    print("채택 기준(계약 §5.3) — 3-seed 평균 +0.001 이상 · 3개 중 최소 2개 개선 ·"
          " 특정 seed 큰 하락 없음")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
