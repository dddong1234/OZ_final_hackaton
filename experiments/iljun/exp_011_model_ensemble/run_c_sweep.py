"""챔피언 피처 위에서 규제 C 재탐색.

    .venv/bin/python experiments/iljun/exp_011_model_ensemble/run_c_sweep.py

왜 다시 재는가
  팀 표준 `C=0.07` 은 **GBV 피처(4,235차원)** 에서 튜닝된 값이다(`LR-002q`, 11점 스윕).
  그런데 지금 챔피언은 **8,425차원**이고 그중 26개는 dense 한 enrichment 점수다.
  피처 공간이 근본적으로 달라졌으므로 최적점이 그대로일 이유가 없다.

  C 가 이 데이터에서 얼마나 큰 축인지는 이미 측정돼 있다 — G 단독에서
  C=1.0 → 0.1 이 **+0.04428 (3/3 seed)**. FE 에서 몇 주 쫓던 것이 +0.002~0.016 인 걸
  감안하면 자릿수가 다르다.

  그리고 권일준이 7/31 에 남긴 경고가 방향만 바뀐 채 아직 유효하다 —
  "경수님 전처리 10종과 홍주님 v1·v2 가 전부 C=1.0 에서 매겨졌으므로 순위가
  바뀔 수 있어 재측정을 제안합니다."  지금은 반대다: 전부 C=0.07 에서 매겨졌는데
  피처 공간이 두 배가 됐다.

가설
  C 는 모든 피처에 같은 규제를 건다. 지금 피처는 성격이 둘로 갈린다 —
  희소하고 약한 유전자 8,399개, dense 하고 강한 enrichment 26개.
  약한 쪽에 맞춘 강한 규제(0.07)가 강한 쪽까지 누르고 있다면 최적점은 **위로**
  옮겨갔을 것이다.  그래서 0.07 아래위를 모두 본다.

설계
  피처는 fold 당 한 번만 만들고 C 만 갈아끼운다 — 같은 fold 짝지은 비교.
  screening 은 seed 42.  봉우리가 보이면 그 구간만 3-seed 로 확정한다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

CASE = "case_04_shrink10"
TEAM_STANDARD_C = 0.07
C_GRID = (0.03, 0.07, 0.15, 0.30, 1.00)


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def load_sdh(root: Path):
    source = root / "experiments" / "SDH" / "exp_012_enrichment_stability" / "preprocessing.py"
    spec = importlib.util.spec_from_file_location("sdh_exp012_preprocessing", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sdh_exp012_preprocessing"] = module
    spec.loader.exec_module(module)
    return module


def sweep_seed(sdh, context, labels: pd.Series, seed: int, grid) -> dict:
    truth = labels.to_numpy()
    classes = np.unique(truth)
    case = sdh.make_cases()[CASE]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    predicted = {c: np.empty(len(labels), dtype=object) for c in grid}
    warnings_seen = {c: 0 for c in grid}
    feature_counts = []

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), truth), 1):
        # 피처는 한 번만. C 만 바꾸므로 같은 행렬을 재사용한다.
        train_matrix, valid_matrix, _, meta = sdh.build_case_matrices(
            context, tr, va, labels, case, inner_seed=seed)
        feature_counts.append(meta["total_feature_count"])
        for c in grid:
            started = perf_counter()
            model = LogisticRegression(solver="lbfgs", C=c, max_iter=2000,
                                       class_weight="balanced", random_state=seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train_matrix, labels.iloc[tr])
            predicted[c][va] = model.predict(valid_matrix)
            warnings_seen[c] += sum(issubclass(i.category, ConvergenceWarning) for i in caught)
            print(f"      fold {fold} C={c:<5g} {(perf_counter()-started):4.0f}s", flush=True)

    return {
        "seed": seed,
        "feature_count_mean": float(np.mean(feature_counts)),
        "by_C": {
            str(c): {
                "macro_f1": float(f1_score(truth, predicted[c], average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(truth, predicted[c])),
                "convergence_warning_count": warnings_seen[c],
            } for c in grid
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--grid", type=float, nargs="+", default=list(C_GRID))
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "c_sweep.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    sdh = load_sdh(root)
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    context = sdh.make_context(train[genes], genes, show_progress=True)

    runs = []
    for seed in args.seeds:
        print(f"\n  seed {seed}", flush=True)
        started = perf_counter()
        runs.append(sweep_seed(sdh, context, labels, seed, args.grid))
        print(f"    ({(perf_counter()-started)/60:.1f}분)", flush=True)

    summary = {
        "case": CASE, "seeds": args.seeds, "grid": args.grid,
        "team_standard_C": TEAM_STANDARD_C,
        "feature_count_mean": float(np.mean([r["feature_count_mean"] for r in runs])),
        "by_C": {}, "per_seed": runs,
        "note": ("C=0.07 은 GBV 4,235차원에서 튜닝된 값이다(LR-002q). 챔피언은 8,425차원이고 "
                 "그중 26개가 dense enrichment 점수라 최적점이 옮겨갔을 수 있다."),
    }
    for c in args.grid:
        scores = [r["by_C"][str(c)]["macro_f1"] for r in runs]
        summary["by_C"][str(c)] = {
            "macro_f1_mean": float(np.mean(scores)),
            "accuracy_mean": float(np.mean([r["by_C"][str(c)]["accuracy"] for r in runs])),
            "convergence_warning_count": sum(
                r["by_C"][str(c)]["convergence_warning_count"] for r in runs),
        }
    baseline = summary["by_C"][str(TEAM_STANDARD_C)]["macro_f1_mean"]
    for c in args.grid:
        summary["by_C"][str(c)]["delta_vs_team_standard"] = (
            summary["by_C"][str(c)]["macro_f1_mean"] - baseline)
    best = max(args.grid, key=lambda c: summary["by_C"][str(c)]["macro_f1_mean"])
    summary["best_C"] = best
    summary["best_gain_vs_team_standard"] = summary["by_C"][str(best)]["delta_vs_team_standard"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"챔피언 피처 {summary['feature_count_mean']:,.0f}차원 · seeds {args.seeds}")
    print("-" * 64)
    print(f"{'C':>8s}{'Macro F1':>12s}{'Accuracy':>12s}{'0.07 대비':>12s}{'수렴경고':>10s}")
    print("-" * 64)
    for c in args.grid:
        row = summary["by_C"][str(c)]
        mark = "  ← 팀 표준" if c == TEAM_STANDARD_C else ("  ★ 최고" if c == best else "")
        print(f"{c:>8g}{row['macro_f1_mean']:>12.5f}{row['accuracy_mean']:>12.5f}"
              f"{row['delta_vs_team_standard']:>+12.5f}{row['convergence_warning_count']:>10d}{mark}")
    print("=" * 64)
    gain = summary["best_gain_vs_team_standard"]
    if best == TEAM_STANDARD_C:
        print("→ 팀 표준 0.07 이 여전히 최적이다. C 축은 닫힌다.")
    else:
        print(f"→ 최적 C={best:g}, 팀 표준 대비 {gain:+.5f}. 3-seed 로 확정 필요.")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
