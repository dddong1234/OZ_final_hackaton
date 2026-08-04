"""앙상블 후보 3-seed 확정 — 가중치를 고정한 뒤 잰다.

    .venv/bin/python experiments/iljun/exp_011_model_ensemble/run_ensemble_confirm.py

설계 원칙
  run_diversity.py 는 seed 42 에서 w 를 스윕해 후보를 골랐다.  그 seed 는 선택에
  쓰였으므로 **낙관 편향이 있다.**  여기서는

    1. 가중치를 코드에 고정한 뒤(아래 BLENDS) 재측정한다 — 사후 선택 금지
    2. 3-seed 평균과 **깨끗한 seed(2024/777)만의 평균**을 따로 보고한다

  2번은 exp_008 에서 쓴 방식이다 — "seed42 는 confusion_8 을 고르는 데 쓴 seed 라
  낙관 편향 포함(+0.00494). 깨끗한 seed 52/62 만: 평균 +0.00144 가 정직한 추정치."

고정한 가중치 (seed 42 스윕 결과에서 확정, 이후 변경 없음)
  lgbm  w=0.2   최고점(+0.00540). 양수 구간 [0.1, 0.3] 의 가운데.
                w=0.4 부터 음수로 꺾이므로 벼랑에서 두 칸 떨어진 지점을 골랐다.
  ovr   w=0.4   최고점은 w=0.5 였으나 그것은 시험 구간의 경계다.
                전 구간(0.1~0.5) 양수라 벼랑 위험은 없고, 한 칸 안쪽을 골랐다.

같이 재는 것
  three_way   champion + ovr + lgbm.  불일치 축이 다르므로(ovr 18.5%, lgbm 39.0%)
              겹치지 않을 수 있다. 단 gs 가 3-way TF-IDF 를 기각한 전례가 있다.
  seed_avg    seed 별 챔피언 확률의 평균.  새 모델 없이 σ 를 직접 줄이는 축이며
              SDH exp_011 권장 후속 4번이다. 아무도 안 돌렸다.
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
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

CHAMPION_CASE = "case_04_shrink10"
SELECTION_SEED = 42          # w 를 고른 seed — 낙관 편향 있음
BLENDS = {                   # 고정. 실행 후 바꾸지 않는다.
    "champion+lgbm(w=0.2)": {"lgbm": 0.2},
    "champion+ovr(w=0.4)": {"ovr": 0.4},
    "three_way(ovr .3/lgbm .15)": {"ovr": 0.30, "lgbm": 0.15},
}


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


def make_candidates(seed: int) -> dict:
    base = dict(solver="lbfgs", C=0.07, max_iter=2000,
                class_weight="balanced", random_state=seed)
    return {
        "champion": LogisticRegression(**base),
        "ovr": OneVsRestClassifier(LogisticRegression(**base), n_jobs=1),
        "lgbm": LGBMClassifier(objective="multiclass", n_estimators=100,
                               learning_rate=0.05, num_leaves=31,
                               class_weight="balanced", random_state=seed,
                               n_jobs=-1, deterministic=True,
                               force_col_wise=True, verbosity=-1),
    }


def collect_oof(sdh, context, labels: pd.Series, seed: int, classes):
    case = sdh.make_cases()[CHAMPION_CASE]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = {name: np.zeros((len(labels), len(classes))) for name in make_candidates(seed)}
    started = perf_counter()
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels.to_numpy()), 1):
        train_matrix, valid_matrix, _, _ = sdh.build_case_matrices(
            context, tr, va, labels, case, inner_seed=seed)
        for name, model in make_candidates(seed).items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(train_matrix, labels.iloc[tr])
            proba[name][va] = model.predict_proba(valid_matrix)
        print(f"      fold {fold}/5 완료", flush=True)
    print(f"    ({(perf_counter()-started)/60:.1f}분)", flush=True)
    return proba


def blend(proba: dict, weights: dict) -> np.ndarray:
    champion_weight = 1.0 - sum(weights.values())
    mixed = champion_weight * proba["champion"]
    for name, weight in weights.items():
        mixed = mixed + weight * proba[name]
    return mixed


def score(mixed, classes, truth) -> dict:
    predicted = classes[mixed.argmax(axis=1)]
    return {"macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, predicted))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 777])
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "ensemble_confirm.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    sdh = load_sdh(root)
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    truth = labels.to_numpy()
    classes = np.unique(truth)
    context = sdh.make_context(train[genes], genes, show_progress=True)

    per_seed, champion_probabilities = {}, []
    for seed in args.seeds:
        print(f"\n  seed {seed}", flush=True)
        proba = collect_oof(sdh, context, labels, seed, classes)
        champion_probabilities.append(proba["champion"])

        row = {"champion": score(proba["champion"], classes, truth)}
        for name, weights in BLENDS.items():
            row[name] = score(blend(proba, weights), classes, truth)
            row[name]["delta"] = row[name]["macro_f1"] - row["champion"]["macro_f1"]
        per_seed[str(seed)] = row
        print(f"    champion {row['champion']['macro_f1']:.5f}", flush=True)
        for name in BLENDS:
            print(f"    {name:28s} {row[name]['macro_f1']:.5f}  {row[name]['delta']:+.5f}",
                  flush=True)

    # seed 평균 — 새 모델 없이 분산만 줄이는 축 (SDH exp_011 권장 후속 4번)
    seed_average = score(np.mean(champion_probabilities, axis=0), classes, truth)
    champion_mean = float(np.mean([per_seed[str(s)]["champion"]["macro_f1"] for s in args.seeds]))
    seed_average["delta_vs_champion_mean"] = seed_average["macro_f1"] - champion_mean

    clean = [s for s in args.seeds if s != SELECTION_SEED]
    summary = {
        "case": CHAMPION_CASE, "seeds": args.seeds, "selection_seed": SELECTION_SEED,
        "clean_seeds": clean, "fixed_weights": BLENDS,
        "champion_macro_f1_mean": champion_mean,
        "champion_macro_f1_std": float(np.std(
            [per_seed[str(s)]["champion"]["macro_f1"] for s in args.seeds], ddof=1)),
        "seed_averaged_champion": seed_average,
        "blends": {}, "per_seed": per_seed,
        "note": ("가중치는 seed 42 스윕에서 고른 뒤 고정했다. seed 42 는 선택에 쓰였으므로 "
                 "낙관 편향이 있어 clean_seeds(2024/777) 평균을 정직한 추정치로 본다."),
    }
    for name in BLENDS:
        deltas = [per_seed[str(s)][name]["delta"] for s in args.seeds]
        clean_deltas = [per_seed[str(s)][name]["delta"] for s in clean]
        summary["blends"][name] = {
            "macro_f1_mean": float(np.mean(
                [per_seed[str(s)][name]["macro_f1"] for s in args.seeds])),
            "delta_mean_all_seeds": float(np.mean(deltas)),
            "delta_mean_clean_seeds": float(np.mean(clean_deltas)),
            "delta_std": float(np.std(deltas, ddof=1)),
            "positive_seeds": f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}",
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    sigma = summary["champion_macro_f1_std"]
    print("\n" + "=" * 78)
    print(f"champion 3-seed  {champion_mean:.5f} ± {sigma:.5f}")
    print("-" * 78)
    print(f"{'구성':30s}{'3seed 평균':>12s}{'전체 delta':>12s}{'깨끗한 seed':>13s}{'양수':>7s}")
    print("-" * 78)
    for name, row in summary["blends"].items():
        print(f"{name:30s}{row['macro_f1_mean']:>12.5f}{row['delta_mean_all_seeds']:>+12.5f}"
              f"{row['delta_mean_clean_seeds']:>+13.5f}{row['positive_seeds']:>7s}")
    print("-" * 78)
    print(f"{'seed 평균 (챔피언 3seed 확률)':30s}{seed_average['macro_f1']:>12.5f}"
          f"{seed_average['delta_vs_champion_mean']:>+12.5f}")
    print("=" * 78)
    print(f"판정 기준 — 3/3 양수 + 깨끗한 seed delta > σ({sigma:.5f}) 이면 채택 후보")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
