"""앙상블 경로 permutation-label 감사.

    .venv/bin/python experiments/iljun/exp_012_cross_member_ensemble/run_permutation_ensemble.py

왜 하는가
  단일 모델 FE 는 두 번 감사받았다 — exp_009(문서 기준), exp_010(SDH 실제 코드).
  둘 다 PASS 다.  그러나 **앙상블 경로로는 아무도 돌리지 않았다.**
  ENS-011a 가 챔피언 단독 대비 +0.01608 을 주장하는데, 그 이득이 진짜 신호인지
  아니면 (a) supervised FE 누수가 앙상블에서 증폭된 것인지 (b) 같은 OOF 위에서
  가중치를 고른 선택 편향인지 구분된 적이 없다.

무엇을 재는가
  label 을 섞으면 모든 모델이 우연 수준으로 붕괴해야 하고, 모델 간 차이도
  사라져야 하므로 블렌드 이득도 0 이어야 한다.  섞인 label 에서도 이득이 남으면
  그 이득의 출처는 label 이 아니다.

  1. 고정 가중치(0.55/0.30/0.15) 블렌드 이득 — 실제 label vs 섞인 label
  2. **가중치를 같은 OOF 에서 탐색했을 때의 최대 이득** — 섞인 label 에서 나오는
     값이 곧 "탐색만으로 만들어낼 수 있는 가짜 이득"의 크기다.  다양성 문서 §5.3
     이 경고하는 과적합을 수치로 만든다.

주의
  test.csv 를 읽지 않는다.  CV 에 필요 없고, 안 읽는 것이 가장 강한 누수 방어다.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

CASE = "case_04_shrink10"
FIXED_WEIGHTS = {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15}   # ENS-011a 확정값
WEIGHT_STEP = 0.05


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


def make_models(seed: int) -> dict:
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


def macro_f1(proba: np.ndarray, classes: np.ndarray, truth: np.ndarray) -> float:
    return float(f1_score(truth, classes[proba.argmax(axis=1)],
                          average="macro", zero_division=0))


def oof_probabilities(sdh, context, labels: pd.Series, case, seed: int, tag: str):
    """한 번의 fold 순회로 세 모델의 OOF 확률을 모두 만든다."""

    started = perf_counter()
    truth = labels.to_numpy()
    classes = np.unique(truth)
    proba = {name: np.zeros((len(truth), len(classes))) for name in FIXED_WEIGHTS}
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    warnings_seen, feature_counts = 0, []

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(truth)), truth), 1):
        train_matrix, valid_matrix, _, meta = sdh.build_case_matrices(
            context, tr, va, labels, case, inner_seed=seed)
        feature_counts.append(meta["total_feature_count"])
        for name, model in make_models(seed).items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train_matrix, labels.iloc[tr])
            warnings_seen += sum(issubclass(i.category, ConvergenceWarning) for i in caught)
            assert list(model.classes_) == list(classes), "클래스 순서가 fold 마다 다릅니다"
            proba[name][va] = model.predict_proba(valid_matrix)
        print(f"      fold {fold}/5  피처 {meta['total_feature_count']:,}", flush=True)

    print(f"    ({(perf_counter()-started)/60:.1f}분) {tag} 완료", flush=True)
    return proba, classes, {"feature_count_mean": float(np.mean(feature_counts)),
                            "convergence_warning_count": warnings_seen}


def simplex_grid(step: float) -> list[dict]:
    """세 모델 가중치의 모든 조합(합 1, step 간격)."""

    ticks = int(round(1.0 / step))
    grid = []
    for a, b in itertools.product(range(ticks + 1), repeat=2):
        c = ticks - a - b
        if c < 0:
            continue
        grid.append({"champion": a * step, "ovr": b * step, "lgbm": c * step})
    return grid


def evaluate(proba: dict, classes, truth, meta: dict, tag: str) -> dict:
    """단독 점수, 고정 가중치 블렌드, 가중치 탐색 최대치를 한 번에 낸다."""

    singles = {name: macro_f1(proba[name], classes, truth) for name in proba}
    fixed = sum(FIXED_WEIGHTS[name] * proba[name] for name in FIXED_WEIGHTS)
    fixed_score = macro_f1(fixed, classes, truth)

    best = {"weights": None, "macro_f1": -1.0}
    for weights in simplex_grid(WEIGHT_STEP):
        mixed = sum(weights[name] * proba[name] for name in weights)
        score = macro_f1(mixed, classes, truth)
        if score > best["macro_f1"]:
            best = {"weights": weights, "macro_f1": score}

    return {
        "tag": tag,
        "singles": singles,
        "champion_solo": singles["champion"],
        "fixed_blend": fixed_score,
        "fixed_gain": fixed_score - singles["champion"],
        "searched_blend": best["macro_f1"],
        "searched_gain": best["macro_f1"] - singles["champion"],
        "searched_weights": best["weights"],
        "search_grid_size": len(simplex_grid(WEIGHT_STEP)),
        **meta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutation-seed", type=int, default=20260804)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "permutation_ensemble.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    sdh = load_sdh(root)
    case = sdh.make_cases()[CASE]

    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    # test 는 읽지 않는다 — 어휘도 선택도 train 만으로 정해진다.
    context = sdh.make_context(train[genes], genes, show_progress=True)

    print(f"\n[1] 실제 label — ENS-011a 재현 (seed {args.seed})", flush=True)
    real_proba, classes, real_meta = oof_probabilities(
        sdh, context, labels, case, args.seed, "실제 label")
    real = evaluate(real_proba, classes, labels.to_numpy(), real_meta, "real")

    rng = np.random.default_rng(args.permutation_seed)
    shuffled = pd.Series(labels.to_numpy()[rng.permutation(len(labels))], name="SUBCLASS")
    print(f"\n[2] label 무작위 셔플 (permutation_seed={args.permutation_seed})", flush=True)
    permuted_proba, permuted_classes, permuted_meta = oof_probabilities(
        sdh, context, shuffled, case, args.seed, "섞인 label")
    permuted = evaluate(permuted_proba, permuted_classes, shuffled.to_numpy(),
                        permuted_meta, "permuted")

    chance = 1.0 / len(classes)
    verdict = ("PASS"
               if permuted["fixed_blend"] < chance * 3
               and permuted["fixed_gain"] < 0.01
               else "FAIL")
    report = {
        "seed": args.seed,
        "permutation_seed": args.permutation_seed,
        "case": CASE,
        "fixed_weights": FIXED_WEIGHTS,
        "source": "experiments/SDH/exp_012_enrichment_stability/preprocessing.py",
        "n_classes": int(len(classes)),
        "chance_level": chance,
        "real": real,
        "permuted": permuted,
        "selection_bias_ceiling": permuted["searched_gain"],
        "reference": {"ENS-011a_seed42_cv": 0.54202,
                      "ENS-011a_clean_seed_gain": 0.01608,
                      "exp012_champion_seed42": 0.52918},
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    width = 74
    print("\n" + "=" * width)
    print(f"{'':16s}{'champion':>12s}{'ovr':>10s}{'lgbm':>10s}"
          f"{'고정blend':>12s}{'이득':>10s}")
    print("-" * width)
    for row in (real, permuted):
        name = "실제 label" if row["tag"] == "real" else "섞인 label"
        print(f"{name:14s}{row['singles']['champion']:>12.6f}"
              f"{row['singles']['ovr']:>10.6f}{row['singles']['lgbm']:>10.6f}"
              f"{row['fixed_blend']:>12.6f}{row['fixed_gain']:>+10.6f}")
    print("-" * width)
    print(f"우연 수준 (1/{len(classes)}) {chance:.6f}")
    print(f"가중치 탐색 최대 이득   실제 {real['searched_gain']:+.6f} · "
          f"섞인 {permuted['searched_gain']:+.6f}  "
          f"(격자 {real['search_grid_size']}개)")
    print("=" * width)
    if verdict == "PASS":
        print("✅ PASS — 섞인 label 에서 블렌드가 우연 수준을 못 넘고 이득도 사라진다.")
    else:
        print("🚨 FAIL — 섞인 label 에서도 이득이 남는다. 이득의 출처가 label 이 아니다.")
    print(f"\n선택 편향 상한: 같은 OOF 에서 가중치를 탐색하면 순수 잡음에서도 "
          f"{permuted['searched_gain']:+.5f} 를 만들어낼 수 있다.")
    print(f"고정 가중치 이득 {real['fixed_gain']:+.5f} 가 이 값보다 충분히 크면 "
          f"탐색 편향으로 설명되지 않는다.")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
