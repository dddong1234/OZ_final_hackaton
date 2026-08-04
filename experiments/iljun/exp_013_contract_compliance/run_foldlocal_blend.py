"""② 앙상블 가중치를 계약대로 fold-local 로 다시 고른다.

    .venv/bin/python experiments/iljun/exp_013_contract_compliance/run_foldlocal_blend.py \\
        --contrast auto --seeds 42 52 62

계약 조항 (팀 Notion `팀 모델 분업·안전한 앙상블 운영 계약`)
  > 가중치는 public leaderboard나 전체 OOF를 보고 사후 선택하지 않는다.
  > 각 **outer fold의 train split 내부 OOF**에서만 사전 정의된 작은 grid
  > `LR:새 모델 = 95:5, 90:10, 85:15, 80:20, 70:30` 을 선택하고,
  > 고른 가중치를 해당 outer validation 에 적용한다.
  > 전체 outer OOF 는 ... 성능을 **평가·기록**하는 데만 사용한다.

무엇이 달라지나
  ENS-011a 의 0.55/0.30/0.15 는 seed 42 **전체 outer OOF** 스윕에서 골랐다.
  계약은 그것을 금지한다.  여기서는 outer fold 마다 그 fold 의 train split 안에서
  다시 3-fold inner OOF 를 만들고, 거기서만 가중치를 고른 뒤 해당 outer validation
  에 적용한다.  outer OOF 는 마지막에 점수를 적는 용도로만 쓴다.

왜 이 절차가 필요한가 — 실측 근거
  `exp_012_cross_member_ensemble/artifacts/permutation_ensemble.json` 에서, 전체
  OOF 위에서 231점 격자를 탐색하면 **label 을 섞은 데이터에서도 +0.006845** 가
  나왔다.  신호가 0 인 데이터에서 탐색만으로 만들어지는 이득이 그만큼이다.
  fold-local 선택은 이 편향을 구조적으로 제거한다.

grid 에 대하여
  계약의 grid 는 2-model(`LR:새 모델`) 전용이라 우리 three-way 를 표현하지 못한다.
  그래서 두 가지를 **모두** 보고한다.
    · 엄격 준수 — LR + 파트너 1개, 계약 grid 5점 그대로
    · 확장 — three-way 를 위한 사전 선언 grid(아래 THREE_WAY_GRID, 탐색 아님)
  확장 grid 는 실행 전에 코드에 박아 둔 고정 목록이다.  결과를 보고 늘리지 않는다.

비용
  outer 5 × (inner 3 + outer 1) 번 설계행렬을 만든다.  seed 당 20분 안팎.
"""
from __future__ import annotations

import argparse
import dataclasses
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

CASE = "case_04_shrink10"
ANCHOR = "champion"                       # 역할 분담상 우리가 관리하는 기준축 LR
PARTNERS = ("ovr", "lgbm")
LEGACY_WEIGHTS = {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15}   # 비교용(계약 이전 값)

# 계약이 명시한 2-model grid — 파트너 쪽 가중치
CONTRACT_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)

# three-way 확장. 실행 전에 선언된 고정 목록이며 결과를 보고 바꾸지 않는다.
THREE_WAY_GRID = (
    {"champion": 1.00, "ovr": 0.00, "lgbm": 0.00},
    {"champion": 0.90, "ovr": 0.05, "lgbm": 0.05},
    {"champion": 0.85, "ovr": 0.10, "lgbm": 0.05},
    {"champion": 0.80, "ovr": 0.10, "lgbm": 0.10},
    {"champion": 0.75, "ovr": 0.15, "lgbm": 0.10},
    {"champion": 0.70, "ovr": 0.20, "lgbm": 0.10},
    {"champion": 0.70, "ovr": 0.15, "lgbm": 0.15},
    {"champion": 0.60, "ovr": 0.25, "lgbm": 0.15},
    {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15},
)


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def fit_fold(sdh, final, context, labels, case, seed, contrast,
             train_index, valid_index, drop_exact: bool = False) -> tuple[dict, int, int]:
    """한 분할에서 세 모델을 학습하고 valid 확률을 돌려준다."""

    baseline = sdh.B04_CANDIDATE
    pairs = (final.discover_confusion_pairs(context.cache, train_index, labels, seed)
             if contrast == "auto" else baseline.contrast_pairs)
    changes = {"contrast_pairs": pairs}
    if drop_exact:
        # 고정 exact mutation 4개 제거 — 2026-08-04 팀 공지.
        changes["exact_events"] = ()
    sdh.B04_CANDIDATE = dataclasses.replace(baseline, **changes)
    try:
        train_matrix, valid_matrix, _, meta = sdh.build_case_matrices(
            context, train_index, valid_index, labels, case, inner_seed=seed)
    finally:
        sdh.B04_CANDIDATE = baseline

    proba, warned = {}, 0
    for name, model in make_models(seed).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, labels.iloc[train_index])
        warned += sum(issubclass(i.category, ConvergenceWarning) for i in caught)
        proba[name] = model.predict_proba(valid_matrix)
    return proba, warned, meta["total_feature_count"]


def select_weights(inner_proba: dict, classes, inner_truth) -> dict:
    """inner OOF 에서만 가중치를 고른다. 여기가 계약의 핵심이다."""

    selections = {}

    for partner in PARTNERS:                       # 엄격 준수 — 계약 grid 그대로
        best = {"weight": 0.0, "macro_f1": macro_f1(inner_proba[ANCHOR], classes, inner_truth)}
        for weight in CONTRACT_GRID:
            mixed = (1 - weight) * inner_proba[ANCHOR] + weight * inner_proba[partner]
            value = macro_f1(mixed, classes, inner_truth)
            if value > best["macro_f1"]:
                best = {"weight": weight, "macro_f1": value}
        selections[f"pair_{partner}"] = best

    best_three = {"weights": None, "macro_f1": -1.0}
    for weights in THREE_WAY_GRID:                 # 확장 — 사전 선언 목록
        mixed = sum(weights[name] * inner_proba[name] for name in weights)
        value = macro_f1(mixed, classes, inner_truth)
        if value > best_three["macro_f1"]:
            best_three = {"weights": dict(weights), "macro_f1": value}
    selections["three_way"] = best_three
    return selections


def run_seed(sdh, final, context, labels, case, seed: int, contrast: str,
             inner_splits: int, drop_exact: bool = False) -> dict:
    started = perf_counter()
    truth = labels.to_numpy()
    classes = np.unique(truth)
    n = len(truth)

    outer_solo = {name: np.zeros((n, len(classes))) for name in ("champion", "ovr", "lgbm")}
    outer_foldlocal = {key: np.zeros((n, len(classes)))
                       for key in ("pair_ovr", "pair_lgbm", "three_way")}
    outer_legacy = np.zeros((n, len(classes)))
    fold_choices, warnings_seen, feature_counts = [], 0, []

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(outer.split(np.zeros(n), truth), 1):
        # ── 1) outer-train 내부 OOF (여기서만 가중치를 고른다)
        inner_proba = {name: np.zeros((len(tr), len(classes)))
                       for name in ("champion", "ovr", "lgbm")}
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_truth = truth[tr]
        for inner_fit, inner_holdout in inner.split(np.zeros(len(tr)), inner_truth):
            proba, warned, _ = fit_fold(sdh, final, context, labels, case, seed, contrast,
                                        tr[inner_fit], tr[inner_holdout], drop_exact)
            warnings_seen += warned
            for name in inner_proba:
                inner_proba[name][inner_holdout] = proba[name]
        choices = select_weights(inner_proba, classes, inner_truth)

        # ── 2) 고른 가중치를 이 outer validation 에만 적용
        proba, warned, feature_count = fit_fold(
            sdh, final, context, labels, case, seed, contrast, tr, va, drop_exact)
        warnings_seen += warned
        feature_counts.append(feature_count)
        for name in outer_solo:
            outer_solo[name][va] = proba[name]
        for partner in PARTNERS:
            weight = choices[f"pair_{partner}"]["weight"]
            outer_foldlocal[f"pair_{partner}"][va] = (
                (1 - weight) * proba[ANCHOR] + weight * proba[partner])
        three = choices["three_way"]["weights"]
        outer_foldlocal["three_way"][va] = sum(three[name] * proba[name] for name in three)
        outer_legacy[va] = sum(LEGACY_WEIGHTS[name] * proba[name] for name in LEGACY_WEIGHTS)

        fold_choices.append({"fold": fold,
                             "pair_ovr_weight": choices["pair_ovr"]["weight"],
                             "pair_lgbm_weight": choices["pair_lgbm"]["weight"],
                             "three_way_weights": three})
        print(f"      fold {fold}/5  ovr w={choices['pair_ovr']['weight']:.2f} · "
              f"lgbm w={choices['pair_lgbm']['weight']:.2f} · "
              f"three_way {three['champion']:.2f}/{three['ovr']:.2f}/{three['lgbm']:.2f}",
              flush=True)

    def scored(proba):
        predicted = classes[proba.argmax(axis=1)]
        return {"macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(truth, predicted))}

    anchor_score = scored(outer_solo[ANCHOR])
    result = {
        "seed": seed,
        "contrast": contrast,
        "anchor": anchor_score,
        "solo": {name: scored(outer_solo[name]) for name in outer_solo},
        "foldlocal": {key: scored(outer_foldlocal[key]) for key in outer_foldlocal},
        "legacy_fixed_weights": scored(outer_legacy),
        "fold_choices": fold_choices,
        "feature_count_mean": float(np.mean(feature_counts)),
        "convergence_warning_count": warnings_seen,
        "runtime_minutes": (perf_counter() - started) / 60,
    }
    for key, value in result["foldlocal"].items():
        value["gain_vs_anchor"] = value["macro_f1"] - anchor_score["macro_f1"]
    result["legacy_fixed_weights"]["gain_vs_anchor"] = (
        result["legacy_fixed_weights"]["macro_f1"] - anchor_score["macro_f1"])
    print(f"    → seed {seed}: anchor {anchor_score['macro_f1']:.6f} · "
          f"fold-local three_way {result['foldlocal']['three_way']['macro_f1']:.6f} "
          f"({result['runtime_minutes']:.1f}분)\n", flush=True)
    return result


def aggregate(rows: list[dict], path: list[str]) -> dict:
    values = []
    for row in rows:
        node = row
        for key in path:
            node = node[key]
        values.append(node["macro_f1"])
    gains = []
    for row in rows:
        node = row
        for key in path:
            node = node[key]
        gains.append(node.get("gain_vs_anchor", 0.0))
    return {"macro_f1_mean": float(np.mean(values)),
            "macro_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "gain_mean": float(np.mean(gains)),
            "gain_std": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
            "positive_seeds": f"{sum(1 for g in gains if g > 0)}/{len(gains)}",
            "per_seed": {row["seed"]: value for row, value in zip(rows, values)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast", choices=("fixed", "auto"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--drop-exact", action="store_true",
                        help="고정 exact mutation 4개를 제거한다(2026-08-04 팀 공지)")
    parser.add_argument("--inner-splits", type=int, default=3,
                        help="outer-train 내부 OOF 의 fold 수. 계약은 k 를 지정하지 않는다")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    out = args.out or (Path(__file__).parent / "artifacts"
                       / f"foldlocal_blend_{args.contrast}{'_noexact' if args.drop_exact else ''}.json")
    sdh = load_module(root / "experiments" / "SDH" / "exp_012_enrichment_stability"
                      / "preprocessing.py", "sdh_exp012_preprocessing")
    final = load_module(root / "final_pipeline" / "final_submission.py",
                        "final_submission_module")
    case = sdh.make_cases()[CASE]

    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    # test 는 읽지 않는다.
    context = sdh.make_context(train[genes], genes, show_progress=True)

    rows = []
    for seed in args.seeds:
        print(f"\n[seed {seed}] contrast={args.contrast} · "
              f"inner {args.inner_splits}-fold 에서만 가중치 선택", flush=True)
        rows.append(run_seed(sdh, final, context, labels, case, seed,
                             args.contrast, args.inner_splits, args.drop_exact))

    report = {
        "case": CASE, "contrast": args.contrast, "seeds": args.seeds,
        "drop_fixed_exact": args.drop_exact,
        "inner_splits": args.inner_splits,
        "contract_grid": list(CONTRACT_GRID),
        "three_way_grid": [dict(w) for w in THREE_WAY_GRID],
        "legacy_weights": LEGACY_WEIGHTS,
        "runs": rows,
        "summary": {
            "anchor": aggregate(rows, ["anchor"]),
            "pair_ovr": aggregate(rows, ["foldlocal", "pair_ovr"]),
            "pair_lgbm": aggregate(rows, ["foldlocal", "pair_lgbm"]),
            "three_way": aggregate(rows, ["foldlocal", "three_way"]),
            "legacy_fixed_weights": aggregate(rows, ["legacy_fixed_weights"]),
        },
        "note": ("fold-local 선택분은 계약 준수 추정치다. legacy_fixed_weights 는 "
                 "계약 이전 방식(전체 OOF 에서 고른 0.55/0.30/0.15)을 같은 split 에서 "
                 "다시 잰 값으로, 두 값의 차이가 선택 편향의 실측 크기다."),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    width = 78
    print("\n" + "=" * width)
    print(f"{'구성':28s}{'Macro F1 평균':>16s}{'표준편차':>12s}{'anchor 대비':>13s}{'양수':>8s}")
    print("-" * width)
    labels_map = [("anchor (LR 단독)", "anchor"),
                  ("fold-local LR+ovr", "pair_ovr"),
                  ("fold-local LR+lgbm", "pair_lgbm"),
                  ("fold-local three_way", "three_way"),
                  ("계약 이전 고정 0.55/.30/.15", "legacy_fixed_weights")]
    for name, key in labels_map:
        s = report["summary"][key]
        gain = "" if key == "anchor" else f"{s['gain_mean']:>+13.6f}"
        positive = "" if key == "anchor" else f"{s['positive_seeds']:>8s}"
        print(f"{name:28s}{s['macro_f1_mean']:>16.6f}{s['macro_f1_std']:>12.6f}"
              f"{gain:>13s}{positive:>8s}")
    print("=" * width)
    bias = (report["summary"]["legacy_fixed_weights"]["gain_mean"]
            - report["summary"]["three_way"]["gain_mean"])
    print(f"선택 편향 실측: 전체 OOF 방식이 fold-local 대비 {bias:+.6f} 만큼 더 높게 나온다.")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
