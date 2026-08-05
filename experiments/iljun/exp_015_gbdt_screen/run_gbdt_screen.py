"""XGBoost·CatBoost 스크리닝 — 이미 있는 LGBM 위에 무엇을 더하는가.

    .venv/bin/python experiments/iljun/exp_015_gbdt_screen/run_gbdt_screen.py --seeds 42

질문을 정확히 잡는다
  "XGB/CatBoost 가 LR 을 이기나"가 아니다.  우리 앙상블에는 이미 LGBM 이 0.15 로
  들어가 있다.  실질 질문은 **이미 있는 LGBM 위에 무엇을 더하는가** 이고, 계약 §4 도
  다양성을 기준 모델 대비로 재라고 한다.  그래서 두 축을 모두 잰다.

    · LR 대비  — 계약 §4 지표 (전통적인 방식)
    · LGBM 대비 — 확률 상관·불일치 (이번 판단의 핵심)

기각선 (실행 전에 선언)
  1. LGBM 과 확률 상관 > 0.95 → 기각
  2. `LR+LGBM+X` 가 `LR+LGBM` 대비 +0.001 미만 → 기각
  둘 다 통과해야 3-seed 확인으로 넘어간다.

비용을 아끼는 방법
  LR/OVR/LGBM 의 OOF 는 exp_013 이 계약 포맷으로 이미 저장했다(같은 fold, 같은
  피처 구성).  여기서는 XGB·CatBoost 만 새로 학습한다.

피처 구성
  2026-08-04 공지 준수 구성 — fold-train 자동 발견 contrast + 고정 exact 제거.

예산 맞추기 (중요)
  LGBM·XGB 의 `n_estimators=100` 은 multiclass 에서 **클래스당** 100 트리라 총 2,600
  트리다.  CatBoost MultiClass 의 `iterations` 는 multi-output 트리 개수여서 같은 숫자를
  주면 용량이 훨씬 작다.  실제로 iterations=100·depth 5 로는 OOF 0.345999 에 그쳐
  과소적합이었다.  그래서 CatBoost 만 iterations=500·depth 6 으로 올려 다시 잰다.
  숫자를 맞춘 것이 아니라 용량을 맞춘 것이며, 그래도 완전한 등가는 아니다.
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

CASE = "case_04_shrink10"
CONTRACT_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)
CORRELATION_GATE = 0.95        # LGBM 과 이보다 상관이 높으면 기각
GAIN_GATE = 0.001              # LR+LGBM 대비 이 미만이면 기각


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


CATBOOST_ITERATIONS = 500      # LGBM/XGB 의 클래스당 100 트리와 용량을 맞추기 위해 상향


def make_model(name: str, seed: int):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="multi:softprob", num_class=26, n_estimators=100,
            learning_rate=0.05, max_depth=5, tree_method="hist",
            random_state=seed, n_jobs=-1, verbosity=0)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            loss_function="MultiClass", iterations=CATBOOST_ITERATIONS,
            learning_rate=0.05, depth=6, auto_class_weights="Balanced", random_seed=seed,
            thread_count=-1, verbose=False, allow_writing_files=False)
    raise ValueError(name)


def macro_f1(proba: np.ndarray, classes: np.ndarray, truth: np.ndarray) -> float:
    return float(f1_score(truth, classes[proba.argmax(axis=1)],
                          average="macro", zero_division=0))


def oof_probabilities(sdh, final, context, labels, case, seed, name):
    """준수 구성(자동 contrast + exact 제거)에서 한 모델의 5-fold OOF 확률."""

    started = perf_counter()
    truth = labels.to_numpy()
    classes = np.unique(truth)
    lookup = {label: index for index, label in enumerate(classes)}
    encoded = np.array([lookup[label] for label in truth])
    proba = np.zeros((len(truth), len(classes)))
    baseline = sdh.B04_CANDIDATE
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(truth)), truth), 1):
        pairs = final.discover_confusion_pairs(context.cache, tr, labels, seed)
        sdh.B04_CANDIDATE = dataclasses.replace(
            baseline, contrast_pairs=pairs, exact_events=())
        try:
            train_matrix, valid_matrix, _, meta = sdh.build_case_matrices(
                context, tr, va, labels, case, inner_seed=seed)
        finally:
            sdh.B04_CANDIDATE = baseline

        model = make_model(name, seed)
        fold_started = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "xgb":
                weights = compute_sample_weight("balanced", encoded[tr])
                model.fit(train_matrix, encoded[tr], sample_weight=weights)
                proba[va] = model.predict_proba(valid_matrix)
            else:
                model.fit(train_matrix, truth[tr])
                block = model.predict_proba(valid_matrix)
                order = [list(model.classes_).index(name_) for name_ in classes]
                proba[va] = block[:, order]
        print(f"      fold {fold}/5  피처 {meta['total_feature_count']:,} "
              f"({perf_counter()-fold_started:.0f}s)", flush=True)

    print(f"    → {name}: OOF {macro_f1(proba, classes, truth):.6f} "
          f"({(perf_counter()-started)/60:.1f}분)\n", flush=True)
    return proba, classes


def pick_weight(base, extra, classes, truth, rows) -> float:
    best_weight, best = 0.0, macro_f1(base[rows], classes, truth[rows])
    for weight in CONTRACT_GRID:
        mixed = (1 - weight) * base[rows] + weight * extra[rows]
        score = macro_f1(mixed, classes, truth[rows])
        if score > best:
            best_weight, best = weight, score
    return best_weight


def foldlocal_blend(base, extra, classes, truth, folds):
    """계약 절차 — 각 outer fold 의 train 행에서만 가중치를 고른다."""

    blended = np.zeros_like(base)
    weights = {}
    for fold in np.unique(folds):
        valid = folds == fold
        weight = pick_weight(base, extra, classes, truth, ~valid)
        weights[int(fold)] = weight
        blended[valid] = (1 - weight) * base[valid] + weight * extra[valid]
    return blended, weights


def compare(base, other, classes, truth) -> dict:
    base_predicted = classes[base.argmax(axis=1)]
    other_predicted = classes[other.argmax(axis=1)]
    base_ok = base_predicted == truth
    other_ok = other_predicted == truth
    return {
        "disagreement": float((base_predicted != other_predicted).mean()),
        "recovery_rate": float((~base_ok & other_ok).sum() / max((~base_ok).sum(), 1)),
        "reverse_loss_rate": float((base_ok & ~other_ok).sum() / max(base_ok.sum(), 1)),
        "double_fault": float((~base_ok & ~other_ok).mean()),
        "probability_correlation": float(np.corrcoef(base.ravel(), other.ravel())[0, 1]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--models", nargs="+", default=["xgb", "catboost"])
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "gbdt_screen.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    sdh = load_module(root / "experiments" / "SDH" / "exp_012_enrichment_stability"
                      / "preprocessing.py", "sdh_exp012_preprocessing")
    final = load_module(root / "final_pipeline" / "final_submission.py",
                        "final_submission_module")
    case = sdh.make_cases()[CASE]
    oof_dir = root / "experiments" / "iljun" / "results" / "contract_probabilities"
    proba_dir = root / "experiments" / "iljun" / "results" / "gbdt_screen"
    proba_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    # test 는 읽지 않는다.
    context = sdh.make_context(train[genes], genes, show_progress=True)

    report = {"case": CASE, "config": "자동 contrast + 고정 exact 제거 (2026-08-04 공지 준수)",
              "budget": ("LGBM/XGB 는 클래스당 100 트리(총 2,600). CatBoost MultiClass 는 "
                         f"multi-output 트리라 iterations={CATBOOST_ITERATIONS}·depth 6 으로 "
                         "용량을 맞췄다."),
              "gates": {"correlation_vs_lgbm": CORRELATION_GATE, "gain_vs_lr_lgbm": GAIN_GATE},
              "runs": {}}

    for seed in args.seeds:
        anchor_frame = pd.read_csv(oof_dir / f"oof_champion_auto_noexact_seed{seed}.csv")
        classes = np.array([c[len("prob_"):] for c in anchor_frame.columns
                            if c.startswith("prob_")])
        truth = anchor_frame["SUBCLASS"].to_numpy()
        folds = anchor_frame["fold"].to_numpy()
        columns = [f"prob_{name}" for name in classes]
        lr = anchor_frame[columns].to_numpy(dtype=np.float64)
        lgbm = pd.read_csv(oof_dir / f"oof_lgbm_auto_noexact_seed{seed}.csv")[
            columns].to_numpy(dtype=np.float64)

        lr_lgbm, base_weights = foldlocal_blend(lr, lgbm, classes, truth, folds)
        base_score = macro_f1(lr_lgbm, classes, truth)
        print(f"\n[seed {seed}] LR {macro_f1(lr, classes, truth):.6f} · "
              f"LGBM {macro_f1(lgbm, classes, truth):.6f} · "
              f"LR+LGBM(fold-local) {base_score:.6f}", flush=True)

        for name in args.models:
            tag = f"{name}{CATBOOST_ITERATIONS}" if name == "catboost" else name
            cache = proba_dir / f"oof_{tag}_seed{seed}.npy"
            if cache.exists():
                proba = np.load(cache)
                print(f"    {name}: 캐시 사용", flush=True)
            else:
                print(f"    [{name}] 학습", flush=True)
                proba, _ = oof_probabilities(sdh, final, context, labels, case, seed, name)
                np.save(cache, proba)

            solo = macro_f1(proba, classes, truth)
            lr_x, _ = foldlocal_blend(lr, proba, classes, truth, folds)
            stacked, stack_weights = foldlocal_blend(lr_lgbm, proba, classes, truth, folds)
            stacked_score = macro_f1(stacked, classes, truth)
            versus_lgbm = compare(lgbm, proba, classes, truth)
            entry = {
                "solo_macro_f1": solo,
                "solo_accuracy": float(accuracy_score(truth, classes[proba.argmax(axis=1)])),
                "lr_plus_x": macro_f1(lr_x, classes, truth),
                "lr_plus_x_gain": macro_f1(lr_x, classes, truth) - macro_f1(lr, classes, truth),
                "lr_lgbm_base": base_score,
                "lr_lgbm_plus_x": stacked_score,
                "stack_gain": stacked_score - base_score,
                "stack_weights": stack_weights,
                "vs_lr": compare(lr, proba, classes, truth),
                "vs_lgbm": versus_lgbm,
            }
            entry["verdict"] = (
                "기각 — LGBM 과 상관 과다"
                if versus_lgbm["probability_correlation"] > CORRELATION_GATE else
                "기각 — LR+LGBM 대비 이득 부족"
                if entry["stack_gain"] < GAIN_GATE else "통과 — 3-seed 확인 대상")
            report["runs"].setdefault(name, {})[seed] = entry
            print(f"      단독 {solo:.6f} · LR+X {entry['lr_plus_x']:.6f} "
                  f"({entry['lr_plus_x_gain']:+.6f}) · "
                  f"LR+LGBM+X {stacked_score:.6f} ({entry['stack_gain']:+.6f}) · "
                  f"LGBM 상관 {versus_lgbm['probability_correlation']:.3f} "
                  f"→ {entry['verdict']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
