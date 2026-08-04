"""three_way 를 seed 평균한 구성 — CV 측정과 제출본 생성을 한 번에.

    .venv/bin/python experiments/iljun/exp_011_model_ensemble/run_seed_average.py

배경
  ENS-011a(three_way, seed 42 단일)가 LB 0.45349 를 냈고 전달률이 ~101% 였다
  (CV +0.01807 → LB +0.01824, 간격 0.08870 → 0.08853 로 거의 불변).
  전달률이 100% 로 확인된 이상 남은 CV 이득은 그대로 LB 이득으로 읽을 수 있다.

  seed 평균은 새 모델 계열을 늘리지 않고 분산만 줄이는 축이다.
  SDH exp_011 권장 후속 4번("seed 42 단일 제출과 3개 inner-cross-fit seed 확률
  앙상블 비교")인데 아직 아무도 돌리지 않았다.

이 파이프라인에서 seed 가 실제로 바꾸는 것
  - CV 경로: outer fold 분할이 바뀐다 → 서로 다른 80% 부분집합으로 학습 → 배깅 효과 큼
  - 제출 경로: train 100% 로 학습하므로 outer 분할이 없다. 바뀌는 것은
      enrichment inner cross-fit 분할(→ 학습용 enrichment 피처)과 표준화 통계뿐.
      test 쪽 enrichment weight 는 train 전체로 학습하므로 seed 무관.
      lbfgs 는 결정적이고 LGBM 도 subsample/colsample 이 1.0 이라 random_state 영향이 거의 없다.
  → **제출 경로의 변동이 CV 보다 작다.** 그래서 CV 이득을 상한으로 읽어야 하며,
     이 스크립트는 제출본까지 만들어 실제 이득을 LB 로 확인할 수 있게 한다.
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

CASE = "case_04_shrink10"
WEIGHTS = {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15}   # ENS-011a 확정값. 고정.


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


def fit_blend(train_matrix, train_labels, target_matrix, seed):
    """세 모델을 학습해 target 확률을 blend 한다."""
    proba, warnings_seen, classes = {}, 0, None
    for name, model in make_models(seed).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, train_labels)
        warnings_seen += sum(issubclass(i.category, ConvergenceWarning) for i in caught)
        proba[name] = model.predict_proba(target_matrix)
        classes = model.classes_
    mixed = sum(WEIGHTS[name] * proba[name] for name in WEIGHTS)
    return mixed, classes, warnings_seen


def oof_blend(sdh, context, labels: pd.Series, seed: int, n_classes: int):
    case = sdh.make_cases()[CASE]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = np.zeros((len(labels), n_classes))
    classes, warnings_seen = None, 0
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels.to_numpy()), 1):
        train_matrix, valid_matrix, _, _ = sdh.build_case_matrices(
            context, tr, va, labels, case, inner_seed=seed)
        mixed, classes, warned = fit_blend(train_matrix, labels.iloc[tr], valid_matrix, seed)
        proba[va] = mixed
        warnings_seen += warned
        print(f"      fold {fold}/5", flush=True)
    return proba, classes, warnings_seen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 777])
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "seed_average.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    sdh = load_sdh(root)
    data_dir = root / "data" / "raw"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    truth = labels.to_numpy()
    n_classes = int(labels.nunique())

    # ── 1. CV — seed 평균이 OOF 에서 얼마나 이득인가
    print("\n[1] CV — three_way 를 seed 별로 낸 뒤 확률 평균", flush=True)
    cv_context = sdh.make_context(train[genes], genes, show_progress=True)
    oof, classes = [], None
    per_seed = {}
    for seed in args.seeds:
        print(f"  seed {seed}", flush=True)
        started = perf_counter()
        proba, classes, _ = oof_blend(sdh, cv_context, labels, seed, n_classes)
        oof.append(proba)
        predicted = classes[proba.argmax(axis=1)]
        per_seed[str(seed)] = {
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, predicted)),
        }
        print(f"    three_way {per_seed[str(seed)]['macro_f1']:.5f} "
              f"({(perf_counter()-started)/60:.1f}분)", flush=True)

    averaged = np.mean(oof, axis=0)
    predicted = classes[averaged.argmax(axis=1)]
    single_mean = float(np.mean([per_seed[s]["macro_f1"] for s in per_seed]))
    seed_avg = {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
    }
    seed_avg["delta_vs_single_seed_mean"] = seed_avg["macro_f1"] - single_mean

    report = {
        "case": CASE, "weights": WEIGHTS, "seeds": args.seeds,
        "single_seed": per_seed,
        "single_seed_macro_f1_mean": single_mean,
        "single_seed_macro_f1_std": float(np.std(
            [per_seed[s]["macro_f1"] for s in per_seed], ddof=1)),
        "seed_averaged": seed_avg,
        "reference": {"ENS-011a_seed42_cv": 0.54202, "ENS-011a_seed42_lb": 0.4534879688,
                      "exp_011_cv": 0.52395, "exp_011_lb": 0.43525},
        "note": ("CV 는 seed 가 outer fold 분할을 바꿔 배깅 효과가 크지만, 제출 경로는 "
                 "train 100% 로 학습하므로 enrichment inner 분할만 바뀐다. "
                 "CV 이득을 제출 이득의 상한으로 읽어야 한다."),
    }

    # ── 2. 제출본 — train 100% × seed 3개, test 확률 평균
    if not args.skip_submission:
        print("\n[2] 제출본 — train 100% 로 seed 별 학습 후 test 확률 평균", flush=True)
        # 어휘는 train 만 정의한다 — 원본 프레임을 결합하지 않는다.
        sub_context, _, audit = sdh.make_submission_context(
            train[genes], test[genes], genes, show_progress=True)
        assert audit["raw_train_test_concat_used"] is False
        assert audit["vocabulary_source"] == "train"
        report["vocabulary_audit"] = audit
        train_index = np.arange(len(train))
        test_index = np.arange(len(train), len(train) + len(test))
        case = sdh.make_cases()[CASE]

        test_proba, total_warnings = [], 0
        for seed in args.seeds:
            started = perf_counter()
            train_matrix, target_matrix, _, meta = sdh.build_case_matrices(
                sub_context, train_index, test_index, labels, case, inner_seed=seed)
            mixed, classes, warned = fit_blend(train_matrix, labels, target_matrix, seed)
            test_proba.append(mixed)
            total_warnings += warned
            print(f"  seed {seed}  피처 {meta['total_feature_count']:,}  "
                  f"({(perf_counter()-started)/60:.1f}분)", flush=True)

        averaged_test = np.mean(test_proba, axis=0)
        submission = pd.DataFrame({"ID": test["ID"],
                                   "SUBCLASS": classes[averaged_test.argmax(axis=1)]})
        sample_path = data_dir / "sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            assert list(sample["ID"]) == list(test["ID"])
            sample["SUBCLASS"] = submission["SUBCLASS"]
            submission = sample
        assert len(submission) == len(test)
        assert list(submission.columns) == ["ID", "SUBCLASS"]
        assert submission["ID"].equals(test["ID"])
        assert int(submission.isna().sum().sum()) == 0

        outdir = root / "experiments" / "iljun" / "results"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "submission_ENS-011b_three_way_seedavg.csv"
        submission.to_csv(path, index=False)

        previous = outdir / "submission_ENS-011a_three_way_seed42.csv"
        changed = None
        if previous.exists():
            changed = int((pd.read_csv(previous)["SUBCLASS"] != submission["SUBCLASS"]).sum())
        report["submission"] = {
            "path": str(path), "rows": len(submission),
            "distinct_classes": int(submission["SUBCLASS"].nunique()),
            "convergence_warning_count": total_warnings,
            "rows_changed_vs_ENS_011a": changed,
        }
        print(f"\n  제출본 {path.name} · 행 {len(submission):,} · "
              f"클래스 {submission['SUBCLASS'].nunique()} · 수렴경고 {total_warnings}")
        if changed is not None:
            print(f"  ENS-011a(seed42 단일) 대비 예측이 바뀐 행 {changed:,} "
                  f"({changed/len(submission):.1%})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"three_way seed 별 (CV)   {single_mean:.5f} ± {report['single_seed_macro_f1_std']:.5f}")
    for seed in args.seeds:
        print(f"   seed {seed:<5d}          {per_seed[str(seed)]['macro_f1']:.5f}")
    print("-" * 64)
    print(f"three_way seed 평균 (CV) {seed_avg['macro_f1']:.5f}  "
          f"{seed_avg['delta_vs_single_seed_mean']:+.5f}")
    print("=" * 64)
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
