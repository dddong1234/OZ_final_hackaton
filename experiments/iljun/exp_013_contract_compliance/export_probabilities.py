"""③ OOF/test 확률을 팀 계약 포맷으로 내보낸다.

    .venv/bin/python experiments/iljun/exp_013_contract_compliance/export_probabilities.py \\
        --contrast auto --seeds 42 52 62

왜 우리가 하나
  Notion 계약의 역할 분담에서 우리 몫이 이렇게 적혀 있다.

    권일준 | 공용 champion multinomial Logistic Regression |
    현재 팀의 가장 강한 선형 기준선·OOF 기준점을 유지하고 **공통 split/OOF 확률을 관리한다**

  그런데 우리 OOF 는 `results/our_three_way_oof_seed42.npy` 처럼 ID·fold 열이 없는
  배열로만 있었다.  `MODEL_DIVERSITY_STRATEGY.md` §2.3 포맷으로 맞춘다.

    OOF  : ID, SUBCLASS, seed, fold, prob_<class×26>
    test : ID, seed, prob_<class×26>

어떻게 만드나
  OOF 확률은 ① `run_contrast_comparison.py` 가 이미 저장한 npz 를 그대로 쓴다.
  fold 번호는 `StratifiedKFold(5, shuffle=True, random_state=seed)` 가 결정적이라
  같은 label 로 다시 나누면 정확히 복원된다 — 모델을 다시 학습하지 않는다.
  test 확률만 전체 train 학습 후 새로 만든다.

  test 경로는 `make_submission_context` 를 쓴다.  train 이 어휘를 정의하고 test 는
  거기에 투영된다 — 계약의 '원본을 결합하지 않는다' 조항.

확률 파일은 커밋하지 않는다
  §2.3 이 정한 대로 각 실험의 `results/` 아래(gitignore)에만 둔다.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

CASE = "case_04_shrink10"
EXPERIMENT_ID = "exp013-contract-anchor-lr"
OWNER = "권일준"


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


def fold_assignment(labels: pd.Series, seed: int) -> np.ndarray:
    """① 이 쓴 것과 동일한 분할을 재현해 행별 fold 번호를 만든다."""

    truth = labels.to_numpy()
    folds = np.zeros(len(truth), dtype=np.int16)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(truth)), truth), 1):
        folds[valid_index] = fold
    assert folds.min() == 1, "fold 번호가 배정되지 않은 행이 있습니다"
    return folds


def probability_frame(ids, proba: np.ndarray, classes, seed: int,
                      subclass=None, folds=None) -> pd.DataFrame:
    frame = pd.DataFrame({"ID": ids})
    if subclass is not None:
        frame["SUBCLASS"] = np.asarray(subclass)
    frame["seed"] = seed
    if folds is not None:
        frame["fold"] = folds
    for index, name in enumerate(classes):
        frame[f"prob_{name}"] = proba[:, index]
    # 각 행의 확률은 유한하고 합이 1 이어야 한다 — 계약 §7.2
    values = frame[[f"prob_{name}" for name in classes]].to_numpy()
    assert np.isfinite(values).all(), "확률에 비유한값이 있습니다"
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-6)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast", choices=("fixed", "auto"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--drop-exact", action="store_true",
                        help="고정 exact mutation 4개 제거 구성(2026-08-04 팀 공지)")
    parser.add_argument("--models", nargs="+", default=["champion"],
                        help="내보낼 모델. 기본은 계약상 우리 담당인 anchor LR 하나")
    parser.add_argument("--oof-dir", type=Path, default=None,
                        help="① 이 저장한 npz 위치")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--skip-test", action="store_true",
                        help="test 확률 생성을 건너뛴다(OOF 만 내보낼 때)")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    oof_dir = args.oof_dir or (root / "experiments" / "iljun" / "results" / "contract_oof")
    suffix = "_noexact" if args.drop_exact else ""
    outdir = args.outdir or (root / "experiments" / "iljun" / "results" / "contract_probabilities")
    outdir.mkdir(parents=True, exist_ok=True)

    sdh = load_module(root / "experiments" / "SDH" / "exp_012_enrichment_stability"
                      / "preprocessing.py", "sdh_exp012_preprocessing")
    final = load_module(root / "final_pipeline" / "final_submission.py",
                        "final_submission_module")
    case = sdh.make_cases()[CASE]

    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "owner": OWNER,
        "case": CASE,
        "contrast": args.contrast,
        "drop_fixed_exact": args.drop_exact,
        "models": args.models,
        "model_parameters": {
            "champion": {"estimator": "LogisticRegression", "solver": "lbfgs", "C": 0.07,
                         "max_iter": 2000, "class_weight": "balanced"},
            "ovr": {"estimator": "OneVsRestClassifier(LogisticRegression)", "solver": "lbfgs",
                    "C": 0.07, "max_iter": 2000, "class_weight": "balanced"},
            "lgbm": {"estimator": "LGBMClassifier", "objective": "multiclass",
                     "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31,
                     "class_weight": "balanced", "deterministic": True},
        },
        "fold_definition": "StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)",
        "seeds": args.seeds,
        "leakage_audit": {
            "permutation_label": "experiments/iljun/exp_012_cross_member_ensemble/"
                                 "artifacts/permutation_ensemble.json",
            "vocabulary": "train-only (make_submission_context)",
            "raw_train_test_concat_used": False,
        },
        "files": {"oof": {}, "test": {}},
    }

    # ── OOF: ① 산출물 재사용
    print("[1] OOF 확률 — ① 산출물에서 계약 포맷으로 변환", flush=True)
    for seed in args.seeds:
        source = oof_dir / f"oof_{args.contrast}{suffix}_seed{seed}.npz"
        if not source.exists():
            print(f"    seed {seed} 건너뜀 — {source.name} 이 없습니다 "
                  f"(① run_contrast_comparison.py 를 먼저 돌리세요)", flush=True)
            continue
        payload = np.load(source, allow_pickle=True)
        classes = payload["classes"]
        folds = fold_assignment(labels, seed)
        for model in args.models:
            frame = probability_frame(train["ID"], payload[model], classes, seed,
                                      subclass=labels.to_numpy(), folds=folds)
            path = outdir / f"oof_{model}_{args.contrast}{suffix}_seed{seed}.csv"
            frame.to_csv(path, index=False)
            manifest["files"]["oof"].setdefault(model, []).append(path.name)
            print(f"    seed {seed} · {model:9s} → {path.name} ({len(frame):,}행)", flush=True)
    manifest["class_order"] = [str(name) for name in classes]

    # ── test: 전체 train 학습 후 새로 만든다
    if not args.skip_test:
        print("\n[2] test 확률 — 전체 train 학습 (어휘는 train 전용)", flush=True)
        test = pd.read_csv(root / "data" / "raw" / "test.csv")
        assert list(test.columns) == ["ID", *genes], "test 컬럼 구조가 train 과 다릅니다"
        context, _, audit = sdh.make_submission_context(
            train[genes], test[genes], genes, show_progress=True)
        assert audit["raw_train_test_concat_used"] is False
        manifest["vocabulary_audit"] = audit
        train_index = np.arange(len(train))
        test_index = np.arange(len(train), len(train) + len(test))

        baseline = sdh.B04_CANDIDATE
        for seed in args.seeds:
            started = perf_counter()
            pairs = (final.discover_confusion_pairs(context.cache, train_index, labels, seed)
                     if args.contrast == "auto" else baseline.contrast_pairs)
            changes = {"contrast_pairs": pairs}
            if args.drop_exact:
                changes["exact_events"] = ()
            sdh.B04_CANDIDATE = dataclasses.replace(baseline, **changes)
            try:
                train_matrix, test_matrix, _, meta = sdh.build_case_matrices(
                    context, train_index, test_index, labels, case, inner_seed=seed)
            finally:
                sdh.B04_CANDIDATE = baseline

            models = make_models(seed)
            for model_name in args.models:
                model = models[model_name]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train_matrix, labels)
                warned = sum(issubclass(i.category, ConvergenceWarning) for i in caught)
                frame = probability_frame(test["ID"], model.predict_proba(test_matrix),
                                          model.classes_, seed)
                path = outdir / f"test_{model_name}_{args.contrast}{suffix}_seed{seed}.csv"
                frame.to_csv(path, index=False)
                manifest["files"]["test"].setdefault(model_name, []).append(path.name)
                manifest.setdefault("convergence_warning_count", {})[
                    f"{model_name}_seed{seed}"] = warned
                print(f"    seed {seed} · {model_name:9s} → {path.name} "
                      f"(피처 {meta['total_feature_count']:,} · 경고 {warned})", flush=True)
            manifest.setdefault("feature_count", {})[f"seed{seed}"] = meta["total_feature_count"]
            manifest.setdefault("runtime_minutes", {})[f"seed{seed}"] = (
                perf_counter() - started) / 60

    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n계약 포맷 확률 파일: {outdir}")
    print(f"메타데이터: {outdir / 'manifest.json'}")
    print("확률 파일은 results/ 아래(gitignore)라 커밋되지 않습니다 — §2.3 조항.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
