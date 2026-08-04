"""ENS-011a three_way 제출 파일 생성.

    .venv/bin/python experiments/iljun/exp_011_model_ensemble/make_submission.py

무엇을 만드나
  exp_012 챔피언 피처(case_04_shrink10) 위에 세 모델을 소프트보팅한 제출본.
    LR multinomial 0.55 + OVR LR 0.30 + LGBM(100 trees) 0.15
  가중치는 run_ensemble_confirm.py 에서 3-seed 확정한 값 그대로다(변경 없음).
  3-seed OOF 0.54202, 챔피언 대비 깨끗한 seed +0.01608 (6.7σ, 3/3).

왜 이것을 먼저 올리나
  exp_012(shrinkage 10) 단독도 아직 LB 미확인이라 두 변화가 섞이는 문제가 있다.
  그럼에도 이쪽이 1순위인 이유는 **최종 제출을 무엇으로 할지 바꾸는 결정이
  앙상블 쪽**이기 때문이다. shrinkage 는 CV +0.004 짜리 하이퍼파라미터라
  전달 여부와 무관하게 최종 구성이 바뀌지 않지만, 앙상블(+0.016)은 전달되면
  그것이 최종 제출이 된다. 분리는 exp_012 단독을 다음 슬롯에 올려서 한다.

경로에 대하여
  SDH exp_012 `preprocessing.py` 의 make_context / build_case_matrices 를 그대로
  쓴다.  **LB 확인 실험이므로 검증된 코드 경로를 그대로 타는 것이 맞다** — 여기에
  exp_009 하드닝(어휘 분리)을 같이 넣으면 LB 숫자가 어느 변경 때문인지 갈리지
  않는다.  하드닝은 최종 제출 스크립트를 만들 때 별도로 적용한다.

  참고: gs 의 `assert test NaN == 237` 은 gs `main()` 에 있고 이 경로는 그것을
  호출하지 않으므로 여기서는 걸리지 않는다.
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
from sklearn.multiclass import OneVsRestClassifier

CASE = "case_04_shrink10"
WEIGHTS = {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15}   # 3-seed 확정값. 고정.


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    outdir = args.outdir or root / "experiments" / "iljun" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    sdh = load_sdh(root)

    data_dir = root / "data" / "raw"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    assert list(test.columns) == ["ID", *genes], "test 컬럼 구조가 train 과 다릅니다"
    labels = train["SUBCLASS"]
    print(f"train {train.shape} · test {test.shape} · 유전자 {len(genes):,}", flush=True)

    combined = pd.concat([train[genes], test[genes]], axis=0, ignore_index=True)
    context = sdh.make_context(combined, genes, show_progress=True)
    train_index = np.arange(len(train))
    test_index = np.arange(len(train), len(combined))

    started = perf_counter()
    train_matrix, test_matrix, names, meta = sdh.build_case_matrices(
        context, train_index, test_index, labels, sdh.make_cases()[CASE],
        inner_seed=args.seed)
    print(f"피처 {meta['total_feature_count']:,} "
          f"(base {meta['base_feature_count']:,} + enrich {meta['extra_feature_count']}) "
          f"· {(perf_counter()-started)/60:.1f}분", flush=True)

    proba, warnings_seen = {}, 0
    for name, model in make_models(args.seed).items():
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, labels)
        warnings_seen += sum(issubclass(i.category, ConvergenceWarning) for i in caught)
        proba[name] = model.predict_proba(test_matrix)
        classes = model.classes_
        print(f"  {name:10s} 학습 {(perf_counter()-started):5.0f}s", flush=True)

    mixed = sum(WEIGHTS[name] * proba[name] for name in WEIGHTS)
    prediction = classes[mixed.argmax(axis=1)]

    submission = pd.DataFrame({"ID": test["ID"], "SUBCLASS": prediction})
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        assert list(sample.columns) == ["ID", "SUBCLASS"]
        assert list(sample["ID"]) == list(test["ID"]), "sample 과 test 의 ID 순서가 다릅니다"
        sample["SUBCLASS"] = submission["SUBCLASS"]
        submission = sample

    # 제출 형식 검증
    assert len(submission) == len(test)
    assert list(submission.columns) == ["ID", "SUBCLASS"]
    assert submission["ID"].equals(test["ID"])
    assert int(submission.isna().sum().sum()) == 0
    assert submission["ID"].duplicated().sum() == 0

    path = outdir / f"submission_ENS-011a_three_way_seed{args.seed}.csv"
    submission.to_csv(path, index=False)

    # 챔피언 단독도 같이 저장한다 — 확률이 이미 있어 비용 0 이고,
    # exp_012 단독 분리 제출(2순위)에 그대로 쓸 수 있다.
    champion_only = submission.copy()
    champion_only["SUBCLASS"] = classes[proba["champion"].argmax(axis=1)]
    champion_path = outdir / f"submission_exp012_champion_only_seed{args.seed}.csv"
    champion_only.to_csv(champion_path, index=False)

    changed = int((submission["SUBCLASS"] != champion_only["SUBCLASS"]).sum())
    metadata = {
        "experiment_id": "ENS-011a-3way-champion",
        "case": CASE, "weights": WEIGHTS, "seed": args.seed,
        "feature_count": meta["total_feature_count"],
        "convergence_warning_count": warnings_seen,
        "cv_3seed_macro_f1": 0.54202,
        "cv_delta_vs_champion_clean_seeds": 0.01608,
        "rows": len(submission),
        "distinct_classes": int(submission["SUBCLASS"].nunique()),
        "rows_changed_vs_champion_only": changed,
    }
    (outdir / f"{path.stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"1순위 제출본  {path.name}")
    print(f"  행 {len(submission):,} · 클래스 {submission['SUBCLASS'].nunique()} "
          f"· 결측 0 · ID 순서 일치 · 수렴 경고 {warnings_seen}")
    print(f"  챔피언 단독 대비 예측이 바뀐 행 {changed:,} ({changed/len(submission):.1%})")
    print(f"\n2순위(분리용)  {champion_path.name}")
    print("=" * 62)
    print(f"\n최다 예측: {submission['SUBCLASS'].value_counts().head(3).to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
