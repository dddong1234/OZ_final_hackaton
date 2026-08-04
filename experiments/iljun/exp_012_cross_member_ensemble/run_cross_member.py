"""팀원 간 앙상블 — 우리 three_way 와 경수님 P1/EB 라인을 섞는다.

    # 1) 경수님 OOF 확률을 로컬로 추출 (gitignore 되는 results/ 로)
    git fetch origin gs/exp_004
    mkdir -p experiments/iljun/results/gs_oof
    git show origin/gs/exp_004:experiments/gs/notebooks/exp_model_004/result/\\
exp-all-class-evidence-ranker-01_seed42_oof_probabilities.csv \\
      > experiments/iljun/results/gs_oof/gs_ranker_seed42.csv

    # 2) 실행
    .venv/bin/python experiments/iljun/exp_012_cross_member_ensemble/run_cross_member.py

왜 하는가
  지금까지의 앙상블은 전부 **한 사람 파이프라인 안에서** 이뤄졌다.
  ENS-011a(LR multinomial + OVR + LGBM)도 같은 피처 위에서 모델만 바꾼 것이다.

  협업 규정 1 의 원래 논리는 이것이었다 — "LGBM/XGB/CatBoost 를 나누어 맡으면
  예측값 상관이 0.99 를 넘어 앙상블 이득이 사라진다. **각자 다른 천장을 뚫어야**
  블렌드에서 시너지가 난다."  그런데 팀원 간 블렌드는 아무도 해보지 않았다.

  경수님 P1/EB 라인은 우리 class-enrichment 와 **접근이 다르다**(Empirical-Bayes
  기반 selective gate). 계열이 다르므로 서로 다른 실수를 할 가능성이 크다.

전제 확인 (스크립트가 검증한다)
  · 경수님 OOF 의 true_class 가 train.csv 순서와 완전히 일치한다 → 행 정렬 가능
  · 클래스 목록이 같다 → 확률 열 정렬 가능
  · fold 분할은 팀 표준(StratifiedKFold-5, shuffle, random_state=seed)이라 동일할
    것으로 보이나, 설령 다르더라도 양쪽 모두 정직한 out-of-fold 예측이므로
    블렌드 추정 자체는 유효하다. 다만 그 경우 이득 추정이 약간 낙관적일 수 있다.

주의
  경수님 파일은 아직 main 에 병합되지 않은 gs/exp_004 브랜치에 있다.  병합 전에는
  위 git show 명령으로 추출해야 재현된다.  추출본은 results/ 아래(gitignore)에 두어
  OOF 확률이 우리 브랜치로 커밋되지 않게 한다.
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
OUR_WEIGHTS = {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15}   # ENS-011a 확정값
WEIGHT_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)                        # 경수님 라인 쪽 가중치


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


def our_oof(sdh, context, labels: pd.Series, seed: int, classes):
    """ENS-011a three_way 의 OOF 확률."""
    case = sdh.make_cases()[CASE]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = np.zeros((len(labels), len(classes)))
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels.to_numpy()), 1):
        train_matrix, valid_matrix, _, _ = sdh.build_case_matrices(
            context, tr, va, labels, case, inner_seed=seed)
        parts = {}
        for name, model in make_models(seed).items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(train_matrix, labels.iloc[tr])
            parts[name] = model.predict_proba(valid_matrix)
            assert list(model.classes_) == list(classes)
        proba[va] = sum(OUR_WEIGHTS[n] * parts[n] for n in OUR_WEIGHTS)
        print(f"      fold {fold}/5", flush=True)
    return proba


def load_partner(path: Path, prefix: str, classes, truth) -> np.ndarray:
    """경수님 OOF CSV 에서 한 변형의 확률 행렬을 뽑아 우리 클래스 순서로 맞춘다."""
    frame = pd.read_csv(path)
    assert len(frame) == len(truth), f"행 수 불일치: {len(frame)} vs {len(truth)}"
    assert (frame["true_class"].to_numpy() == truth).all(), \
        "true_class 가 train 순서와 다릅니다 — 행 정렬을 확인하세요"
    columns = [f"{prefix}{name}" for name in classes]
    missing = [c for c in columns if c not in frame.columns]
    assert not missing, f"없는 열: {missing[:5]}"
    matrix = frame[columns].to_numpy(dtype=np.float64)
    # 확률로 정규화 — 변형에 따라 합이 1 이 아닐 수 있다
    row_sum = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(row_sum > 0, row_sum, 1.0)


def score(proba, classes, truth) -> dict:
    predicted = classes[proba.argmax(axis=1)]
    return {"macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, predicted))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partner", type=Path, default=None,
                        help="경수님 OOF CSV. 기본 results/gs_oof/gs_ranker_seed42.csv")
    # 경수님 파일의 열 이름은 이중 언더스코어를 쓴다 — 예: gate__ACC, ranker__KIPAN.
    parser.add_argument("--prefixes", nargs="+", default=["gate__", "p1_eb__", "ranker__"],
                        help="비교할 변형 prefix")
    parser.add_argument("--cache", type=Path, default=None,
                        help="우리 OOF 확률 캐시(.npy). 있으면 재계산하지 않는다")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "cross_member.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    partner_path = args.partner or root / "experiments" / "iljun" / "results" / "gs_oof" / "gs_ranker_seed42.csv"
    if not partner_path.exists():
        print(f"경수님 OOF 파일이 없습니다: {partner_path}\n"
              f"docstring 상단의 git show 명령으로 먼저 추출하세요.")
        return 2

    sdh = load_sdh(root)
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    truth = labels.to_numpy()
    classes = np.unique(truth)

    # OOF 확률은 계산에 6분 남짓 걸리고 seed 가 같으면 결정적이므로 캐시한다.
    # 캐시는 results/ 아래(gitignore)에 둔다 — OOF 확률은 커밋 금지 대상이다.
    cache_path = args.cache or (root / "experiments" / "iljun" / "results"
                                / f"our_three_way_oof_seed{args.seed}.npy")
    print(f"\n[1] 우리 three_way OOF (seed {args.seed})", flush=True)
    if cache_path.exists():
        ours = np.load(cache_path)
        assert ours.shape == (len(truth), len(classes)), "캐시 형상이 맞지 않습니다"
        print(f"    캐시 사용: {cache_path.name}", flush=True)
    else:
        started = perf_counter()
        context = sdh.make_context(train[genes], genes, show_progress=True)
        ours = our_oof(sdh, context, labels, args.seed, classes)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, ours)
        print(f"    ({(perf_counter()-started)/60:.1f}분) 캐시 저장", flush=True)
    print(f"    Macro F1 {score(ours, classes, truth)['macro_f1']:.5f}", flush=True)

    report = {"seed": args.seed, "partner_file": str(partner_path),
              "our_weights": OUR_WEIGHTS,
              "ours": score(ours, classes, truth), "partners": {}}
    our_predicted = classes[ours.argmax(axis=1)]
    our_correct = our_predicted == truth

    print(f"\n[2] 경수님 변형별 다양성·블렌드", flush=True)
    for prefix in args.prefixes:
        try:
            partner = load_partner(partner_path, prefix, classes, truth)
        except AssertionError as error:
            print(f"    {prefix:12s} 건너뜀 — {error}", flush=True)
            continue
        partner_predicted = classes[partner.argmax(axis=1)]
        partner_correct = partner_predicted == truth

        blends = []
        for weight in WEIGHT_GRID:
            mixed = (1 - weight) * ours + weight * partner
            blends.append({"weight_partner": weight, **score(mixed, classes, truth)})
        best = max(blends, key=lambda row: row["macro_f1"])

        entry = {
            "solo": score(partner, classes, truth),
            "disagreement_rate": float((our_predicted != partner_predicted).mean()),
            "oracle_either_correct": float((our_correct | partner_correct).mean()),
            "rescue_rate": float((~our_correct & partner_correct).sum() / max((~our_correct).sum(), 1)),
            "blend_sweep": blends,
            "best_blend": best,
            "best_gain": best["macro_f1"] - report["ours"]["macro_f1"],
            "positive_weights": sum(1 for b in blends if b["macro_f1"] > report["ours"]["macro_f1"]),
        }
        report["partners"][prefix] = entry
        print(f"    {prefix:12s} 단독 {entry['solo']['macro_f1']:.5f} · "
              f"불일치 {entry['disagreement_rate']:.1%} · 구제율 {entry['rescue_rate']:.1%} · "
              f"최적 blend {best['macro_f1']:.5f} ({entry['best_gain']:+.5f}, w={best['weight_partner']})",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"우리 three_way 단독  {report['ours']['macro_f1']:.5f}")
    print("-" * 76)
    print(f"{'경수님 변형':14s}{'단독':>10s}{'불일치':>9s}{'오라클':>9s}{'구제율':>9s}"
          f"{'최적 blend':>12s}{'이득':>10s}{'양수':>6s}")
    print("-" * 76)
    for prefix, entry in report["partners"].items():
        print(f"{prefix:14s}{entry['solo']['macro_f1']:>10.5f}"
              f"{entry['disagreement_rate']:>9.1%}{entry['oracle_either_correct']:>9.1%}"
              f"{entry['rescue_rate']:>9.1%}{entry['best_blend']['macro_f1']:>12.5f}"
              f"{entry['best_gain']:>+10.5f}{entry['positive_weights']:>4d}/5")
    print("=" * 76)
    print("가중치는 같은 OOF 에서 고른 것이라 낙관 편향이 있다. 채택하려면 w 고정 후 3-seed 재측정.")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
