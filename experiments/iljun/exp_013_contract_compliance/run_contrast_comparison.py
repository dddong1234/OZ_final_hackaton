"""① 고정 contrast pair vs fold-train 자동 발견 — 앙상블 경로 대조.

    .venv/bin/python experiments/iljun/exp_013_contract_compliance/run_contrast_comparison.py

왜 하는가
  팀 Notion 계약(`팀 모델 분업·안전한 앙상블 운영 계약`)의 누수 방지 표에 이런
  행이 있다.

    외부 지식 | 허용되지 않음: 논문·OncoKB·ClinVar·pathway DB 의 유전자/핫스팟/
              암종쌍을 모델 입력, 선택 규칙, 가중치 또는 임계값으로 사용하기

  그런데 챔피언과 앙상블이 전부 아래를 태우고 있다.

    FINAL_CONTRAST_PAIRS = (("KIRC","KIPAN",5), ("LGG","GBMLGG",5))

  KIPAN 은 pan-kidney, GBMLGG 는 GBM+LGG — TCGA 코호트 계층 정의다.  자료를 안 보고
  손으로 적을 수 있는 값이 아니므로 감사자에게는 외부 지식으로 읽힌다.  exp_009 §7 이
  "규칙 3 리스크"로 열어 둔 항목이고, 이제 근거 문서가 생겼다.

무엇을 대조하는가
  fold-train 안에서 3-fold 대리모델(G 블록 LR)을 돌려 실제로 많이 혼동되는 쌍을
  찾는 `discover_confusion_pairs` 로 교체한다.  **fold 마다 다시 발견**하므로
  계약의 fold-only 원칙을 만족한다.  구현은 이미 감사받은
  `final_pipeline/final_submission.py` 것을 그대로 불러 쓴다 — 같은 함수가 이미
  `submission_FINAL_autopairs.csv` 를 만들었으므로 중복 구현하지 않는다.

  seed 는 계약이 정한 42/52/62 를 쓴다.  기존 우리 기록(42/2024/777)과 다르므로
  고정 쪽도 같이 다시 잰다 — 같은 split 에서 비교해야 차이가 contrast 때문이다.

주의
  SDH·gs 폴더는 수정하지 않는다.  contrast 쌍 교체는 우리 프로세스 안에서
  `dataclasses.replace` 로 candidate 를 갈아끼우는 방식으로만 한다(협업 규정 4).
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
WEIGHTS = {"champion": 0.55, "ovr": 0.30, "lgbm": 0.15}   # ENS-011a 확정값 — ② 에서 재선택
CONTRACT_SEEDS = (42, 52, 62)


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


def score(proba: np.ndarray, classes: np.ndarray, truth: np.ndarray) -> dict:
    predicted = classes[proba.argmax(axis=1)]
    return {"macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, predicted))}


def override(baseline, pairs, drop_exact: bool):
    """candidate 에서 고정 도메인 피처를 걷어낸다.

    `exact_events` 는 BRAF V600E · IDH1 R132H · PIK3CA H1047R/E545K — 문헌에서
    가져온 고정 hotspot 이라 2026-08-04 팀 공지가 금지한 항목이다.  이것을 비워도
    `R__` recurrent missense 블록은 fold-train 빈도(>=5)로만 뽑히므로 그대로 남는다.
    """
    changes = {"contrast_pairs": pairs}
    if drop_exact:
        changes["exact_events"] = ()
    return dataclasses.replace(baseline, **changes)


def run_config(sdh, final, context, labels, case, seed: int, contrast: str,
               drop_exact: bool = False) -> dict:
    """5-fold outer OOF. contrast='auto' 면 fold 마다 쌍을 다시 발견한다."""

    started = perf_counter()
    truth = labels.to_numpy()
    classes = np.unique(truth)
    proba = {name: np.zeros((len(truth), len(classes))) for name in WEIGHTS}
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    baseline_candidate = sdh.B04_CANDIDATE
    warnings_seen, feature_counts, discovered = 0, [], []

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(truth)), truth), 1):
        if contrast == "auto":
            pairs = final.discover_confusion_pairs(context.cache, tr, labels, seed)
            discovered.append([f"{left}↔{right}" for left, right, _ in pairs])
        else:
            pairs = baseline_candidate.contrast_pairs
        # candidate 를 갈아끼워 이 fold 의 쌍으로 설계행렬을 만든다.
        sdh.B04_CANDIDATE = override(baseline_candidate, pairs, drop_exact)
        try:
            train_matrix, valid_matrix, _, meta = sdh.build_case_matrices(
                context, tr, va, labels, case, inner_seed=seed)
        finally:
            sdh.B04_CANDIDATE = baseline_candidate
        feature_counts.append(meta["total_feature_count"])

        for name, model in make_models(seed).items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train_matrix, labels.iloc[tr])
            warnings_seen += sum(issubclass(i.category, ConvergenceWarning) for i in caught)
            assert list(model.classes_) == list(classes)
            proba[name][va] = model.predict_proba(valid_matrix)
        print(f"      fold {fold}/5  피처 {meta['total_feature_count']:,}"
              + (f"  쌍 {len(pairs)}개" if contrast == "auto" else ""), flush=True)

    blended = sum(WEIGHTS[name] * proba[name] for name in WEIGHTS)
    result = {
        "contrast": contrast,
        "seed": seed,
        "champion": score(proba["champion"], classes, truth),
        "ovr": score(proba["ovr"], classes, truth),
        "lgbm": score(proba["lgbm"], classes, truth),
        "blend": score(blended, classes, truth),
        "feature_count_mean": float(np.mean(feature_counts)),
        "convergence_warning_count": warnings_seen,
        "runtime_minutes": (perf_counter() - started) / 60,
    }
    if discovered:
        result["discovered_pairs_per_fold"] = discovered
    print(f"    → {contrast:5s} seed {seed}: champion {result['champion']['macro_f1']:.6f} "
          f"· blend {result['blend']['macro_f1']:.6f} "
          f"({result['runtime_minutes']:.1f}분)\n", flush=True)
    return result, proba, classes


def summarize(rows: list[dict], key: str) -> dict:
    values = [row[key]["macro_f1"] for row in rows]
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "per_seed": {row["seed"]: row[key]["macro_f1"] for row in rows}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONTRACT_SEEDS))
    parser.add_argument("--configs", nargs="+", default=["fixed", "auto"],
                        choices=["fixed", "auto"])
    parser.add_argument("--drop-exact", action="store_true",
                        help="고정 exact mutation 4개를 제거한다(2026-08-04 팀 공지)")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "contrast_comparison.json")
    parser.add_argument("--proba-dir", type=Path, default=None,
                        help="OOF 확률 저장 위치(gitignore 되는 results/ 아래). ③ 에서 쓴다")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    sdh = load_module(root / "experiments" / "SDH" / "exp_012_enrichment_stability"
                      / "preprocessing.py", "sdh_exp012_preprocessing")
    final = load_module(root / "final_pipeline" / "final_submission.py",
                        "final_submission_module")
    case = sdh.make_cases()[CASE]

    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    # test 는 읽지 않는다 — 계약의 '개발 단계에 test.csv 를 열지 않는다' 조항.
    context = sdh.make_context(train[genes], genes, show_progress=True)

    proba_dir = args.proba_dir or (root / "experiments" / "iljun" / "results" / "contract_oof")
    proba_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_noexact" if args.drop_exact else ""
    report = {"case": CASE, "weights": WEIGHTS, "seeds": args.seeds,
              "fixed_pairs": [list(p) for p in sdh.B04_CANDIDATE.contrast_pairs],
              "fixed_exact_events": [list(e) for e in sdh.B04_CANDIDATE.exact_events],
              "drop_fixed_exact": args.drop_exact,
              "auto_pair_count": final.CONFUSION_PAIR_COUNT,
              "auto_genes_per_pair": final.CONFUSION_GENES_PER_PAIR,
              "runs": {name: [] for name in args.configs}}

    for contrast in args.configs:
        print(f"\n[{contrast}] contrast"
              + (" · 고정 exact mutation 제거" if args.drop_exact else ""), flush=True)
        for seed in args.seeds:
            result, proba, classes = run_config(
                sdh, final, context, labels, case, seed, contrast, args.drop_exact)
            report["runs"][contrast].append(result)
            # ③ 에서 계약 포맷으로 내보낼 원자료를 남긴다.
            np.savez_compressed(proba_dir / f"oof_{contrast}{suffix}_seed{seed}.npz",
                                classes=classes, **proba)

    for contrast in args.configs:
        report[f"{contrast}_summary"] = {
            "champion": summarize(report["runs"][contrast], "champion"),
            "blend": summarize(report["runs"][contrast], "blend"),
        }

    # 두 구성을 같이 돌렸을 때만 짝지어 비교한다.
    deltas = {}
    if {"fixed", "auto"} <= set(args.configs):
        for key in ("champion", "blend"):
            per_seed = {}
            for fixed_row, auto_row in zip(report["runs"]["fixed"], report["runs"]["auto"]):
                assert fixed_row["seed"] == auto_row["seed"]
                per_seed[fixed_row["seed"]] = (auto_row[key]["macro_f1"]
                                               - fixed_row[key]["macro_f1"])
            values = list(per_seed.values())
            deltas[key] = {"per_seed": per_seed, "mean": float(np.mean(values)),
                           "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                           "positive_seeds": f"{sum(1 for v in values if v > 0)}/{len(values)}"}
        report["auto_minus_fixed"] = deltas
    report["note"] = ("자동 발견은 fold-train 안에서만 쌍을 정하므로 계약의 외부 지식 "
                      "금지·fold-only 조항을 만족한다. 비용이 작으면 규칙 리스크를 "
                      "없애는 쪽이 기대값이 크다.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    width = 72
    print("\n" + "=" * width)
    print(f"{'':10s}{'champion 평균':>18s}{'blend 평균':>16s}{'blend 표준편차':>16s}")
    print("-" * width)
    for contrast in args.configs:
        s = report[f"{contrast}_summary"]
        print(f"{contrast:10s}{s['champion']['mean']:>18.6f}"
              f"{s['blend']['mean']:>16.6f}{s['blend']['std']:>16.6f}")
    if deltas:
        print("-" * width)
        print(f"{'차이':10s}{deltas['champion']['mean']:>+18.6f}"
              f"{deltas['blend']['mean']:>+16.6f}{'':>16s}")
        print(f"{'':10s}{'':>18s}{deltas['blend']['positive_seeds']:>16s}  (자동이 더 높은 seed)")
    print("=" * width)
    if deltas:
        for seed in args.seeds:
            print(f"  seed {seed}: champion {deltas['champion']['per_seed'][seed]:+.6f} "
                  f"· blend {deltas['blend']['per_seed'][seed]:+.6f}")
    else:
        for contrast in args.configs:
            for row in report["runs"][contrast]:
                print(f"  seed {row['seed']}: champion {row['champion']['macro_f1']:.6f} "
                      f"· blend {row['blend']['macro_f1']:.6f} "
                      f"· 피처 {row['feature_count_mean']:,.1f}")
    print(f"\nOOF 확률: {proba_dir}")
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
