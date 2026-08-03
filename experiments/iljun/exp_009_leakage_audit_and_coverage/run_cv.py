"""exp_009 하드닝 파이프라인의 CV 재현 — gs 기준값 0.478502 와 대조.

    .venv/bin/python experiments/iljun/exp_009_leakage_audit_and_coverage/run_cv.py

왜 도는가
  exp_009 의 동치성 검증은 '전체 train 학습 → test 예측' 경로만 확인했다.  CV 는
  다른 코드 경로(fold 분할)를 타므로 따로 재현해야 TEST log 에 올릴 수 있다.

  gs 의 CV 경로(`exp-gs-002-memory-safe.py:753`)는 원래부터 `RowCache.build(train)`
  으로 train 만 쓴다 — concat 은 제출 생성 경로에만 있었다.  따라서 exp_009 의
  어휘 분리 리팩터는 CV 를 바꾸지 않아야 하고, 이 스크립트가 그것을 확인한다.

프로토콜 (gs `run_oof` 와 동일)
  StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  seeds 42 / 2024 / 777
  LogisticRegression(lbfgs, C=0.07, max_iter=2000, class_weight='balanced',
                     random_state=seed)
  점수는 fold 평균이 아니라 **OOF 전체**에 대한 macro F1 (zero_division=0)

기준값 (exp-gs-002-08A_factorial_summary.csv)
  H-AS-LR-exact-confusion-pairs-Apair-log1p
  macro_f1_mean 0.47850164684994484 ± 0.002483753024006579
  feature_count_mean 8173.533333 · convergence_warning_count 0
"""
from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_submission as rs  # noqa: E402

REFERENCE = {
    "source": "exp-gs-002-08A_factorial_summary.csv",
    "experiment_id": "H-AS-LR-exact-confusion-pairs-Apair-log1p",
    "macro_f1_mean": 0.47850164684994484,
    "macro_f1_std": 0.002483753024006579,
    "feature_count_mean": 8173.533333333333,
    "convergence_warning_count": 0,
}


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def run_oof(cache: rs.RowCache, labels: pd.Series, seed: int) -> dict:
    """gs run_oof 와 같은 절차. cache 는 train 만으로 만든 것이어야 한다."""
    splitter = StratifiedKFold(n_splits=rs.CONFIG.n_splits, shuffle=True, random_state=seed)
    predicted = np.empty(len(labels), dtype=object)
    fold_scores: list[float] = []
    feature_counts: list[int] = []
    warnings_seen = 0
    started = perf_counter()

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        builder = rs.FoldMatrixBuilder(
            cache,
            rs.FINAL_CANDIDATE.backbone,
            rs.FINAL_CANDIDATE.exact_events,
            rs.FINAL_CANDIDATE.gene_pairs,
            rs.FINAL_CANDIDATE.gene_groups,
            rs.FINAL_CANDIDATE.hotspot_top_k,
            rs.FINAL_CANDIDATE.contrast_pairs,
            rs.FINAL_CANDIDATE.amino_mode,
            rs.FINAL_CANDIDATE.log1p_counts,
        )
        train_matrix, valid_matrix, names = builder.build(tr, va, labels)
        model = LogisticRegression(
            solver="lbfgs", C=rs.CONFIG.lr_c, max_iter=rs.CONFIG.lr_max_iter,
            class_weight="balanced", random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, labels.iloc[tr])
        prediction = model.predict(valid_matrix)
        predicted[va] = prediction
        fold_scores.append(f1_score(labels.iloc[va], prediction, average="macro", zero_division=0))
        feature_counts.append(len(names))
        warnings_seen += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        print(f"    fold {fold}/{rs.CONFIG.n_splits}  macro F1 {fold_scores[-1]:.5f}  "
              f"features {len(names):,}", flush=True)

    return {
        "seed": seed,
        "oof_macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "oof_accuracy": float(accuracy_score(labels, predicted)),
        "fold_macro_f1_mean": float(np.mean(fold_scores)),
        "fold_macro_f1_std": float(np.std(fold_scores)),
        "feature_count_mean": float(np.mean(feature_counts)),
        "convergence_warning_count": warnings_seen,
        "runtime_seconds": perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(rs.CONFIG.stability_seeds))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts" / "cv.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    data_dir = args.data_dir or root / "data" / "raw"
    train = pd.read_csv(data_dir / "train.csv")
    genes = [c for c in train.columns if c not in (rs.CONFIG.id_col, rs.CONFIG.target_col)]
    labels = train[rs.CONFIG.target_col]

    # test 는 읽지도 않는다 — CV 에 필요 없고, 안 읽는 것이 가장 강한 누수 방어다.
    print(f"train {train.shape} · 유전자 {len(genes):,} · 클래스 {labels.nunique()}", flush=True)
    cache = rs.RowCache.build(train[genes], genes)

    results = []
    for seed in args.seeds:
        print(f"\n  seed {seed}", flush=True)
        result = run_oof(cache, labels, seed)
        print(f"    → OOF macro F1 {result['oof_macro_f1']:.6f} · "
              f"acc {result['oof_accuracy']:.6f} · "
              f"수렴경고 {result['convergence_warning_count']}", flush=True)
        results.append(result)

    scores = [r["oof_macro_f1"] for r in results]
    summary = {
        "experiment_id": rs.FINAL_CANDIDATE.experiment_id,
        "pipeline": "exp_009 hardened (train-only vocabulary)",
        "validation": f"StratifiedKFold-{rs.CONFIG.n_splits}",
        "seeds": args.seeds,
        "macro_f1_mean": float(np.mean(scores)),
        # gs 기준값과 같은 관례를 쓴다 — 표본표준편차(ddof=1).
        # ddof=0 이면 같은 데이터에서 0.002028 vs 0.002484 로 갈려 대조가 안 된다.
        "macro_f1_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "accuracy_mean": float(np.mean([r["oof_accuracy"] for r in results])),
        "feature_count_mean": float(np.mean([r["feature_count_mean"] for r in results])),
        "convergence_warning_count": sum(r["convergence_warning_count"] for r in results),
        "per_seed": results,
        "reference": REFERENCE,
        "delta_vs_reference": float(np.mean(scores) - REFERENCE["macro_f1_mean"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    delta = summary["delta_vs_reference"]
    print("\n" + "=" * 66)
    print(f"{'':22s}{'이번 실행':>16s}{'gs 기준값':>16s}{'차이':>10s}")
    print("-" * 66)
    print(f"{'OOF Macro F1 평균':20s}{summary['macro_f1_mean']:>16.6f}"
          f"{REFERENCE['macro_f1_mean']:>16.6f}{delta:>+10.6f}")
    print(f"{'표준편차':20s}{summary['macro_f1_std']:>16.6f}"
          f"{REFERENCE['macro_f1_std']:>16.6f}")
    print(f"{'피처 수 평균':20s}{summary['feature_count_mean']:>16.2f}"
          f"{REFERENCE['feature_count_mean']:>16.2f}")
    print(f"{'수렴 경고':20s}{summary['convergence_warning_count']:>16d}"
          f"{REFERENCE['convergence_warning_count']:>16d}")
    print("=" * 66)
    if abs(delta) < 5e-4:
        print(f"✅ 재현 — 차이 {abs(delta):.6f} < 0.0005 (팀 규정: 소수점 셋째 자리까지 비교)")
    else:
        print(f"⚠ 차이 {delta:+.6f} — 환경 차이(3.14 vs 3.12)인지 로직 차이인지 확인 필요")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
