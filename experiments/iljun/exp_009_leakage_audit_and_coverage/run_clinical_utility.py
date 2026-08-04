"""이 모델로 무엇을 할 수 있는가 — 후보 좁히기·확신도 게이팅·클래스별 신뢰도.

    .venv/bin/python .../run_clinical_utility.py

왜 재는가
  Macro F1 0.52 는 대회 지표지만 "이 모델을 어디에 쓸 수 있나"에는 답하지 못한다.
  원발부위 불명암(CUP) 맥락에서 실제로 의미 있는 것은 top-1 정답이 아니라
  **26개를 몇 개로 좁혀 주는가**, 그리고 **어느 환자에게는 믿을 수 있는가** 다.

  임상에 들어간 CUP 분류기는 발현 94.63% / 메틸화 87.59% 다. 변이만 쓰는 우리
  모델이 그 자리를 대체할 수는 없다. 대신 확진 검사를 무엇부터 돌릴지 좁혀 주는
  triage 도구로서의 규격을 정량화한다.

재는 것
  1. top-k 정확도 (k=1,2,3,5,10) — 후보를 몇 개로 좁히면 정답이 들어오는가
  2. 정답 클래스의 순위 분포 — 중앙값 순위, 상위 3위 안에 드는 비율
  3. 확신도 게이팅 — max 확률로 정렬해 상위 X% 환자만 답할 때의 정확도
  4. 클래스별 신뢰도 — 어떤 암종에서 쓸 수 있는가

주의
  전부 **OOF(train 교차검증) 기준**이다. LB 는 이보다 낮다 — Macro F1 기준
  CV↔LB 간격이 약 0.089 다(exp_011). 절대값이 아니라 형태(좁히기 효과, 게이팅
  기울기, 클래스 순위)를 읽는 용도로 쓴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_submission as rs  # noqa: E402
import run_permutation_check as pc  # noqa: E402  (enrichment 구현 재사용)

COVERAGE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TOP_K = (1, 2, 3, 5, 10)


def oof_probabilities(cache, labels, seed: int):
    """OOF 확률 행렬 (n_samples x 26). 챔피언 구성 = base + class-enrichment."""
    classes = np.unique(labels)
    matrix = pc.token_presence(cache)
    splitter = StratifiedKFold(n_splits=rs.CONFIG.n_splits, shuffle=True, random_state=seed)
    series = pd.Series(labels)
    proba = np.zeros((len(labels), len(classes)), dtype=np.float64)

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        builder = rs.FoldMatrixBuilder(
            cache, rs.FINAL_CANDIDATE.backbone, rs.FINAL_CANDIDATE.exact_events,
            rs.FINAL_CANDIDATE.gene_pairs, rs.FINAL_CANDIDATE.gene_groups,
            rs.FINAL_CANDIDATE.hotspot_top_k, rs.FINAL_CANDIDATE.contrast_pairs,
            rs.FINAL_CANDIDATE.amino_mode, rs.FINAL_CANDIDATE.log1p_counts,
        )
        train_matrix, valid_matrix, _ = builder.build(tr, va, series)
        e_train, e_valid = pc.enrichment_features(matrix, labels, tr, va, classes, seed)
        train_matrix = sparse.hstack([train_matrix, sparse.csr_matrix(e_train)], format="csr")
        valid_matrix = sparse.hstack([valid_matrix, sparse.csr_matrix(e_valid)], format="csr")

        model = LogisticRegression(solver="lbfgs", C=rs.CONFIG.lr_c,
                                   max_iter=rs.CONFIG.lr_max_iter,
                                   class_weight="balanced", random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(train_matrix, labels[tr])
        assert list(model.classes_) == list(classes), "fold 마다 클래스 순서가 달라졌습니다"
        proba[va] = model.predict_proba(valid_matrix)
        print(f"      fold {fold}/5", flush=True)
    return proba, classes


def evaluate(proba, labels, classes) -> dict:
    truth_index = np.searchsorted(classes, labels)
    order = np.argsort(-proba, axis=1)                      # 확률 내림차순 클래스 순서
    rank = np.argmax(order == truth_index[:, None], axis=1)  # 정답의 0-based 순위

    top_k = {str(k): float((rank < k).mean()) for k in TOP_K}
    predicted = classes[order[:, 0]]

    confidence = proba.max(axis=1)
    by_confidence = np.argsort(-confidence)
    correct = (rank == 0)[by_confidence]
    gating = []
    for level in COVERAGE_LEVELS:
        cut = max(1, int(round(level * len(labels))))
        gating.append({
            "coverage": level,
            "n_patients": cut,
            "top1_accuracy": float(correct[:cut].mean()),
            "min_confidence": float(np.sort(-confidence)[cut - 1] * -1),
        })

    per_class = []
    for index, name in enumerate(classes):
        mask = labels == name
        per_class.append({
            "class": str(name),
            "support": int(mask.sum()),
            "f1": float(f1_score(labels == name, predicted == name, zero_division=0)),
            "top1": float((rank[mask] == 0).mean()),
            "top3": float((rank[mask] < 3).mean()),
            "median_rank": float(np.median(rank[mask]) + 1),
        })
    per_class.sort(key=lambda row: -row["f1"])

    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "top_k_accuracy": top_k,
        "median_rank_of_truth": float(np.median(rank) + 1),
        "mean_rank_of_truth": float(rank.mean() + 1),
        "confidence_gating": gating,
        "per_class": per_class,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(rs.CONFIG.stability_seeds))
    # exp_012 가 shrinkage 20 -> 10 으로 챔피언을 갱신했다(3seed 0.52824 ± 0.00187).
    # 현재 챔피언을 재는 것이 목적이므로 기본값을 10 으로 둔다.
    parser.add_argument("--shrinkage", type=float, default=10.0)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "clinical_utility.json")
    args = parser.parse_args(argv)
    pc.SHRINK = args.shrinkage

    root = pc.find_root(Path(__file__).resolve())
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in (rs.CONFIG.id_col, rs.CONFIG.target_col)]
    labels = train[rs.CONFIG.target_col].to_numpy()
    cache = rs.RowCache.build(train[genes], genes)

    runs = []
    for seed in args.seeds:
        print(f"\n  seed {seed}", flush=True)
        proba, classes = oof_probabilities(cache, labels, seed)
        result = evaluate(proba, labels, classes)
        result["seed"] = seed
        runs.append(result)
        print(f"    → Macro F1 {result['macro_f1']:.6f} · "
              f"top-1 {result['top_k_accuracy']['1']:.1%} · "
              f"top-3 {result['top_k_accuracy']['3']:.1%} · "
              f"top-5 {result['top_k_accuracy']['5']:.1%}", flush=True)

    summary = {
        "model": f"exp_012 champion (08 base + gene x event_type class-enrichment 26, "
                 f"shrinkage={pc.SHRINK:g}, min_support={pc.MIN_SUPPORT})",
        "note": "전부 OOF(train 5-fold) 기준. LB 는 이보다 낮다(Macro F1 기준 간격 약 0.089).",
        "seeds": args.seeds,
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
        "top_k_accuracy_mean": {
            str(k): float(np.mean([r["top_k_accuracy"][str(k)] for r in runs])) for k in TOP_K
        },
        "median_rank_of_truth_mean": float(np.mean([r["median_rank_of_truth"] for r in runs])),
        "confidence_gating_mean": [
            {"coverage": level,
             "top1_accuracy": float(np.mean([r["confidence_gating"][i]["top1_accuracy"] for r in runs]))}
            for i, level in enumerate(COVERAGE_LEVELS)
        ],
        "per_class_mean": None,
        "per_seed": runs,
    }
    names = [row["class"] for row in runs[0]["per_class"]]
    lookup = {name: [] for name in names}
    for run in runs:
        for row in run["per_class"]:
            lookup[row["class"]].append(row)
    summary["per_class_mean"] = sorted(
        [{"class": name,
          "support": rows[0]["support"],
          "f1": float(np.mean([r["f1"] for r in rows])),
          "top1": float(np.mean([r["top1"] for r in rows])),
          "top3": float(np.mean([r["top3"] for r in rows]))}
         for name, rows in lookup.items()],
        key=lambda row: -row["f1"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 64)
    print("후보를 몇 개로 좁혀 주는가 (26개 중, OOF 기준)")
    print("-" * 64)
    for k in TOP_K:
        print(f"  top-{k:<2d} 안에 정답  {summary['top_k_accuracy_mean'][str(k)]:>7.1%}")
    print(f"\n  정답의 중앙값 순위  {summary['median_rank_of_truth_mean']:.1f}위 / 26")
    print("\n" + "-" * 64)
    print("확신도 상위 X% 환자에게만 답할 때의 top-1 정확도")
    print("-" * 64)
    for row in summary["confidence_gating_mean"]:
        print(f"  상위 {row['coverage']:>4.0%}  →  {row['top1_accuracy']:>7.1%}")
    print("\n" + "-" * 64)
    print("클래스별 신뢰도 상위 8개 / 하위 4개")
    print("-" * 64)
    print(f"  {'클래스':10s}{'n':>6s}{'F1':>9s}{'top-1':>9s}{'top-3':>9s}")
    for row in summary["per_class_mean"][:8]:
        print(f"  {row['class']:10s}{row['support']:>6d}{row['f1']:>9.3f}"
              f"{row['top1']:>9.1%}{row['top3']:>9.1%}")
    print("  ...")
    for row in summary["per_class_mean"][-4:]:
        print(f"  {row['class']:10s}{row['support']:>6d}{row['f1']:>9.3f}"
              f"{row['top1']:>9.1%}{row['top3']:>9.1%}")
    print("=" * 64)
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
