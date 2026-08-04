"""SDH exp_011 class-enrichment 의 supervised FE 경로 누수 감사.

    # 1) 재현 확인 + permutation check (기본)
    .venv/bin/python .../run_permutation_check.py

    # 2) permutation 만
    .venv/bin/python .../run_permutation_check.py --skip-real

왜 필요한가
  exp_011 은 label 을 써서 피처를 만드는 supervised FE 다(암종별 class-enrichment
  score 26개).  이 범주는 코드 검증에서 가장 집중적으로 보는 지점이고, 누수로
  판정되면 수상 제외다.  LB 가 CV 만큼 올랐다는 것(+0.048 vs +0.045)이 이미 강한
  정황 증거지만, permutation check 는 그것과 독립적인 직접 증거다.

무엇을 재는가
  label 을 무작위로 섞은 뒤 **같은 파이프라인을 그대로** 돌린다.  섞인 label 에는
  실제 신호가 없으므로 OOF Macro F1 은 우연 수준(26클래스, 약 0.04)이어야 한다.

    핵심 비교 — permutation 하에서 enrichment 가 base 보다 높은가?

  높다면 cross-fit 이 깨져서 모델이 섞인 label 을 되찾고 있다는 뜻이다(= 누수).
  같다면 enrichment 는 label 정보를 학습 행으로 흘리지 않는다(= 안전).

구현 근거
  SDH 'TEAM_REPORT.md' 3.4 / 4.1 절의 수식과 절차를 그대로 옮겼다.
    token       = gene + "__" + event_type  (샘플 단위 presence)
    raw_weight  = log((n_pos+a)/(N_pos-n_pos+a)) - log((n_neg+a)/(N_neg-n_neg+a))
    weight      = clip(raw_weight * support/(support+20), -4, 4),  a=1, support>=10
    score_c(x)  = sum(weight(c,t) for active t) / sqrt(active token 수)
  중첩 cross-fit: outer-train 을 다시 inner 5-fold 로 나눠 OOF enrichment 를 만들고,
  outer-valid 에는 outer-train 전체로 학습한 weight 를 적용만 한다.
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
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_submission as rs  # noqa: E402

ALPHA, MIN_SUPPORT, SHRINK, CLIP = 1.0, 10, 20.0, 4.0


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def token_presence(cache: rs.RowCache) -> sparse.csr_matrix:
    """(n_rows x n_tokens) 이진 행렬. token = gene __ event_type, 샘플 단위 presence."""
    events = cache.events
    key = events.gene.to_numpy().astype(object) + "__" + events.event_type.to_numpy().astype(object)
    codes, uniques = pd.factorize(pd.Index(key))
    matrix = sparse.coo_matrix(
        (np.ones(len(codes), dtype=np.float32), (events.row.to_numpy(), codes)),
        shape=(cache.mutation_matrix.shape[0], len(uniques)),
    ).tocsr()
    matrix.data[:] = 1.0          # 중복 event 는 presence 1 로 접는다
    return matrix


def fit_weights(matrix, labels, rows, classes) -> np.ndarray:
    """rows(학습 분할) 에서만 암종별 token log-odds weight 를 학습한다."""
    sub = matrix[rows]
    y = labels[rows]
    support = np.asarray(sub.sum(axis=0)).ravel()
    eligible = support >= MIN_SUPPORT
    total = len(rows)
    weights = np.zeros((len(classes), matrix.shape[1]), dtype=np.float32)
    for index, name in enumerate(classes):
        mask = y == name
        n_positive_rows = int(mask.sum())
        n_negative_rows = total - n_positive_rows
        if n_positive_rows == 0 or n_negative_rows == 0:
            continue
        n_pos = np.asarray(sub[mask].sum(axis=0)).ravel()
        n_neg = support - n_pos
        raw = (np.log((n_pos + ALPHA) / (n_positive_rows - n_pos + ALPHA))
               - np.log((n_neg + ALPHA) / (n_negative_rows - n_neg + ALPHA)))
        shrunk = np.clip(raw * support / (support + SHRINK), -CLIP, CLIP)
        shrunk[~eligible] = 0.0
        weights[index] = shrunk
    return weights


def apply_weights(matrix, rows, weights) -> np.ndarray:
    active = np.asarray(matrix[rows].sum(axis=1)).ravel()
    return (matrix[rows] @ weights.T) / np.sqrt(np.maximum(active, 1.0))[:, None]


def enrichment_features(matrix, labels, outer_train, outer_valid, classes, seed):
    """4.1 절의 중첩 cross-fit. 학습 행은 자기 label 이 든 weight 를 받지 않는다."""
    scores_train = np.zeros((len(outer_train), len(classes)), dtype=np.float32)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_outer = labels[outer_train]
    for inner_train, inner_holdout in inner.split(np.zeros(len(outer_train)), y_outer):
        weights = fit_weights(matrix, labels, outer_train[inner_train], classes)
        scores_train[inner_holdout] = apply_weights(matrix, outer_train[inner_holdout], weights)
    weights_full = fit_weights(matrix, labels, outer_train, classes)
    scores_valid = apply_weights(matrix, outer_valid, weights_full)

    mean = scores_train.mean(axis=0)
    std = scores_train.std(axis=0)
    std[std == 0] = 1.0
    return (scores_train - mean) / std, (scores_valid - mean) / std


def run(cache, labels, seed: int, with_enrichment: bool, tag: str) -> dict:
    classes = np.unique(labels)
    matrix = token_presence(cache) if with_enrichment else None
    splitter = StratifiedKFold(n_splits=rs.CONFIG.n_splits, shuffle=True, random_state=seed)
    series = pd.Series(labels)
    predicted = np.empty(len(labels), dtype=object)
    warnings_seen = 0
    started = perf_counter()

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        builder = rs.FoldMatrixBuilder(
            cache, rs.FINAL_CANDIDATE.backbone, rs.FINAL_CANDIDATE.exact_events,
            rs.FINAL_CANDIDATE.gene_pairs, rs.FINAL_CANDIDATE.gene_groups,
            rs.FINAL_CANDIDATE.hotspot_top_k, rs.FINAL_CANDIDATE.contrast_pairs,
            rs.FINAL_CANDIDATE.amino_mode, rs.FINAL_CANDIDATE.log1p_counts,
        )
        train_matrix, valid_matrix, _ = builder.build(tr, va, series)
        if with_enrichment:
            e_train, e_valid = enrichment_features(matrix, labels, tr, va, classes, seed)
            train_matrix = sparse.hstack([train_matrix, sparse.csr_matrix(e_train)], format="csr")
            valid_matrix = sparse.hstack([valid_matrix, sparse.csr_matrix(e_valid)], format="csr")

        model = LogisticRegression(solver="lbfgs", C=rs.CONFIG.lr_c,
                                   max_iter=rs.CONFIG.lr_max_iter,
                                   class_weight="balanced", random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, labels[tr])
        predicted[va] = model.predict(valid_matrix)
        warnings_seen += sum(issubclass(i.category, ConvergenceWarning) for i in caught)
        print(f"      fold {fold}/5  피처 {train_matrix.shape[1]:,}", flush=True)

    score = float(f1_score(labels, predicted, average="macro", zero_division=0))
    print(f"    → {tag}: OOF Macro F1 {score:.6f}  ({(perf_counter()-started)/60:.1f}분)\n", flush=True)
    return {"tag": tag, "oof_macro_f1": score, "convergence_warning_count": warnings_seen}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutation-seed", type=int, default=20260803)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "permutation_check.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in (rs.CONFIG.id_col, rs.CONFIG.target_col)]
    labels = train[rs.CONFIG.target_col].to_numpy()
    cache = rs.RowCache.build(train[genes], genes)

    report = {"seed": args.seed, "permutation_seed": args.permutation_seed,
              "n_classes": int(len(np.unique(labels))), "runs": {}}

    if not args.skip_real:
        print("\n[1] 실제 label — 재구현이 SDH 숫자를 재현하는가", flush=True)
        report["runs"]["real_enrichment"] = run(cache, labels, args.seed, True, "real + enrichment")

    rng = np.random.default_rng(args.permutation_seed)
    shuffled = labels[rng.permutation(len(labels))]
    print(f"[2] label 무작위 셔플 (permutation_seed={args.permutation_seed})", flush=True)
    report["runs"]["permuted_base"] = run(cache, shuffled, args.seed, False, "permuted + base")
    report["runs"]["permuted_enrichment"] = run(cache, shuffled, args.seed, True, "permuted + enrichment")

    base_p = report["runs"]["permuted_base"]["oof_macro_f1"]
    enr_p = report["runs"]["permuted_enrichment"]["oof_macro_f1"]
    chance = 1.0 / report["n_classes"]
    report["permuted_delta"] = enr_p - base_p
    report["chance_level"] = chance
    # 누수 판정: 섞인 label 에서 enrichment 가 base 를 의미 있게 넘으면 cross-fit 이 깨진 것
    report["verdict"] = "PASS" if (enr_p - base_p) < 0.01 and enr_p < chance * 3 else "FAIL"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 62)
    if not args.skip_real:
        real = report["runs"]["real_enrichment"]["oof_macro_f1"]
        print(f"실제 label  + enrichment : {real:.6f}   (SDH 보고 seed42 0.52640)")
    print(f"섞인 label  + base       : {base_p:.6f}")
    print(f"섞인 label  + enrichment : {enr_p:.6f}")
    print(f"섞인 label  delta        : {report['permuted_delta']:+.6f}")
    print(f"우연 수준 (1/{report['n_classes']})         : {chance:.6f}")
    print("=" * 62)
    if report["verdict"] == "PASS":
        print("✅ PASS — 섞인 label 에서 enrichment 가 base 를 넘지 못한다.")
        print("   supervised FE 가 학습 행으로 label 을 흘리지 않는다는 직접 증거다.")
    else:
        print("🚨 FAIL — 섞인 label 에서 enrichment 가 base 를 넘었다.")
        print("   cross-fit 이 깨져 label 이 학습 피처로 새고 있다. 제출하면 안 된다.")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
