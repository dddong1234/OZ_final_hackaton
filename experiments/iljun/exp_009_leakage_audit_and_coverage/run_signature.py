"""경수님 Event Ontology 의 signature 블록을 gs 1위 라인 위에서 독립 검증.

    .venv/bin/python experiments/iljun/exp_009_leakage_audit_and_coverage/run_signature.py

검증 대상
  gene x 기능유형 x 50aa-bin signature.  경수님 페이지 기준 seed 42 에서
  +0.004032 (8,175 -> 35,183 피처), 세 후보 중 최고.

왜 이 블록만 재는가
  run_ontology_coverage.py 로 세 후보의 test 전이율을 쟀더니 실제 운용 지점에서
    gene x position   ~1.5%   <- CV 이득은 큰데 test 에서 거의 죽는다
    gene x 50aa-bin   ~70%    <- 전이는 되는데 CV 이득이 미검출
    signature         ~35%    <- CV 최고 + 전이 중간
  signature 만 둘을 겸한다.  나머지 둘에 seed 를 쓸 이유가 없다.

설계
  - base 는 exp_009 하드닝 러너의 08 구성(3seed OOF 0.47850165, gs 기록과 일치).
    run_submission.py 는 검증된 제출 러너이므로 수정하지 않고, signature 블록만
    그 위에 hstack 한다.
  - signature 어휘는 **fold-train 행에서만** 만든다 (누수 방지).
  - baseline 은 artifacts/cv.json 의 같은 seed 값을 그대로 쓴다.
    StratifiedKFold(shuffle=True, random_state=seed) 는 결정적이라 fold 가
    동일하므로 이것이 정당한 paired 비교다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_submission as rs  # noqa: E402

POSITION_RE = re.compile(r"(\d+)")


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def signature_codes(cache: rs.RowCache) -> tuple[np.ndarray, np.ndarray, int]:
    """(row, code, n_codes) — fold 와 무관한 부분이라 캐시에 한 번만 만든다.

    key = gene __ 기능유형 __ b<50aa 구간>.  위치는 변이 문자열의 첫 정수를 쓴다
    (missense 뿐 아니라 frameshift/nonsense 표기에서도 위치를 얻기 위해).
    """
    cached = getattr(cache, "_signature_codes", None)
    if cached is not None:
        return cached
    events = cache.events
    if events.empty:
        result = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0)
    else:
        position = pd.to_numeric(
            events.event.str.extract(POSITION_RE, expand=False), errors="coerce"
        )
        bin50 = position // 50
        keep = bin50.notna().to_numpy()
        key = (events.gene.to_numpy()[keep].astype(object)
               + "__" + events.event_type.to_numpy()[keep].astype(object)
               + "__b" + bin50[keep].astype(int).astype(str).to_numpy().astype(object))
        codes, uniques = pd.factorize(pd.Index(key))
        result = (events.row.to_numpy()[keep], codes.astype(np.int64), len(uniques))
    cache._signature_codes = result
    return result


def signature_block(cache: rs.RowCache, train_index: np.ndarray, min_count: int):
    """fold-train 에서 min_count 회 이상 관측된 signature 만 열로 만든다."""
    rows, codes, n_codes = signature_codes(cache)
    n_rows = cache.mutation_matrix.shape[0]
    if n_codes == 0:
        return sparse.csr_matrix((n_rows, 0), dtype=np.float32), 0

    in_train = np.zeros(n_rows, dtype=bool)
    in_train[train_index] = True
    counts = np.bincount(codes[in_train[rows]], minlength=n_codes)
    selected = np.flatnonzero(counts >= min_count)
    if selected.size == 0:
        return sparse.csr_matrix((n_rows, 0), dtype=np.float32), 0

    remap = np.full(n_codes, -1, dtype=np.int64)
    remap[selected] = np.arange(selected.size)
    mapped = remap[codes]
    keep = mapped >= 0
    matrix = sparse.coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), (rows[keep], mapped[keep])),
        shape=(n_rows, selected.size),
    ).tocsr()
    matrix.data[:] = 1.0
    return matrix, selected.size


def run_seed(cache, labels, seed: int, min_count: int) -> dict:
    splitter = StratifiedKFold(n_splits=rs.CONFIG.n_splits, shuffle=True, random_state=seed)
    predicted = np.empty(len(labels), dtype=object)
    feature_counts, warnings_seen = [], 0
    started = perf_counter()

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        builder = rs.FoldMatrixBuilder(
            cache, rs.FINAL_CANDIDATE.backbone, rs.FINAL_CANDIDATE.exact_events,
            rs.FINAL_CANDIDATE.gene_pairs, rs.FINAL_CANDIDATE.gene_groups,
            rs.FINAL_CANDIDATE.hotspot_top_k, rs.FINAL_CANDIDATE.contrast_pairs,
            rs.FINAL_CANDIDATE.amino_mode, rs.FINAL_CANDIDATE.log1p_counts,
        )
        base_train, base_valid, base_names = builder.build(tr, va, labels)
        block, n_sig = signature_block(cache, tr, min_count)
        # fold-train 기준 상수열 제거 — base 쪽과 같은 기준을 적용한다.
        keep = rs.nonconstant_columns(block[tr]) if n_sig else np.zeros(0, dtype=bool)
        train_matrix = sparse.hstack([base_train, block[tr][:, keep]], format="csr")
        valid_matrix = sparse.hstack([base_valid, block[va][:, keep]], format="csr")

        model = LogisticRegression(
            solver="lbfgs", C=rs.CONFIG.lr_c, max_iter=rs.CONFIG.lr_max_iter,
            class_weight="balanced", random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, labels.iloc[tr])
        predicted[va] = model.predict(valid_matrix)
        warnings_seen += sum(issubclass(i.category, ConvergenceWarning) for i in caught)
        feature_counts.append(train_matrix.shape[1])
        print(f"    fold {fold}/5  피처 {train_matrix.shape[1]:,} "
              f"(base {len(base_names):,} + sig {int(keep.sum()):,})", flush=True)

    return {
        "seed": seed,
        "oof_macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "oof_accuracy": float(accuracy_score(labels, predicted)),
        "feature_count_mean": float(np.mean(feature_counts)),
        "convergence_warning_count": warnings_seen,
        "runtime_seconds": perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(rs.CONFIG.stability_seeds))
    parser.add_argument("--baseline", type=Path,
                        default=Path(__file__).parent / "artifacts" / "cv.json")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "signature.json")
    args = parser.parse_args(argv)

    baseline = {r["seed"]: r["oof_macro_f1"]
                for r in json.loads(args.baseline.read_text())["per_seed"]}

    root = find_root(Path(__file__).resolve())
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in (rs.CONFIG.id_col, rs.CONFIG.target_col)]
    labels = train[rs.CONFIG.target_col]
    cache = rs.RowCache.build(train[genes], genes)

    _, _, n_codes = signature_codes(cache)
    print(f"\nsignature 원어휘 {n_codes:,}개 · min_count={args.min_count}\n", flush=True)

    results, deltas = [], []
    for seed in args.seeds:
        print(f"  seed {seed}", flush=True)
        result = run_seed(cache, labels, seed, args.min_count)
        delta = result["oof_macro_f1"] - baseline[seed]
        result["baseline_oof_macro_f1"] = baseline[seed]
        result["delta"] = delta
        deltas.append(delta)
        results.append(result)
        print(f"    → {result['oof_macro_f1']:.6f}  (base {baseline[seed]:.6f})  "
              f"delta {delta:+.6f}  수렴경고 {result['convergence_warning_count']}\n", flush=True)

    positive = sum(1 for d in deltas if d > 0)
    summary = {
        "block": "gene x functype x 50aa-bin signature",
        "base": "exp_009 hardened 08 (H-AS + exact4 + contrast + A_pair + log1p)",
        "min_count": args.min_count,
        "seeds": args.seeds,
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "positive_seeds": f"{positive}/{len(deltas)}",
        "mean_oof_macro_f1": float(np.mean([r["oof_macro_f1"] for r in results])),
        "feature_count_mean": float(np.mean([r["feature_count_mean"] for r in results])),
        "convergence_warning_count": sum(r["convergence_warning_count"] for r in results),
        "reference_claim": {"source": "임경수 Event Ontology 정리", "seed42_delta": 0.004032,
                            "seed42_features": 35183},
        "per_seed": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"paired delta  {summary['mean_delta']:+.6f} ± {summary['std_delta']:.6f}"
          f"  ({summary['positive_seeds']} seed 양수)")
    print(f"피처 수 평균  {summary['feature_count_mean']:,.0f}"
          f"   (경수님 seed42 기준 35,183)")
    print(f"수렴 경고     {summary['convergence_warning_count']}")
    print("=" * 60)
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
