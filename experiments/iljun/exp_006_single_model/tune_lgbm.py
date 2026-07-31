"""
LGBM 단독 최고점 찾기 (트리 친화 표현 위에서)
===============================================

    .venv/bin/python3 experiments/iljun/exp_006_single_model/tune_lgbm.py
    .venv/bin/python3 experiments/iljun/exp_006_single_model/tune_lgbm.py --smoke

무엇을 하나
-----------
"단독 모델 각각의 천장을 먼저 찍는다"는 팀 방침. 여기선 LGBM.
LR 은 wide-sparse(GBV)에서 이미 수렴(~0.413)이라 앵커로만 재현하고,
LGBM 은 **트리 친화 표현(features_T: SVD+B+V)** 위에서 튜닝한다.

단계 (시간 절약: 탐색은 seed 42 만, 최종만 3-seed)
--------------------------------------------------
0. 참고점 — LGBM 을 GBV(wide-sparse)에 그냥 태우면 얼마나 낮나 (표현 불일치 확인)
1. SVD 차원 고르기 — {64,128,256} 중 (seed 42)
2. LGBM 격자 탐색 — num_leaves·min_child_samples·n_estimators (seed 42)
3. 최종 확정 — 상위 1개를 3-seed 로. LR 앵커(GBV, 3-seed)와 나란히.

★ CV↔LB 교훈: CV 이득의 ~63%만 LB 로 갔다. 여기 CV 최고도 제출로 확인해야
  진짜다. 특히 n_estimators 는 많을수록 CV 가 오르다 LB 에서 과적합으로 꺾일 수
  있으니(exp_004 에서 100>300 확인), 최댓값만 보고 채택하지 않는다.

fold 학습 분할에서만 fit. test 는 열지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
for p in (_HERE, _EXP2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline as pa                                                 # noqa: E402
import features_A as fa                                              # noqa: E402
import features_T as ft                                              # noqa: E402
import single_model as sm                                            # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from lightgbm import LGBMClassifier                                  # noqa: E402

# 팀 표준 LR (앵커 재현용)
LR_PARAMS = {"solver": "lbfgs", "C": 0.07, "max_iter": 2000, "class_weight": "balanced"}
# LGBM 고정 파라미터 (격자에서 안 흔드는 것)
LGBM_FIXED = {"objective": "multiclass", "class_weight": "balanced",
              "learning_rate": 0.05, "colsample_bytree": 0.7,
              "subsample": 0.8, "subsample_freq": 1,
              "n_jobs": -1, "verbose": -1}


# ── 표현(rep) 클로저들 ─────────────────────────────────────────────
def rep_gbv_sparse(train, counts, gene_cols, i_tr, i_va, seed):
    """LR·참고용 — GBV wide-sparse."""
    spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=seed)
    Xtr, _ = fa.build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, ("G", "B", "V"))
    Xva, _ = fa.build_features(train.iloc[i_va], counts.iloc[i_va], spec, ("G", "B", "V"))
    return Xtr, Xva


def make_rep_tree(svd_dim):
    """트리용 — SVD(G)+B+V dense. svd_dim 을 닫아 둔 클로저."""
    def _rep(train, counts, gene_cols, i_tr, i_va, seed):
        spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=seed)
        tspec = ft.fit_svd(train.iloc[i_tr], counts.iloc[i_tr], spec, svd_dim, seed=seed)
        Xtr = ft.transform(train.iloc[i_tr], counts.iloc[i_tr], spec, tspec)
        Xva = ft.transform(train.iloc[i_va], counts.iloc[i_va], spec, tspec)
        return Xtr, Xva
    return _rep


def lgbm_factory(params):
    def _make(seed):
        return LGBMClassifier(random_state=seed, **{**LGBM_FIXED, **params})
    return _make


def lr_factory(seed):
    return LogisticRegression(random_state=seed, **LR_PARAMS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LGBM 단독 최고점")
    ap.add_argument("--root", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cv_seeds = pa.CONFIRMATION_CV_SEEDS[: (2 if a.smoke else None)]
    seed1 = (cv_seeds[0],)                        # 탐색용 단일 seed
    n_splits = 2 if a.smoke else 5
    dims = [32, 64] if a.smoke else [64, 128, 256]
    if a.smoke:
        leaves_grid, mcs_grid, ntree_grid = [15], [20], [100]
    else:
        leaves_grid, mcs_grid, ntree_grid = [15, 31], [20, 50], [150, 300]

    print("=" * 92)
    print("  LGBM 단독 최고점 — 트리 친화 표현(SVD+B+V) 위에서")
    print(f"  탐색 seed {list(seed1)} · 최종 seed {list(cv_seeds)} · KFold-{n_splits}")
    print("=" * 92)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    counts, _ = pa.parse_all(train, test, gene_cols)

    def ev(make_model, make_rep, seeds):
        return sm.evaluate(train, y, gene_cols, counts, make_model, make_rep,
                           seeds, n_splits, model_seed=pa.MODEL_SEED)

    # ── 0. 참고점: LGBM on GBV wide-sparse (표현 불일치 확인) ───────────
    print("\n[0] 참고 — LGBM 을 GBV wide-sparse 에 그냥 태우기 (seed 탐색)")
    t0 = time.time()
    ref = ev(lgbm_factory({"num_leaves": 31, "min_child_samples": 20, "n_estimators": 300}),
             rep_gbv_sparse, seed1)
    ref_m = sm.summary(ref)[0]
    print(f"  LGBM/GBV-sparse  {ref_m:.5f}   ({time.time()-t0:.0f}s)")

    # ── 1. SVD 차원 고르기 (seed 탐색, 중간 LGBM 설정 고정) ─────────────
    print("\n[1] SVD 차원 탐색 (LGBM num_leaves31·mcs20·300trees 고정)")
    dim_scores = {}
    for d in dims:
        t0 = time.time()
        ps = ev(lgbm_factory({"num_leaves": 31, "min_child_samples": 20, "n_estimators": 300}),
                make_rep_tree(d), seed1)
        dim_scores[d] = sm.summary(ps)[0]
        print(f"  dim {d:4}  {dim_scores[d]:.5f}   ({time.time()-t0:.0f}s)", flush=True)
    best_dim = max(dim_scores, key=dim_scores.get)
    print(f"  → 최적 SVD 차원 {best_dim}  (GBV-sparse 참고점 {ref_m:.5f} 대비 "
          f"{dim_scores[best_dim]-ref_m:+.5f})")

    # ── 2. LGBM 격자 탐색 (best_dim, seed 탐색) ────────────────────────
    print(f"\n[2] LGBM 격자 탐색 (SVD {best_dim}, seed {list(seed1)})")
    rep_best = make_rep_tree(best_dim)
    grid = []
    for nl in leaves_grid:
        for mcs in mcs_grid:
            for nt in ntree_grid:
                grid.append({"num_leaves": nl, "min_child_samples": mcs, "n_estimators": nt})
    rows = []
    for g in grid:
        t0 = time.time()
        ps = ev(lgbm_factory(g), rep_best, seed1)
        m = sm.summary(ps)[0]
        rows.append({**g, "seed42": m})
        print(f"  leaves{g['num_leaves']:>3} mcs{g['min_child_samples']:>3} "
              f"trees{g['n_estimators']:>4}  {m:.5f}   ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["seed42"], reverse=True)
    print("\n  seed42 랭킹")
    print(pd.DataFrame(rows).to_string(index=False))
    top = rows[0]

    # ── 3. 최종 3-seed 확정 + LR 앵커 ─────────────────────────────────
    print(f"\n[3] 최종 확정 — 상위 설정 3-seed  ·  LR 앵커(GBV) 3-seed")
    top_cfg = {"num_leaves": top["num_leaves"],
               "min_child_samples": top["min_child_samples"],
               "n_estimators": top["n_estimators"]}
    t0 = time.time()
    lgbm_final = ev(lgbm_factory(top_cfg), rep_best, cv_seeds)
    lgbm_m, lgbm_s = sm.summary(lgbm_final)
    print(f"  LGBM(SVD{best_dim}, {top_cfg})")
    print(f"    3-seed  {lgbm_m:.5f} ± {lgbm_s}  (seed별 {lgbm_final})")
    lr_anchor = ev(lr_factory, rep_gbv_sparse, cv_seeds)
    lr_m, lr_s = sm.summary(lr_anchor)
    print(f"  LR(GBV)  앵커")
    print(f"    3-seed  {lr_m:.5f} ± {lr_s}  (seed별 {lr_anchor})")
    dm, dsd, pos, nn = sm.paired(lgbm_final, lr_anchor)
    print(f"\n  LGBM − LR  {dm:+.5f} ± {dsd}  ({pos}/{nn} seed 에서 LGBM 우세)")
    print(f"  (참고: LGBM/GBV-sparse {ref_m:.5f} → SVD 표현으로 {lgbm_m-ref_m:+.5f})")
    print("  ※ 이건 CV. LB 는 제출로. n_estimators 는 많을수록 LB 에서 꺾일 수 있음.")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        (art / "lgbm_tuning.json").write_text(json.dumps({
            "ref_lgbm_gbv_sparse_seed42": ref_m,
            "svd_dim_scores_seed42": dim_scores, "best_svd_dim": best_dim,
            "grid_seed42": rows,
            "final": {"model": "LGBM", "svd_dim": best_dim,
                      "params": {**LGBM_FIXED, **top_cfg},
                      "f1_macro": lgbm_m, "f1_macro_std": lgbm_s, "per_seed": lgbm_final},
            "lr_anchor": {"params": LR_PARAMS, "f1_macro": lr_m,
                          "f1_macro_std": lr_s, "per_seed": lr_anchor},
            "lgbm_minus_lr": {"mean": dm, "std": dsd, "pos": f"{pos}/{nn}"},
            "validation": pa.validation_spec(n_splits, pa.MODEL_SEED, cv_seeds),
            "caveat": "탐색은 seed42, 최종만 3-seed. CV 최고는 LB 로 확인.",
            "fingerprint": {**pa.fingerprint(),
                            "features_T_sha256": pa.sha256(ft.__file__)},
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n저장: artifacts/lgbm_tuning.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
