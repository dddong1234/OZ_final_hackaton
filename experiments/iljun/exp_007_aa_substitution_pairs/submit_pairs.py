"""
제출 파일 생성 — GBV + A_pair(log1p)
=====================================

    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/submit_pairs.py
    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/submit_pairs.py --smoke

exp_007 에서 CV 최대 리프트였던 축(A_pair)을 전체 train 으로 학습해 제출한다.
raw 카운트가 미수렴이었으므로 여기선 **log1p 스케일**을 적용해 깨끗이 수렴시킨다.

왜 이 한 장인가
---------------
A_pair 는 팀 상위(SDH·홍주)의 이기는 축이고, 우리 CV 를 0.413→0.450 으로 올렸다.
A/S 축은 팀 실측상 LB 전달이 잘 된다(SDH gap 0.122). 예상 LB ~0.32~0.34 로
지금 우리 최고(0.284)에서 큰 도약을 확인하는 게 목적이다.

marg 는 CV 를 깎아 제외(SDH case_06 와 동일). base 는 GBV(우리 실측 앵커, LB 0.2795).

규칙: pair 380 은 고정정의라 fit 없음. spec 은 전체 train 으로만 fit, test 는 예측만.
      외부 지식 없음. leakage 없음.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
_EXP3 = _HERE.parent / "exp_003_discriminative_tokens"
for p in (_HERE, _EXP2, _EXP3):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline as pa                                                 # noqa: E402
import features_A as fa                                              # noqa: E402
import features_P as fp                                              # noqa: E402
from run_tokens import load_cfg                                      # noqa: E402

BASE = ("G", "B", "V")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    root = pa.find_project_root()
    cfg = load_cfg()
    mp = dict(cfg["model_params"])                    # 팀 표준 C=0.07 max_iter=2000
    model_seed = cfg.get("model_seed", 42)
    n_splits = 2 if a.smoke else cfg["n_splits"]
    name, fn = pa.MODELS[cfg["model"]]

    out = root / "experiments" / "iljun" / "results" / "iljun-exp007-pairs"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  제출 — GBV + A_pair(log1p)")
    print(f"  {mp} · base {''.join(BASE)} + pair380(log1p)")
    print("=" * 80)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    counts_tr, counts_te = pa.parse_all(train, test, gene_cols)

    # A_pair (고정정의, fit 없음) — 전체 train/test 각각 생성
    Ptr, _ = fp.build_P(train, gene_cols, "p", log1p=True)
    Pte, _ = fp.build_P(test, gene_cols, "p", log1p=True)
    print(f"  A_pair {Ptr.shape[1]}차원 (log1p)")

    # ── seed42 CV 정합성 + 수렴 점검 ──────────────────────────────────
    print("\n[1] seed42 CV + 수렴 점검")
    t0 = time.time()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.empty(len(y), dtype=object)
    n_warn = 0
    for i_tr, i_va in cv.split(train, y):
        spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=model_seed)
        Xa, _ = fa.build_features(train.iloc[i_tr], counts_tr.iloc[i_tr], spec, BASE)
        Xb, _ = fa.build_features(train.iloc[i_va], counts_tr.iloc[i_va], spec, BASE)
        Xa = sparse.hstack([Xa, Ptr[i_tr]], format="csr")
        Xb = sparse.hstack([Xb, Ptr[i_va]], format="csr")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            model = fn(model_seed, mp).fit(Xa, y[i_tr])
            n_warn += sum(1 for x in w if issubclass(x.category, ConvergenceWarning))
        oof[i_va] = model.predict(Xb)
    oof = np.array(list(oof))
    cv_f1 = round(float(f1_score(y, oof, average="macro")), 5)
    cv_acc = round(float(accuracy_score(y, oof)), 5)
    conv = "수렴(경고 0건)" if n_warn == 0 else f"미수렴 {n_warn}/{n_splits} fold"
    print(f"  seed42 CV Macro F1 {cv_f1:.5f} / Acc {cv_acc:.5f}  · {conv}")
    print(f"  (raw 버전 3-seed 0.45025 와 대조 — log1p 로 수렴 확인)  ({time.time()-t0:.0f}s)")

    # ── 전체 train fit → test 예측 ────────────────────────────────────
    print("\n[2] 전체 train 학습 → test 예측")
    t0 = time.time()
    spec = fa.fit_spec(train, gene_cols, seed=model_seed)
    Xtr, _ = fa.build_features(train, counts_tr, spec, BASE)
    Xte, _ = fa.build_features(test, counts_te, spec, BASE)
    Xtr = sparse.hstack([Xtr, Ptr], format="csr")
    Xte = sparse.hstack([Xte, Pte], format="csr")
    pred = fn(model_seed, mp).fit(Xtr, y).predict(Xte)
    print(f"  ({time.time()-t0:.0f}s)")

    # ── 검증 + 저장 ───────────────────────────────────────────────────
    ids = test[pa.ID].values
    df = pd.DataFrame({pa.ID: ids, pa.TARGET: pred})
    assert len(df) == len(test), "행 수 불일치"
    assert df[pa.TARGET].notna().all(), "NaN 예측"
    unknown = set(df[pa.TARGET]) - set(y)
    assert not unknown, f"train 에 없는 라벨: {unknown}"
    f = out / f"submission_GBVpair_log1p_cv{cv_f1:.5f}.csv"
    df.to_csv(f, index=False)

    print("\n" + "=" * 80)
    print(f"  저장: {f}")
    print(f"  분포 상위: {df[pa.TARGET].value_counts().head(5).to_dict()}")
    print(f"  예상 LB ~0.32~0.34 (A/S 축은 전달 잘 됨). 현재 최고 0.284 대비 큰 도약.")
    print("  업로드 후 LB 를 TEST log(LR-007a)에 적어주세요.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
