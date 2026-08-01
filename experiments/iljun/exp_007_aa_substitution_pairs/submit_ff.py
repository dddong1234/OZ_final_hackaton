"""
제출 파일 생성 — functional-full(SDH base) + A_pair(log1p) [+S]
================================================================

    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/submit_ff.py
    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/submit_ff.py --with-s
    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/submit_ff.py --smoke

run_pairs_ff 에서 CV 최고였던 구성을 전체 train 으로 학습해 제출한다.
  기본(--with-s 없음): ff + pair  = CV 0.46952 (SDH 0.4636 초과). 예상 LB ~0.348.
  --with-s          : ff + pair + S = CV 0.46693. 홍주식 A+S — LB 일반화 후보.

규칙: SDH base 는 전체 train 으로만 fit, test 는 transform·예측만. A_pair 고정정의.
      외부지식 없음. leakage 없음.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
_EXP3 = _HERE.parent / "exp_003_discriminative_tokens"
_REPO = _HERE.parents[2]
for p in (_HERE, _EXP2, _EXP3, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline as pa                                                 # noqa: E402
import features_P as fp                                              # noqa: E402
from run_tokens import load_cfg                                      # noqa: E402
from experiments.SDH.exp_007_fe_combinations.preprocessing import (  # noqa: E402
    CombinedMutationTransformer,
)


def to_csr(x):
    arr = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    return sparse.csr_matrix(arr.astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-s", action="store_true", help="S(표기구조) 블록 추가")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    root = pa.find_project_root()
    cfg = load_cfg()
    mp = dict(cfg["model_params"])
    model_seed = cfg.get("model_seed", 42)
    name, fn = pa.MODELS[cfg["model"]]
    tag = "ff+pair+S" if a.with_s else "ff+pair"

    out = root / "experiments" / "iljun" / "results" / "iljun-exp007-pairs"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  제출 — {tag}  (SDH functional-full base + A_pair log1p"
          + (" + S)" if a.with_s else ")"))
    print(f"  {mp}")
    print("=" * 80)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values

    print("SDH functional-full base fit (전체 train)...", flush=True)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = CombinedMutationTransformer().fit(train[gene_cols], y)
        Xtr = to_csr(tf.transform(train[gene_cols]))
        Xte = to_csr(tf.transform(test[gene_cols]))
    print(f"  ff {Xtr.shape[1]}차원  ({time.time()-t0:.0f}s)")

    blocks = "ps" if a.with_s else "p"
    Ptr, _ = fp.build_P(train, gene_cols, blocks, log1p=True)
    Pte, _ = fp.build_P(test, gene_cols, blocks, log1p=True)
    Xtr = sparse.hstack([Xtr, Ptr], format="csr")
    Xte = sparse.hstack([Xte, Pte], format="csr")
    print(f"  최종 {Xtr.shape[1]}차원 (base + {blocks})")

    print("\n전체 train 학습 → test 예측...", flush=True)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model = fn(model_seed, mp).fit(Xtr, y)
        n_warn = sum(1 for x in w if issubclass(x.category, ConvergenceWarning))
    pred = model.predict(Xte)
    conv = "수렴(경고 0건)" if n_warn == 0 else f"⚠ 미수렴 {n_warn}건"
    print(f"  {conv}  ({time.time()-t0:.0f}s)")

    ids = test[pa.ID].values
    df = pd.DataFrame({pa.ID: ids, pa.TARGET: pred})
    assert len(df) == len(test), "행 수 불일치"
    assert df[pa.TARGET].notna().all(), "NaN 예측"
    unknown = set(df[pa.TARGET]) - set(y)
    assert not unknown, f"train 에 없는 라벨: {unknown}"
    f = out / f"submission_{tag.replace('+','_')}.csv"
    df.to_csv(f, index=False)

    cv_ref = "0.46693" if a.with_s else "0.46952"
    print("\n" + "=" * 80)
    print(f"  저장: {f}")
    print(f"  분포 상위: {df[pa.TARGET].value_counts().head(5).to_dict()}")
    print(f"  기록 CV(3-seed) {cv_ref} · 예상 LB ~0.34~0.35 (SDH 0.342/홍주 0.351 수준)")
    print("  업로드 후 LB 를 TEST log(LR-007c/007d)에 적어주세요.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
