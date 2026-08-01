"""
functional-full(SDH base) + A_pair(+S) — SDH 따라잡기/추월
============================================================

    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/run_pairs_ff.py
    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/run_pairs_ff.py --smoke

왜
--
exp_007(GBV base + A_pair)는 CV 0.456 으로 SDH sdh-009(functional-full + A_pair,
CV 0.4636 / LB 0.342)에 0.007 못 미쳤다. 차이의 원인은 base — SDH 는 'functional
full'(전체 토큰), 우리는 GBV+근사토큰이었다. 그래서 여기선 **SDH 의 실제 base
(CombinedMutationTransformer)를 그대로** 쓰고 그 위에 우리 A_pair(log1p)[+S]를 얹는다.

  ff            SDH functional-full base 단독 (case_01 재현, ~0.432)
  ff + pair     + 치환쌍 380 (SDH case_06 재현 목표, ~0.4636)
  ff + pair + S + 표기구조 S (홍주 A+S 계열 — CV 는 낮을 수 있으나 LB 일반화가
                  더 좋았다: 홍주 A+S LB 0.351 > SDH A_pair LB 0.342)

★ marg(ref/alt)는 exp_007 에서 -0.03 으로 해로워 뺐다.
★ S 는 CV 를 크게 안 올릴 수 있다 — 진짜 가치는 LB. CV 만으로 S 를 버리지 말 것.

규칙: base 는 fold-train 에서만 fit(SDH transformer 가 fold vocab 학습). A_pair 는
      고정정의(fit 없음). test 미열람. 외부지식 없음.
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
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
_EXP3 = _HERE.parent / "exp_003_discriminative_tokens"
_REPO = _HERE.parents[2]                                  # repo 루트 (experiments.* import 용)
for p in (_HERE, _EXP2, _EXP3, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline as pa                                                 # noqa: E402
import features_P as fp                                              # noqa: E402
from run_tokens import load_cfg                                      # noqa: E402
from experiments.SDH.exp_007_fe_combinations.preprocessing import (  # noqa: E402
    CombinedMutationTransformer,
)


def macro(y, p):
    return round(float(f1_score(y, p, average="macro")), 5)


def paired(after, before):
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def to_csr(df_or_arr):
    arr = df_or_arr.to_numpy() if hasattr(df_or_arr, "to_numpy") else np.asarray(df_or_arr)
    return sparse.csr_matrix(arr.astype(np.float32))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="functional-full + A_pair(+S)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = load_cfg()
    mp = dict(cfg["model_params"])                        # 팀 표준 C=0.07 max_iter=2000
    cv_seeds = tuple(cfg["cv_seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg.get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["n_splits"]
    name, fn = pa.MODELS[cfg["model"]]

    print("=" * 92)
    print("  functional-full(SDH base) + A_pair(+S) — SDH 따라잡기")
    print(f"  {mp} · cv_seeds {list(cv_seeds)} · KFold-{n_splits}")
    print("=" * 92)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values

    # A_pair(log1p) · S — 고정정의, 전체 train 에서 한 번 만들어 인덱스로 슬라이스
    P_pair, _ = fp.build_P(train, gene_cols, "p", log1p=True)
    P_s, _ = fp.build_P(train, gene_cols, "s")
    print(f"  A_pair {P_pair.shape[1]} (log1p) · S {P_s.shape[1]}", flush=True)

    stages = [("ff", None), ("ff+pair", P_pair),
              ("ff+pair+S", sparse.hstack([P_pair, P_s], format="csr"))]
    per_seed = {lab: {} for lab, _ in stages}
    ff_dim = None

    for s in cv_seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        oof = {lab: np.empty(len(y), dtype=object) for lab, _ in stages}
        t0 = time.time()
        for i_tr, i_va in cv.split(train, y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tf = CombinedMutationTransformer().fit(
                    train.iloc[i_tr][gene_cols], y[i_tr])
                Xff_tr = to_csr(tf.transform(train.iloc[i_tr][gene_cols]))
                Xff_va = to_csr(tf.transform(train.iloc[i_va][gene_cols]))
            ff_dim = Xff_tr.shape[1]
            for lab, extra in stages:
                Xa, Xb = Xff_tr, Xff_va
                if extra is not None:
                    Xa = sparse.hstack([Xff_tr, extra[i_tr]], format="csr")
                    Xb = sparse.hstack([Xff_va, extra[i_va]], format="csr")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    oof[lab][i_va] = fn(model_seed, mp).fit(Xa, y[i_tr]).predict(Xb)
        for lab, _ in stages:
            per_seed[lab][s] = macro(y, np.array(list(oof[lab])))
        print("  cv_seed {}  ".format(s)
              + "  ".join(f"{lab}:{per_seed[lab][s]:.5f}" for lab, _ in stages)
              + f"   (ff {ff_dim}차원, {time.time()-t0:.0f}s)", flush=True)

    def ms(lab):
        v = np.array(list(per_seed[lab].values()), dtype=float)
        return round(v.mean(), 5), (round(v.std(ddof=1), 5) if len(v) > 1 else None)

    print("\n" + "=" * 92)
    print("  누적 — SDH sdh-009: CV 0.4636 / LB 0.342 와 대조")
    print("=" * 92)
    rows, prev = [], None
    for lab, _ in stages:
        m, sd = ms(lab)
        if prev is None:
            rows.append({"단계": lab, "Macro F1": m, "σ": sd, "직전 대비": "—", "양수seed": "—"})
        else:
            dm, dsd, pos, nn = paired(per_seed[lab], per_seed[prev])
            rows.append({"단계": lab, "Macro F1": m, "σ": sd,
                         "직전 대비": f"{dm:+.5f}", "양수seed": f"{pos}/{nn}"})
        prev = lab
    print(pd.DataFrame(rows).to_string(index=False))

    ffp_m, _ = ms("ff+pair")
    print(f"\n  ff+pair {ffp_m:.5f}  vs  SDH sdh-009 0.4636  ({ffp_m-0.4636:+.5f})")
    print(f"  예상 LB (SDH gap 0.121 적용) ~{ffp_m-0.121:.4f}")
    print("  ※ S 는 CV 로 판단 말 것 — 홍주 A+S 는 CV 낮아도 LB 0.351 로 SDH 0.342 넘음.")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        (art / "pairs_ff.json").write_text(json.dumps({
            "base": "SDH_functional_full(CombinedMutationTransformer)",
            "ff_dim": ff_dim, "model_parameters": mp,
            "validation": pa.validation_spec(n_splits, model_seed, cv_seeds),
            "stages": {lab: {"f1_macro": ms(lab)[0], "per_seed": per_seed[lab]} for lab, _ in stages},
            "vs_sdh009": {"sdh_cv": 0.4636, "sdh_lb": 0.34238, "our_ff_pair_cv": ffp_m},
            "table": rows,
            "note": "SDH 실제 base 재사용. S 의 가치는 LB(제출)로만 판정.",
            "fingerprint": {**pa.fingerprint(), "features_P_sha256": pa.sha256(fp.__file__)},
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(rows).to_csv(art / "pairs_ff.csv", index=False, encoding="utf-8-sig")
        print("\n저장: artifacts/pairs_ff.json · pairs_ff.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
