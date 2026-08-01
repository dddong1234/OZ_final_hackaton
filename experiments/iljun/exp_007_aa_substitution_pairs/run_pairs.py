"""
아미노산 치환쌍(A_pair) 효과 — 우리 파이프라인에서 재현
=========================================================

    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/run_pairs.py
    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/run_pairs.py --smoke
    .venv/bin/python3 experiments/iljun/exp_007_aa_substitution_pairs/run_pairs.py --tokens   # functional-full 근사 추가

무엇을 답하나
-------------
팀 상위(홍주 biodomain02 LB 0.351, SDH case_06 LB 0.342)의 '이기는 축'인
아미노산 치환 표현을, 우리 base(GBV) 위에서 팀 표준 LR 로 붙여 얼마가 오르나 잰다.

누적 단계 (각 단계가 직전 대비 얼마를 더하나, paired 3-seed):
  GBV                 우리 앵커 (LB 0.2795 로 실측된 base)
  + P_pair            ref→alt 치환쌍 380            ← SDH CV 1위 블록
  + P_pair + P_marg   ref/alt 카운트 40 추가
  + P_pair+marg + S   표기 구조 S 추가              ← 홍주가 얹은 축
  (--tokens 면 functional-full 근사 토큰을 base 에 먼저 깔고 위 단계 반복)

★ 주의 — SDH base 는 'functional full'(전체 토큰), 우리 base 는 GBV 라 절대값은
  SDH 0.464 와 다를 수 있다. 여기서 보는 건 'A_pair 가 우리에게도 크게 오르나'다.
  절대 재현이 필요하면 --tokens 로 base 를 functional-full 에 근접시킨다.

CV↔LB: exp_003 에서 ~63~65% 전달 확인됨. 여기 CV 이득 × ~0.64 가 예상 LB 이득.
fold 학습 분할에서만 fit. test 는 열지 않는다. (P 블록은 고정정의라 fit 자체가 없음)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
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


def macro(y, p):
    return round(float(f1_score(y, p, average="macro")), 5)


def paired(after: dict, before: dict):
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A_pair 효과 — GBV 위 누적")
    ap.add_argument("--root", default=None)
    ap.add_argument("--tokens", action="store_true",
                    help="functional-full 근사 토큰을 base 에 추가(SDH base 근접)")
    ap.add_argument("--tok-min-count", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = load_cfg()
    mp = dict(cfg["model_params"])                    # 팀 표준 C=0.07 max_iter=2000
    cv_seeds = tuple(cfg["cv_seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg.get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["n_splits"]
    name, fn = pa.MODELS[cfg["model"]]
    base_blocks = ("G", "B", "V")

    print("=" * 92)
    print("  A_pair(아미노산 치환쌍) 효과 — GBV 위 누적 (팀 표준 LR)")
    print(f"  {mp}")
    print(f"  cv_seeds {list(cv_seeds)} · KFold-{n_splits} · base {''.join(base_blocks)}"
          + (" + functional-full토큰" if a.tokens else ""))
    print("=" * 92)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    counts, _ = pa.parse_all(train, test, gene_cols)

    # P 블록은 고정정의(fit 없음) → 전체 train 에서 한 번 만들어 fold 인덱스로 슬라이스
    print("P 블록 생성 (pair 380 / marg 40 / S 9)...", flush=True)
    P_pair, _ = fp.build_P(train, gene_cols, "p")
    P_marg, _ = fp.build_P(train, gene_cols, "m")
    P_s, _ = fp.build_P(train, gene_cols, "s")
    print(f"  pair {P_pair.shape[1]} · marg {P_marg.shape[1]} · S {P_s.shape[1]}", flush=True)

    # 선택: functional-full 근사 토큰 (features_D, freq, 전량)
    tokD = None
    if a.tokens:
        import features_D as fd
        print("functional-full 토큰 파싱...", flush=True)
        tokD = fd.parse_token_sets(train, gene_cols)

    stages = [("GBV", None), ("+pair", P_pair),
              ("+pair+marg", sparse.hstack([P_pair, P_marg], format="csr")),
              ("+pair+marg+S", sparse.hstack([P_pair, P_marg, P_s], format="csr"))]

    per_seed = {lab: {} for lab, _ in stages}
    for s in cv_seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        oof = {lab: np.empty(len(y), dtype=object) for lab, _ in stages}
        t0 = time.time()
        for i_tr, i_va in cv.split(train, y):
            spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=model_seed)
            Xg_tr, _ = fa.build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, base_blocks)
            Xg_va, _ = fa.build_features(train.iloc[i_va], counts.iloc[i_va], spec, base_blocks)
            if tokD is not None:                     # functional-full 근사 추가
                st = fd.fit_tokens(tokD, y, i_tr, top_k=10**9,
                                   min_count=a.tok_min_count, method="freq")
                Dt, _ = fd.transform_tokens(tokD, i_tr, st, len(st["order"]))
                Dv, _ = fd.transform_tokens(tokD, i_va, st, len(st["order"]))
                Xg_tr = sparse.hstack([Xg_tr, Dt], format="csr")
                Xg_va = sparse.hstack([Xg_va, Dv], format="csr")
            for lab, extra in stages:
                Xa, Xb = Xg_tr, Xg_va
                if extra is not None:
                    Xa = sparse.hstack([Xg_tr, extra[i_tr]], format="csr")
                    Xb = sparse.hstack([Xg_va, extra[i_va]], format="csr")
                oof[lab][i_va] = fn(model_seed, mp).fit(Xa, y[i_tr]).predict(Xb)
        for lab, _ in stages:
            per_seed[lab][s] = macro(y, np.array(list(oof[lab])))
        print("  cv_seed {}  ".format(s)
              + "  ".join(f"{lab}:{per_seed[lab][s]:.5f}" for lab, _ in stages)
              + f"   ({time.time()-t0:.0f}s)", flush=True)

    def ms(lab):
        v = np.array(list(per_seed[lab].values()), dtype=float)
        return round(v.mean(), 5), (round(v.std(ddof=1), 5) if len(v) > 1 else None)

    print("\n" + "=" * 92)
    print("  누적 (각 단계가 직전 대비 얼마를 더하나) — 팀 표준 LR · 3-seed")
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
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    base_m, _ = ms("GBV")
    best = max(per_seed, key=lambda L: np.mean(list(per_seed[L].values())))
    bm, _ = ms(best)
    dm, dsd, pos, nn = paired(per_seed[best], per_seed["GBV"])
    print(f"\n최고 단계  {best}  {bm:.5f}  (GBV {base_m:.5f} 대비 {dm:+.5f}, {pos}/{nn} seed)")
    print(f"  예상 LB 이득 ~{dm*0.64:+.4f} (패스스루 64% 가정) → GBV LB 0.2795 기준 ~{0.2795+dm*0.64:.4f}")
    print("  ※ SDH functional-full base 가 아니라 GBV base — 절대값 비교보다 '치환쌍이 오르나'가 요점.")
    if not a.tokens:
        print("  ※ SDH 0.34~0.35 에 근접하려면 --tokens (functional-full 근사) 로 재실행.")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        tab.to_csv(art / "pairs.csv", index=False, encoding="utf-8-sig")
        (art / "pairs.json").write_text(json.dumps({
            "base": "".join(base_blocks) + ("+functional_full_tok" if a.tokens else ""),
            "model_parameters": mp,
            "validation": pa.validation_spec(n_splits, model_seed, cv_seeds),
            "tokens": bool(a.tokens), "tok_min_count": a.tok_min_count,
            "stages": {lab: {"f1_macro": ms(lab)[0], "per_seed": per_seed[lab]} for lab, _ in stages},
            "best_stage": best, "best_vs_GBV": {"mean": dm, "std": dsd, "pos": f"{pos}/{nn}"},
            "table": rows,
            "note": "P 블록은 고정정의(leakage 없음). CV 이득의 ~64%가 LB 로 전달(exp_003 실측).",
            "fingerprint": {**pa.fingerprint(), "features_P_sha256": pa.sha256(fp.__file__)},
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n저장: artifacts/pairs.json · pairs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
