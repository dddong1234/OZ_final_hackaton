"""
수렴 실험 — 검증된 블록을 C=0.07 에서 쌓아 올린다
=================================================

    .venv/bin/python3 experiments/iljun/exp_003_discriminative_tokens/converge.py
    .venv/bin/python3 experiments/iljun/exp_003_discriminative_tokens/converge.py --smoke

----------------------------------------------------------------------
목적 (팀 방침: 새 FE 는 멈추고, 된 것들을 조합)
----------------------------------------------------------------------
지금까지 '된다'가 확인된 내 블록만 모은다.

    G  유전자 이진화
    B  변이 부담 3 (log1p)
    V  변이 유형 카운트 6 (log1p)
    D  빈도 기반 변이 토큰 (freq, 판별력에 이김)

기각된 것(R 비율·판별력 토큰·계층형·확률보정·log1p대조)은 넣지 않는다.
모델은 팀 표준 LR(max_iter=2000, C=0.07)로 고정한다.

----------------------------------------------------------------------
무엇을 답하나
----------------------------------------------------------------------
1. 누적 쌓기 — G → +B → +V → +D 로 갈 때 각 단계가 얼마를 더하나 (paired 증분)
   = 각 블록의 '앞의 것들을 감안한' 한계 기여. 발표의 핵심 표.
2. 토큰 개수 K — 몇 개가 최적인가 (C=0.07 에서 다시)
3. 최종 조합 확정 — 제출 후보

★ 여전히 CV 만으로는 배치 아티팩트를 못 가른다. 토큰 단계의 증분은 LB 로
  확인해야 진짜다. (freq 토큰 100 안에도 아티팩트 의심 7개가 있다.)

fold 학습 분할에서만 fit 한다. test 는 열지 않는다.
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
for p in (_HERE, _EXP2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline as pa                                                # noqa: E402
import features_A as fa                                              # noqa: E402
import features_D as fd                                              # noqa: E402
from run_tokens import load_cfg                                      # noqa: E402


def macro(y, p):
    return round(float(f1_score(y, p, average="macro")), 5)


def paired(after: dict, before: dict):
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="수렴 실험 — 블록 누적 · C=0.07")
    ap.add_argument("--root", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = load_cfg()
    mp = dict(cfg["model_params"])                       # 팀 표준 C=0.07
    cv_seeds = tuple(cfg["cv_seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg.get("model_seed", 42)
    n_splits = 2 if a.smoke else cfg["n_splits"]
    min_count = cfg["min_count"]
    Ks = [50, 100] if a.smoke else [50, 100, 200]
    maxK = max(Ks)
    name, fn = pa.MODELS[cfg["model"]]

    # 누적 단계: (라벨, 블록문자, 토큰K)
    stages = [("G", "G", 0), ("G+B", "GB", 0), ("G+B+V", "GBV", 0)]
    stages += [(f"G+B+V+tok{k}", "GBV", k) for k in Ks]

    print("=" * 84)
    print("  수렴 실험 — 검증 블록 누적 (팀 표준 LR)")
    print(f"  {mp}")
    print(f"  cv_seeds {list(cv_seeds)} · model_seed {model_seed} · KFold-{n_splits}")
    print(f"  토큰: freq · functional · min_count≥{min_count} · K {Ks}")
    print("=" * 84)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    counts, _ = pa.parse_all(train, test, gene_cols)
    print("토큰 파싱...", flush=True)
    tok = fd.parse_token_sets(train, gene_cols)

    per_seed = {lab: {} for lab, _, _ in stages}
    flag_hist = []
    for s in cv_seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        oof = {lab: np.empty(len(y), dtype=object) for lab, _, _ in stages}
        t0 = time.time()
        for i_tr, i_va in cv.split(train, y):
            spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=model_seed)
            # 블록별 GBV 부분행렬은 필요한 최대(GBV)만 만들고 재사용
            feats = {}
            for blk in ("G", "GB", "GBV"):
                Xa, _ = fa.build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, tuple(blk))
                Xb, _ = fa.build_features(train.iloc[i_va], counts.iloc[i_va], spec, tuple(blk))
                feats[blk] = (Xa, Xb)
            # 토큰 spec (fold-train, freq)
            st = fd.fit_tokens(tok, y, i_tr, top_k=maxK, min_count=min_count, method="freq")
            flag_hist.append(st["diag"]["n_flagged_artifact"])

            for lab, blk, k in stages:
                Xa, Xb = feats[blk]
                if k > 0:
                    Da, _ = fd.transform_tokens(tok, i_tr, st, k)
                    Db, _ = fd.transform_tokens(tok, i_va, st, k)
                    Xa = sparse.hstack([Xa, Da], format="csr")
                    Xb = sparse.hstack([Xb, Db], format="csr")
                oof[lab][i_va] = fn(model_seed, mp).fit(Xa, y[i_tr]).predict(Xb)
        for lab, _, _ in stages:
            per_seed[lab][s] = macro(y, np.array(list(oof[lab])))
        best_here = max(per_seed, key=lambda L: per_seed[L].get(s, 0))
        print(f"  cv_seed {s}  "
              + "  ".join(f"{lab.split('+')[-1]}:{per_seed[lab][s]:.4f}"
                          for lab, _, _ in stages)
              + f"   ({time.time() - t0:.0f}s)", flush=True)

    # ── 누적 표 (각 단계의 이전 대비 증분) ────────────────────────────
    def ms(lab):
        v = np.array(list(per_seed[lab].values()), dtype=float)
        return round(v.mean(), 5), (round(v.std(ddof=1), 5) if len(v) > 1 else None)

    print("\n" + "=" * 84)
    print("  누적 쌓기 (C=0.07) — 각 단계가 이전 대비 얼마를 더하나")
    print("=" * 84)
    rows, prev = [], None
    for lab, _, _ in stages:
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

    # 최종 후보 = 평균 최고
    best = max(per_seed, key=lambda L: np.mean(list(per_seed[L].values())))
    bm, bsd = ms(best)
    base_m, _ = ms("G+B+V")
    dm, dsd, pos, nn = paired(per_seed[best], per_seed["G+B+V"])
    print(f"\n최종 후보  {best}   {bm:.5f} ± {bsd}")
    print(f"  GBV(토큰 전) {base_m:.5f} 대비 {dm:+.5f} ({pos}/{nn} seed)")
    if flag_hist:
        print(f"  ⚠ 토큰 상위 {maxK} 중 아티팩트 의심 평균 {np.mean(flag_hist):.1f}개 "
              f"— 이 증분이 진짜인지는 LB 로만 확인된다")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        tab.to_csv(art / "converge.csv", index=False, encoding="utf-8-sig")
        (art / "converge.json").write_text(json.dumps({
            "model_parameters": mp,
            "validation": pa.validation_spec(n_splits, model_seed, cv_seeds),
            "min_count": min_count, "token_K": Ks,
            "stages": {lab: {"f1_macro": ms(lab)[0], "f1_macro_std": ms(lab)[1],
                             "per_seed": per_seed[lab]} for lab, _, _ in stages},
            "best_stage": best,
            "artifact_flags_mean": float(np.mean(flag_hist)) if flag_hist else None,
            "table": rows,
            "caveat": "토큰 단계 증분은 배치 아티팩트를 포함할 수 있다. LB 로 확인.",
            "fingerprint": {**pa.fingerprint(),
                            "features_D_sha256": pa.sha256(fd.__file__)},
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n저장: artifacts/converge.json · converge.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
