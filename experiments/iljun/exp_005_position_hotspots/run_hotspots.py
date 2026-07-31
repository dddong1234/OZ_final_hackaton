"""
코돈 핫스팟(H) vs 정확토큰(D) — 어느 해상도가 GBV 위에서 더 얹히나
====================================================================

    .venv/bin/python3 experiments/iljun/exp_005_position_hotspots/run_hotspots.py
    .venv/bin/python3 experiments/iljun/exp_005_position_hotspots/run_hotspots.py --smoke

----------------------------------------------------------------------
이 실험이 답하는 질문
----------------------------------------------------------------------
같은 base GBV · 같은 fold · 같은 seed 에서 네 가지를 잰다.
  base            GBV (토큰 없음)
  +D:K            GBV + 정확토큰 상위 K (exp_003 의 최고 축)
  +H:K            GBV + 코돈 핫스팟 상위 K (이번 새 축)
  +D+H:K          둘 다

1. 코돈 핫스팟(H)이 GBV 위에 오르나? (paired 증분)
2. **H 가 정확토큰(D)보다 나은가?** 같은 base·K 에서 H − D.
3. H 와 D 를 같이 쓰면 서로 독립적으로 더하나, 겹치나?
4. 각 방법의 배치 아티팩트 플래그 수 — H 가 코돈으로 묶어 더 적어야 정상.

----------------------------------------------------------------------
★ 한계 — 반드시 읽을 것
----------------------------------------------------------------------
배치 아티팩트는 train 안에서 CV 를 올린다(train·val 이 같은 배치라서). 그래서
**여기서 CV 가 올라도 진짜 신호인지 아티팩트인지 CV 로는 판정 못 한다.**
이 실험의 결론은 "제출 후보를 좁힌다"까지다. 최종 판정은 리더보드.

가설: H 는 D 보다 아티팩트가 적으니, CV 이득이 비슷하거나 조금 낮아도 LB
전달률은 H 가 높을 수 있다. 그건 제출 두 장(같은 K 의 D · H)으로만 갈린다.

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
_EXP3 = _HERE.parent / "exp_003_discriminative_tokens"
for p in (_HERE, _EXP2, _EXP3):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline as pa                                                 # noqa: E402
import features_A as fa                                              # noqa: E402
import features_D as fd                                              # noqa: E402
import features_H as fh                                              # noqa: E402
from run_tokens import load_cfg                                      # noqa: E402


def macro(y, p):
    return round(float(f1_score(y, p, average="macro")), 5)


def paired(after: dict, before: dict):
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="코돈 핫스팟(H) vs 정확토큰(D)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--min-count", type=int, default=None,
                    help="재현 임계. 기본은 exp_003 config 값")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = load_cfg()
    blocks = tuple(cfg["blocks"])                    # GBV
    mp = dict(cfg["model_params"])                   # 팀 표준 C=0.07 max_iter=2000
    cv_seeds = tuple(cfg["cv_seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg.get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["n_splits"]
    min_count = a.min_count if a.min_count is not None else cfg["min_count"]
    Ks = [50, 100] if a.smoke else [50, 100, 200]
    maxK = max(Ks)
    name, fn = pa.MODELS[cfg["model"]]

    print("=" * 92)
    print("  코돈 핫스팟(H) vs 정확토큰(D)  —  base GBV 위에 얹기")
    print(f"  {mp}")
    print(f"  cv_seeds {list(cv_seeds)} · model_seed {model_seed} · KFold-{n_splits}")
    print(f"  후보: functional · min_count≥{min_count} · freq · K {Ks}")
    print("  ※ 배치 아티팩트는 CV 를 올린다. 최종 판정은 리더보드로.")
    print("=" * 92)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    counts, _ = pa.parse_all(train, test, gene_cols)

    print("토큰 파싱 (정확변이 D)...", flush=True)
    tokD = fd.parse_token_sets(train, gene_cols)
    print("핫스팟 파싱 (코돈 H)...", flush=True)
    tokH = fh.parse_codon_sets(train, gene_cols)

    labels = ["base"] + [f"D:{k}" for k in Ks] + [f"H:{k}" for k in Ks] \
                      + [f"D+H:{k}" for k in Ks]
    per_seed = {lab: {} for lab in labels}
    flags = {"D": [], "H": []}
    cand_rec = {"D": [], "H": []}

    for s in cv_seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        oof = {lab: np.empty(len(y), dtype=object) for lab in labels}
        t0 = time.time()
        for i_tr, i_va in cv.split(train, y):
            spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=model_seed)
            Xg_tr, _ = fa.build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, blocks)
            Xg_va, _ = fa.build_features(train.iloc[i_va], counts.iloc[i_va], spec, blocks)
            oof["base"][i_va] = fn(model_seed, mp).fit(Xg_tr, y[i_tr]).predict(Xg_va)

            specD = fd.fit_tokens(tokD, y, i_tr, top_k=maxK, min_count=min_count, method="freq")
            specH = fh.fit_codons(tokH, y, i_tr, top_k=maxK, min_count=min_count, method="freq")
            flags["D"].append(specD["diag"]["n_flagged_artifact"])
            flags["H"].append(specH["diag"]["n_flagged_artifact"])
            cand_rec["D"].append(specD["n_candidates"])
            cand_rec["H"].append(specH["n_candidates"])

            for k in Ks:
                Da_tr, _ = fd.transform_tokens(tokD, i_tr, specD, k)
                Da_va, _ = fd.transform_tokens(tokD, i_va, specD, k)
                Ha_tr, _ = fh.transform_codons(tokH, i_tr, specH, k)
                Ha_va, _ = fh.transform_codons(tokH, i_va, specH, k)

                Xd_tr = sparse.hstack([Xg_tr, Da_tr], format="csr")
                Xd_va = sparse.hstack([Xg_va, Da_va], format="csr")
                oof[f"D:{k}"][i_va] = fn(model_seed, mp).fit(Xd_tr, y[i_tr]).predict(Xd_va)

                Xh_tr = sparse.hstack([Xg_tr, Ha_tr], format="csr")
                Xh_va = sparse.hstack([Xg_va, Ha_va], format="csr")
                oof[f"H:{k}"][i_va] = fn(model_seed, mp).fit(Xh_tr, y[i_tr]).predict(Xh_va)

                Xb_tr = sparse.hstack([Xg_tr, Da_tr, Ha_tr], format="csr")
                Xb_va = sparse.hstack([Xg_va, Da_va, Ha_va], format="csr")
                oof[f"D+H:{k}"][i_va] = fn(model_seed, mp).fit(Xb_tr, y[i_tr]).predict(Xb_va)

        for lab in labels:
            per_seed[lab][s] = macro(y, np.array(list(oof[lab])))
        print(f"  cv_seed {s}  base {per_seed['base'][s]:.5f}  "
              f"D:{maxK} {per_seed[f'D:{maxK}'][s]:.5f}  "
              f"H:{maxK} {per_seed[f'H:{maxK}'][s]:.5f}   ({time.time()-t0:.0f}s)", flush=True)

    def ms(lab):
        v = np.array(list(per_seed[lab].values()), dtype=float)
        return round(v.mean(), 5), (round(v.std(ddof=1), 5) if len(v) > 1 else None)

    base_m, base_s = ms("base")
    print("\n" + "=" * 92)
    print(f"  기준선 base(GBV)  {base_m:.5f} ± {base_s}")
    print("=" * 92)
    rows = []
    for grp in ("D", "H", "D+H"):
        for k in Ks:
            lab = f"{grp}:{k}"
            mm, ss = ms(lab)
            dm, dsd, pos, nn = paired(per_seed[lab], per_seed["base"])
            rows.append({"방법": grp, "K": k, "Macro F1": mm, "σ": ss,
                         "base 대비": f"{dm:+.5f}", "양수seed": f"{pos}/{nn}"})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    print("\nH − D  (같은 K, paired) — 코돈이 정확변이보다 나은가")
    hd_rows = []
    for k in Ks:
        dm, dsd, pos, nn = paired(per_seed[f"H:{k}"], per_seed[f"D:{k}"])
        print(f"  K={k:4}   H−D {dm:+.5f} ± {dsd:.5f}   ({pos}/{nn} seed 에서 H 우세)")
        hd_rows.append({"K": k, "H-D": dm, "σ": dsd, "H우세": f"{pos}/{nn}"})

    print("\n배치 아티팩트 플래그 (작은 암종에 100% 몰린 후보 수, fold 평균)")
    for m in ("D", "H"):
        fc = np.mean(flags[m]) if flags[m] else 0
        cc = np.mean(cand_rec[m]) if cand_rec[m] else 0
        print(f"  {m}  상위 {maxK} 중 평균 {fc:.1f} 개 플래그   (후보 풀 평균 {cc:.0f})")

    print("\n읽는 법")
    print("  · H 가 D 만큼 오르면서 아티팩트가 적다 → LB 전달률이 더 좋을 후보")
    print("  · D+H 증분이 D·H 각각의 합보다 작다 → 두 축이 겹친다(같은 신호)")
    print("  · 무엇이든 최종 판정은 같은 K 의 D·H 를 제출로 맞대봐야 한다")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        tab.to_csv(art / "hotspots.csv", index=False, encoding="utf-8-sig")
        (art / "hotspots.json").write_text(json.dumps({
            "base_blocks": "".join(blocks), "model_parameters": mp,
            "validation": pa.validation_spec(n_splits, model_seed, cv_seeds),
            "min_count": min_count, "K": Ks,
            "base": {"f1_macro": base_m, "f1_macro_std": base_s, "per_seed": per_seed["base"]},
            "results": {lab: {"f1_macro": ms(lab)[0], "per_seed": per_seed[lab]}
                        for lab in labels if lab != "base"},
            "H_minus_D": hd_rows,
            "artifact_flags_mean": {m: float(np.mean(flags[m])) for m in ("D", "H")},
            "candidate_pool_mean": {m: float(np.mean(cand_rec[m])) for m in ("D", "H")},
            "table": rows,
            "caveat": "배치 아티팩트가 CV 를 올린다. H·D 순전달은 리더보드로만 판정.",
            "fingerprint": {**pa.fingerprint(),
                            "features_D_sha256": pa.sha256(fd.__file__),
                            "features_H_sha256": pa.sha256(fh.__file__)},
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n저장: artifacts/hotspots.json · hotspots.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
