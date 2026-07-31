"""
판별력 vs 빈도 토큰 — 어느 선택 기준이 나은가
=============================================

    .venv/bin/python3 experiments/member_d/exp_003_discriminative_tokens/run_tokens.py
    .venv/bin/python3 experiments/member_d/exp_003_discriminative_tokens/run_tokens.py --smoke

----------------------------------------------------------------------
이 실험이 답하는 질문
----------------------------------------------------------------------
1. GBV(현재 최고, CV 0.412) 위에 변이 토큰을 얹으면 CV 가 오르나?
2. **판별력(lift) 선택이 빈도(freq) 선택보다 나은가?**
   같은 후보 풀에서 순위 기준만 다르게 해 K 개씩 고른다. 선택 기준만의 차이.
3. 판별력이 고른 토큰 중 배치 아티팩트로 의심되는 게 몇 개인가?

----------------------------------------------------------------------
★ 이 실험의 한계 — 반드시 읽을 것
----------------------------------------------------------------------
배치 아티팩트는 train 안에서 CV 를 올린다. train fold 와 val fold 가 같은
배치라 아티팩트가 양쪽에 다 있기 때문이다. 그래서 **여기서 CV 가 올라도
그게 진짜 신호인지 아티팩트인지 CV 로는 판정할 수 없다.**

이 실험의 결론은 "제출 후보를 좁힌다"까지다. 최종 판정은 리더보드로 한다.
disc 가 freq 보다 CV 는 높은데 아티팩트 플래그가 많다면, 그건 경고 신호다.

fold 학습 분할에서만 토큰을 고른다. test 는 열지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
for p in (_HERE, _EXP2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline as pa                                                # noqa: E402
import features_A as fa                                              # noqa: E402
import features_D as fd                                              # noqa: E402


def load_cfg():
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))["exp003"]


def macro(y, p):
    return round(float(f1_score(y, p, average="macro")), 5)


def paired(after: dict, before: dict):
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="판별력 vs 빈도 토큰")
    ap.add_argument("--root", default=None)
    ap.add_argument("--min-count", type=int, default=None,
                    help="재현 임계. 낮출수록 후보 풀이 커져 freq·disc 가 갈린다. "
                         "기본은 config.yaml 값")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = load_cfg()
    blocks = tuple(cfg["blocks"])
    mp = dict(cfg["model_params"])
    cv_seeds = tuple(cfg["cv_seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg.get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["n_splits"]
    min_count = a.min_count if a.min_count is not None else cfg["min_count"]
    Ks = [25, 50] if a.smoke else list(cfg["top_k_list"])
    maxK = max(Ks)
    methods = ["freq", "disc"]

    print("=" * 90)
    print("  판별력 vs 빈도 토큰  (D 블록을 GBV 위에 얹는다)")
    print(f"  base {''.join(blocks)} · {mp} · features_D {fd.__version__}")
    print(f"  후보: functional · min_count≥{min_count} · K {Ks}")
    print(f"  cv_seeds {list(cv_seeds)} · model_seed {model_seed} · KFold-{n_splits}")
    print("  ※ 배치 아티팩트는 CV 를 올린다. 최종 판정은 리더보드로.")
    print("=" * 90)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    counts, _ = pa.parse_all(train, test, gene_cols)
    name, fn = pa.MODELS[cfg["model"]]

    print("토큰 파싱 중...", flush=True)
    t0 = time.time()
    token_sets = fd.parse_token_sets(train, gene_cols)
    print(f"  functional 토큰 집합 완성 ({time.time() - t0:.0f}s)", flush=True)

    # 결과 누적: oof[(method,K)] 와 oof_base, seed 별 F1
    labels = ["base"] + [f"{m}:{k}" for m in methods for k in Ks]
    per_seed = {lab: {} for lab in labels}
    flag_counts = {m: [] for m in methods}          # disc/freq 아티팩트 플래그 수
    overlap_rec = []                                 # freq∩disc 겹침

    for s in cv_seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        oof = {lab: np.empty(len(y), dtype=object) for lab in labels}
        for i_tr, i_va in cv.split(train, y):
            # ── GBV (fold-train fit) ─────────────────────────────────
            spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=model_seed)
            Xg_tr, _ = fa.build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, blocks)
            Xg_va, _ = fa.build_features(train.iloc[i_va], counts.iloc[i_va], spec, blocks)

            oof["base"][i_va] = fn(model_seed, mp).fit(Xg_tr, y[i_tr]).predict(Xg_va)

            # ── 토큰 (fold-train fit, y 사용) ────────────────────────
            specs = {m: fd.fit_tokens(token_sets, y, i_tr, top_k=maxK,
                                      min_count=min_count, method=m) for m in methods}
            for m in methods:
                flag_counts[m].append(specs[m]["diag"]["n_flagged_artifact"])
            # freq∩disc 겹침(최대 K 기준)
            fset = set(specs["freq"]["order"]); dset = set(specs["disc"]["order"])
            if fset and dset:
                overlap_rec.append(len(fset & dset) / len(fset | dset))

            for m in methods:
                for k in Ks:
                    Dt, _ = fd.transform_tokens(token_sets, i_tr, specs[m], k)
                    Dv, _ = fd.transform_tokens(token_sets, i_va, specs[m], k)
                    Xtr = sparse.hstack([Xg_tr, Dt], format="csr")
                    Xva = sparse.hstack([Xg_va, Dv], format="csr")
                    oof[f"{m}:{k}"][i_va] = fn(model_seed, mp).fit(Xtr, y[i_tr]).predict(Xva)

        for lab in labels:
            per_seed[lab][s] = macro(y, np.array(list(oof[lab])))
        line = "  ".join(f"{lab} {per_seed[lab][s]:.5f}" for lab in ["base"] +
                         [f"disc:{maxK}", f"freq:{maxK}"])
        print(f"  cv_seed {s}  {line}", flush=True)

    # ── 표 ────────────────────────────────────────────────────────────
    def mean_std(lab):
        v = np.array(list(per_seed[lab].values()), dtype=float)
        return round(v.mean(), 5), (round(v.std(ddof=1), 5) if len(v) > 1 else None)

    base_m, base_s = mean_std("base")
    print("\n" + "=" * 90)
    print(f"  기준선 base({''.join(blocks)})  {base_m:.5f} ± {base_s}")
    print("=" * 90)
    rows = []
    for m in methods:
        for k in Ks:
            lab = f"{m}:{k}"
            mm, ss = mean_std(lab)
            dm, dsd, pos, nn = paired(per_seed[lab], per_seed["base"])
            rows.append({"방법": m, "K": k, "Macro F1": mm, "σ": ss,
                         "base 대비": dm, "증분σ": dsd, "양수seed": f"{pos}/{nn}"})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    # freq vs disc 직접 비교 (같은 K)
    print("\n판별력 − 빈도  (같은 K, paired)")
    for k in Ks:
        dm, dsd, pos, nn = paired(per_seed[f"disc:{k}"], per_seed[f"freq:{k}"])
        print(f"  K={k:4}   disc−freq {dm:+.5f} ± {dsd:.5f}   ({pos}/{nn} seed 에서 disc 우세)")

    # 아티팩트 진단
    print("\n배치 아티팩트 진단 (작은 암종에 100% 몰린 토큰 수, fold 평균)")
    for m in methods:
        fc = np.mean(flag_counts[m]) if flag_counts[m] else 0
        print(f"  {m:5}  상위 {maxK} 중 평균 {fc:.1f} 개 플래그")
    if overlap_rec:
        print(f"\n  freq 와 disc 선택의 겹침 (Jaccard, 상위 {maxK}) 평균 {np.mean(overlap_rec):.2f}")
        print("  겹침이 크면 두 방법이 사실상 같은 토큰을 고르는 것 → 구분 의미 적음")

    print("\n읽는 법")
    print("  · disc 가 freq 보다 CV 높은데 아티팩트 플래그도 많다 → CV 상승이 아티팩트일 수 있다")
    print("  · 어느 쪽이든 base 대비 오르면 제출로 확인한다 (CV 만으로는 진위 판정 불가)")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        stem = f"token_compare_mc{min_count}"
        tab.to_csv(art / f"{stem}.csv", index=False, encoding="utf-8-sig")
        (art / f"{stem}.json").write_text(json.dumps({
            "base_blocks": "".join(blocks), "model_parameters": mp,
            "validation": pa.validation_spec(n_splits, model_seed, cv_seeds),
            "min_count": min_count, "top_k_list": Ks,
            "base": {"f1_macro": base_m, "f1_macro_std": base_s, "per_seed": per_seed["base"]},
            "results": {lab: {"per_seed": per_seed[lab]} for lab in labels if lab != "base"},
            "table": rows,
            "artifact_flags_mean": {m: float(np.mean(flag_counts[m])) for m in methods},
            "freq_disc_jaccard_mean": float(np.mean(overlap_rec)) if overlap_rec else None,
            "caveat": "배치 아티팩트가 CV 를 올린다. 최종 판정은 리더보드 제출로.",
            "fingerprint": {**pa.fingerprint(),
                            "features_D": f"features_D.py@{fd.__version__}",
                            "features_D_sha256": pa.sha256(fd.__file__)},
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n저장: artifacts/{stem}.json · {stem}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
