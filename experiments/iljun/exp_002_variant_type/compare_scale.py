"""
스케일 처리 대조 — 경수님 결과와 우리 결과가 갈리는 원인 좁히기
================================================================

    python3 experiments/iljun/exp_002_variant_type/compare_scale.py
    python3 experiments/iljun/exp_002_variant_type/compare_scale.py --smoke   # 1분

----------------------------------------------------------------------
왜 이 실험을 하나
----------------------------------------------------------------------
경수님 【EDA 및 전처리 실험】 결론:

    member_b_binary_plus_stats   baseline 대비 큰 폭 하락
    member_b_binary_type_stats   baseline 대비 큰 폭 하락
    → sample/gene statistics, mutation type count 는 채택하지 않음

우리 exp_002 측정 (cv_seeds 42/52/62, paired):

    B  변이 부담   +0.02608 ± 0.00441  (3/3 seed 상승)
    V  유형 카운트 +0.01388 ± 0.00536  (3/3 seed 상승)

**정반대다.** 그런데 두 사람 baseline 은 같다.

    경수님 team_baseline_binary = 0.344689
    우리   G @ cv_seed 42       = 0.34469

5자리까지 일치하므로 데이터·CV 프로토콜·모델 차이는 배제된다.
남는 것은 **같은 이름의 피처를 서로 다르게 만들었다**는 가능성이다.

----------------------------------------------------------------------
이 스크립트가 가르는 변수: log1p 여부
----------------------------------------------------------------------
우리 구현은 모든 집계에 log1p 를 씌운다. TMB 는 중앙값 14 · 최대 2393 이라
raw 로 넣으면 0/1 이진 열 4,226개와 스케일이 100배 이상 벌어진다.
정규화가 없는 LogisticRegression(lbfgs) 은 여기에 민감하다.

비교 구성 (전부 같은 cv_seeds 로 paired):

    G        유전자 이진화만                       ← 공통 기준
    G+B      + 부담 log1p
    G+b      + 부담 raw              ← B 와 이것의 차이가 곧 log1p 효과
    G+B+V    + 부담·유형 둘 다 log1p
    G+b+v    + 부담·유형 둘 다 raw    ← 경수님 구성에 더 가까울 것으로 추정

읽는 법 — raw 구성이 G 근처이거나 그 아래로 떨어지고 log1p 구성만 오른다면,
두 사람 결과가 갈린 원인은 **스케일 처리**로 좁혀진다.
반대로 raw 도 비슷하게 오른다면 원인은 다른 곳(집계 정의 범위, 파싱 규칙 등)이다.

⚠️ 이 스크립트는 어느 쪽이 맞는지 정하지 않는다. 변수 하나를 가를 뿐이다.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pipeline as pa                                                # noqa: E402
import features_A as fa                                              # noqa: E402

CONFIGS = [
    ("G",    "G       유전자 이진화만 (공통 기준)"),
    ("GB",   "G+B     + 부담 log1p"),
    ("Gb",   "G+b     + 부담 raw"),
    ("GBV",  "G+B+V   + 부담·유형 log1p"),
    ("Gbv",  "G+b+v   + 부담·유형 raw"),
]


def paired_delta(after: dict, before: dict):
    """같은 cv_seed 끼리 짝지어 뺀 증분. (평균, 표준편차, 양수 개수, 원본)"""
    a = {d["cv_seed"]: d["f1_macro"] for d in after["per_seed"]}
    b = {d["cv_seed"]: d["f1_macro"] for d in before["per_seed"]}
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return float(d.mean()), sd, int((d > 0).sum()), d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="log1p 여부 대조")
    ap.add_argument("--root", default=None)
    ap.add_argument("--smoke", action="store_true", help="클래스당 12행 · 2fold")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = pa.load_cfg()["pipeline"]
    cv_seeds = tuple(cfg["cv"]["seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg["cv"].get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["cv"]["n_splits"]

    print("=" * 78)
    print("  스케일 처리 대조 — log1p vs raw")
    print(f"  features_A {fa.__version__} · cv_seeds {list(cv_seeds)} · "
          f"model_seed {model_seed} · StratifiedKFold-{n_splits}")
    print("  경수님 team_baseline_binary = 0.344689 (단일 seed 42 기준)")
    print("=" * 78)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    cnt_train, _ = pa.parse_all(train, test, gene_cols)

    n_fits = len(cv_seeds) * n_splits
    res, conv = {}, {}
    for blocks, label in CONFIGS:
        # lbfgs 수렴 실패 횟수를 센다 — 스케일 가설의 직접 증거가 된다
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            r = pa.cross_validate_multi(train, y, cnt_train, gene_cols, tuple(blocks),
                                        model_key="logreg", cv_seeds=cv_seeds,
                                        model_seed=model_seed, n_splits=n_splits, v=False)
        conv[blocks] = sum(1 for x in caught
                           if issubclass(x.category, ConvergenceWarning))
        res[blocks] = r
        per = "  ".join(f"{d['f1_macro']:.5f}" for d in r["per_seed"])
        std = f"± {r['f1_macro_std']:.5f}" if r["f1_macro_std"] is not None else ""
        flag = f"  ⚠ 미수렴 {conv[blocks]}/{n_fits}" if conv[blocks] else ""
        print(f"{label:34} dim {r['dim']:5d}  F1 {r['f1_macro']:.5f} {std}   "
              f"[seed별 {per}]{flag}")

    print()
    print(f"lbfgs 수렴 실패 (max_iter={pa.DEFAULT_MODEL_PARAMS['logreg']['max_iter']}, "
          f"fit {n_fits}회 기준)")
    for blocks, label in CONFIGS:
        print(f"  {label.split()[0]:8} {conv[blocks]}/{n_fits}")
    print("  → raw 쪽에서만 미수렴이 난다면, 두 구현의 차이는 스케일 처리로 좁혀진다.")
    print("    (미수렴 모델의 점수는 최적해가 아니므로 그 자체로 신뢰도가 낮다)")

    # ── 공통 기준 G 대비 paired 증분 ────────────────────────────────
    print()
    print("G 대비 paired 증분")
    rows = []
    for blocks, label in CONFIGS[1:]:
        m, sd, pos, d = paired_delta(res[blocks], res["G"])
        rows.append({"구성": label.split()[0], "설명": " ".join(label.split()[1:]),
                     "평균 증분": round(m, 5), "증분 σ": round(sd, 5),
                     "양수 seed": f"{pos}/{len(d)}"})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    # ── log1p 효과 = (log1p 구성) − (raw 구성) ──────────────────────
    print()
    print("log1p 효과 (같은 블록 구성에서 log1p − raw)")
    eff = []
    for name, hi, lo in [("부담만", "GB", "Gb"), ("부담+유형", "GBV", "Gbv")]:
        m, sd, pos, d = paired_delta(res[hi], res[lo])
        eff.append({"비교": f"{name}  {hi} − {lo}", "평균 차이": round(m, 5),
                    "차이 σ": round(sd, 5), "양수 seed": f"{pos}/{len(d)}"})
    eff_tab = pd.DataFrame(eff)
    print(eff_tab.to_string(index=False))

    print()
    print("※ n=3 의 σ 는 그 자체로 부정확하다. 방향과 대략적 크기만 읽는다.")
    print("※ 이 결과는 '어느 구현이 옳은가'가 아니라 'log1p 가 원인인가'만 가른다.")

    if not a.smoke:
        art = root / "experiments" / "iljun" / "exp_002_variant_type" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        tab.to_csv(art / "scale_check_vs_G.csv", index=False, encoding="utf-8-sig")
        eff_tab.to_csv(art / "scale_check_log1p_effect.csv", index=False, encoding="utf-8-sig")
        summary = {
            "features_version": fa.__version__,
            "features_sha256": pa.sha256(fa.__file__),
            "cv_seeds": [int(s) for s in cv_seeds], "model_seed": int(model_seed),
            "n_splits": int(n_splits),
            "configs": {b: {"f1_macro": res[b]["f1_macro"],
                            "f1_macro_std": res[b]["f1_macro_std"],
                            "dim": res[b]["dim"],
                            "per_seed": res[b]["per_seed"]} for b, _ in CONFIGS},
            "vs_G": rows, "log1p_effect": eff,
            "lbfgs_nonconvergence": {b: f"{conv[b]}/{n_fits}" for b, _ in CONFIGS},
            "note": "경수님 결과와의 충돌 원인 좁히기. log1p 여부만 가른다.",
        }
        (art / "scale_check.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: artifacts/scale_check.json · scale_check_vs_G.csv · "
              f"scale_check_log1p_effect.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
