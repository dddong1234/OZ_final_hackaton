"""
규제 C 세부 탐색 — 팀 공용
==========================

    .venv/bin/python3 experiments/member_d/exp_002_variant_type/tune_C.py            # 정본 블록
    .venv/bin/python3 experiments/member_d/exp_002_variant_type/tune_C.py --quick    # 5개만 (빠름)
    .venv/bin/python3 experiments/member_d/exp_002_variant_type/tune_C.py --blocks G # 팀 공용 baseline 조건

----------------------------------------------------------------------
왜 지금 이걸 하나 — 팀 전체가 C=1.0 으로 돌고 있다
----------------------------------------------------------------------
경수님 exp_003 문서에 이렇게 적혀 있다.

    "현재 공용 Logistic Regression 설정에는 C 가 명시되어 있지 않으므로
     sklearn 기본값 C=1.0 이 적용된다."

그런데 제 3-seed 측정에서 **C 하나만 0.1 로 바꿔도 모든 블록에서 +0.035~+0.044**
(3/3 seed 양수)가 나왔다. 피처 15개를 새로 만든 것보다 큰 효과다.

    G      C=1.0 → C=0.1    0.33717 → 0.38145   (+0.04428)
    G+B    C=1.0 → C=0.1    0.36325 → 0.40344   (+0.04019)
    G+B+V  C=1.0 → C=0.1    0.37713 → 0.41202   (+0.03489)

즉 팀의 모든 전처리 비교표가 **규제가 안 걸린 상태에서 매겨진 순위**다.
C 를 바꾸면 순위가 뒤바뀔 수도 있다.

그런데 저도 **C 를 1.0 과 0.1 두 값만 봤다.** 0.1 이 최적이라는 근거는 없다.
이 스크립트가 그 빈칸을 채운다.

----------------------------------------------------------------------
왜 규제가 이렇게 큰가 (초보자용 설명)
----------------------------------------------------------------------
LogisticRegression 은 유전자마다 암종별 점수(가중치)를 학습한다.

    유전자 4,226개 × 암종 26개 = 약 11만 개의 숫자를 환자 6,201명에게서 배운다

배울 게 데이터보다 훨씬 많다. 이러면 모델이 **우연히 그 환자들에게만 맞는 규칙**을
외워버린다 (과적합). C 는 그 숫자들을 얼마나 크게 허용할지를 정한다.

    C 가 크다 (1.0)  → 숫자를 마음껏 키움 → 외우기 쉬움 → 새 환자에게 약함
    C 가 작다 (0.1)  → 숫자를 작게 누름   → 큰 흐름만 배움 → 새 환자에게 강함

너무 작으면 아무것도 못 배운다. 그래서 **적당한 지점**이 있고, 그걸 찾는 게 이 실험이다.

----------------------------------------------------------------------
결과를 읽을 때 주의 (중요)
----------------------------------------------------------------------
후보 10개 중 **최댓값을 고르면 그 값 자체는 부풀려져 있다.** 10번 던져 가장 잘 나온
것을 고른 셈이라, 운 좋게 높게 나온 값을 집는 경향이 있다.

그래서 이 스크립트는 최댓값 하나가 아니라 **σ 안에 들어오는 평평한 구간**을 같이
보여준다. 평평하면 그 구간의 가운데를 고르는 게 안전하다. 그게 seed 가 바뀌어도
덜 흔들린다.

train 만 사용한다. test 는 열지 않는다.
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

# C 만 변수. 나머지는 팀 전처리 비교 조건과 동일하게 고정한다.
BASE_PARAMS = {"solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced"}

FULL_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
QUICK_GRID = [0.03, 0.05, 0.1, 0.2, 0.5]
ANCHOR = 0.1          # 지금 정본. 이 값 대비 paired 증분을 본다.


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="규제 C 세부 탐색")
    ap.add_argument("--root", default=None)
    ap.add_argument("--blocks", default=None,
                    help="기본은 config.yaml 의 blocks. 팀 공용 baseline 조건은 G")
    ap.add_argument("--quick", action="store_true", help="C 5개만 (대략 절반 시간)")
    ap.add_argument("--smoke", action="store_true", help="2 seed × 2 fold 배선 점검")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = pa.load_cfg()["pipeline"]
    blocks = tuple(a.blocks or cfg["blocks"])
    cv_seeds = tuple(cfg["cv"]["seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg["cv"].get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["cv"]["n_splits"]
    grid = ([0.1, 1.0] if a.smoke else (QUICK_GRID if a.quick else FULL_GRID))
    n_fits = len(cv_seeds) * n_splits

    print("=" * 88)
    print("  규제 C 세부 탐색")
    print(f"  피처 {''.join(blocks)} · features_A {fa.__version__}")
    print(f"  고정: solver=lbfgs · max_iter={BASE_PARAMS['max_iter']} · class_weight=balanced")
    print(f"  변수: C {grid}")
    print(f"  cv_seeds {list(cv_seeds)} · model_seed {model_seed} · StratifiedKFold-{n_splits}")
    print(f"  총 {len(grid) * n_fits} 회 학습")
    print("=" * 88)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    cnt_train, _ = pa.parse_all(train, test, gene_cols)

    res, conv = {}, {}
    for C in grid:
        mp = dict(BASE_PARAMS, C=C)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            r = pa.cross_validate_multi(train, y, cnt_train, gene_cols, blocks,
                                        model_key="logreg", model_params=mp,
                                        cv_seeds=cv_seeds, model_seed=model_seed,
                                        n_splits=n_splits, v=False)
        conv[C] = sum(1 for x in caught if issubclass(x.category, ConvergenceWarning))
        res[C] = r
        per = "  ".join(f"{d['f1_macro']:.5f}" for d in r["per_seed"])
        std = f"± {r['f1_macro_std']:.5f}" if r["f1_macro_std"] is not None else ""
        flag = f"  ⚠미수렴 {conv[C]}/{n_fits}" if conv[C] else ""
        print(f"  C = {C:<5}  F1 {r['f1_macro']:.5f} {std}   Acc {r['accuracy']:.5f}"
              f"   [seed별 {per}]{flag}", flush=True)

    # ── 표 ────────────────────────────────────────────────────────────
    rows = []
    anchor = res.get(ANCHOR)
    for C in grid:
        r = res[C]
        row = {"C": C, "Macro F1": round(r["f1_macro"], 5),
               "σ": round(r["f1_macro_std"], 5) if r["f1_macro_std"] is not None else None,
               "Accuracy": round(r["accuracy"], 5),
               "미수렴": f"{conv[C]}/{n_fits}"}
        if anchor is not None and C != ANCHOR:
            av = {d["cv_seed"]: d["f1_macro"] for d in anchor["per_seed"]}
            bv = {d["cv_seed"]: d["f1_macro"] for d in r["per_seed"]}
            seeds = sorted(set(av) & set(bv))
            d = np.array([bv[s] - av[s] for s in seeds], dtype=float)
            row[f"C={ANCHOR} 대비"] = round(float(d.mean()), 5)
            row["양수 seed"] = f"{int((d > 0).sum())}/{len(d)}"
        rows.append(row)
    tab = pd.DataFrame(rows)

    print("\n" + "=" * 88)
    print("  전체 결과 (C 오름차순)")
    print("=" * 88)
    print(tab.to_string(index=False))

    # ── 최댓값과 '평평한 구간' ────────────────────────────────────────
    best_C = max(grid, key=lambda c: res[c]["f1_macro"])
    best = res[best_C]
    own_sigma = best["f1_macro_std"] or 0.0
    # ★ 최댓값 지점의 σ 를 잣대로 쓰면 안 된다. n=3 의 σ 는 매우 부정확해서
    #   우연히 작게 나온 지점이 뽑히면 '평평한 구간'이 자기 혼자만 남는다.
    #   그리드 전체 σ 의 중앙값을 공통 잣대로 쓴다.
    sigmas = [res[c]["f1_macro_std"] for c in grid if res[c]["f1_macro_std"] is not None]
    sigma = float(np.median(sigmas)) if sigmas else own_sigma
    flat = [c for c in grid if res[c]["f1_macro"] >= best["f1_macro"] - sigma]

    print(f"\n최댓값        C = {best_C}   F1 {best['f1_macro']:.5f} ± {own_sigma:.5f}")
    print(f"평평한 구간   C ∈ {flat}")
    print(f"              (최댓값에서 σ={sigma:.5f} 이내로 사실상 구분이 안 되는 값들)")
    print(f"              ※ 잣대 σ 는 그리드 전체 σ 의 중앙값이다. 최댓값 지점의")
    print(f"                σ({own_sigma:.5f})를 쓰면 그 값이 우연히 작을 때 구간이 사라진다.")
    if len(flat) > 1:
        # C 는 로그 눈금으로 움직이는 값이라 산술 중앙이 아니라 기하 중앙을 쓴다.
        # (0.03 과 0.3 의 가운데는 0.165 가 아니라 0.095 다)
        gmean = float(np.exp(np.mean(np.log(flat))))
        mid = min(flat, key=lambda c: abs(np.log(c) - np.log(gmean)))
        print(f"\n권장          C = {mid}   (평평한 구간의 기하 중앙 {gmean:.4f} 에 가장 가까운 값)")
        print("              최댓값 하나를 집으면 그 값은 부풀려져 있다. 구분이 안 되는")
        print("              구간에서는 가운데를 고르는 편이 seed 변화에 덜 흔들린다.")
    else:
        print(f"\n권장          C = {best_C}  (σ 안에 다른 후보가 없어 뾰족한 최적점)")

    if ANCHOR in res:
        d = best["f1_macro"] - res[ANCHOR]["f1_macro"]
        print(f"\n현재 정본 C={ANCHOR} ({res[ANCHOR]['f1_macro']:.5f}) 대비 최댓값 차이 {d:+.5f}")
        if abs(d) < sigma:
            print("  → 차이가 σ 안이다. **C=0.1 을 바꿀 근거가 없다.**")
        else:
            print("  → 차이가 σ 를 넘는다. 3-seed 재확인 후 정본 교체를 검토한다.")

    print("\n주의 — 후보 여러 개 중 최댓값을 고른 것이므로 그 값 자체에는 낙관 편향이 있다.")
    print("      실제 리더보드 점수는 이 CV 값보다 낮게 나올 것으로 본다.")

    # ── 저장 ──────────────────────────────────────────────────────────
    if not a.smoke:
        art = root / "experiments" / "member_d" / "exp_002_variant_type" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        stem = f"tune_C_{''.join(blocks)}"
        tab.to_csv(art / f"{stem}.csv", index=False, encoding="utf-8-sig")
        (art / f"{stem}.json").write_text(json.dumps({
            "blocks": "".join(blocks), "features_version": fa.__version__,
            "fixed_params": BASE_PARAMS, "C_grid": grid,
            "validation": pa.validation_spec(n_splits, model_seed, cv_seeds),
            "results": {str(C): {"f1_macro": res[C]["f1_macro"],
                                 "f1_macro_std": res[C]["f1_macro_std"],
                                 "accuracy": res[C]["accuracy"],
                                 "dim": res[C]["dim"],
                                 "per_seed": res[C]["per_seed"],
                                 "nonconvergence": f"{conv[C]}/{n_fits}"} for C in grid},
            "best_C": best_C, "flat_region": flat,
            "recommended_C": (min(flat, key=lambda c: abs(np.log(c) - np.mean(np.log(flat))))
                              if len(flat) > 1 else best_C),
            "anchor_C": ANCHOR,
            "note": "후보 중 최댓값 선택이므로 낙관 편향이 있다. σ 안에서는 구간 중앙을 권장.",
            "fingerprint": pa.fingerprint(),
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n저장: artifacts/{stem}.json · {stem}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
