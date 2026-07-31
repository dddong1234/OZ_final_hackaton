"""
규제(C) × 피처 블록 대조 — 내 피처가 규제와 독립적으로 기여하는가
================================================================

    .venv/bin/python3 experiments/iljun/exp_002_variant_type/compare_regularization.py
    .venv/bin/python3 experiments/iljun/exp_002_variant_type/compare_regularization.py --smoke

----------------------------------------------------------------------
왜 이 실험을 하나
----------------------------------------------------------------------
SDH 님 【전처리 검증 및 제출 전략】에서 확인된 것:

    lr_baseline_c1     0.344525      C=1.0
    lr_baseline_c0.1   0.383000      C 만 0.1 로 → +0.03847
    lr_burden_c0.1     0.396571      + log1p(변이 유전자 수) 1개 → 총 +0.05205

**exp_002 는 전부 C=1.0 (sklearn 기본값)으로 돌렸습니다.** 규제를 한 번도
건드리지 않았습니다. 6,201행 × 4,384 피처의 고차원 희소 데이터에서는 규제가
지배적인 변수일 수 있고, 실제로 SDH 님 결과에서는 제 피처 15개보다 C 하나가
더 큰 효과를 냈습니다.

    SDH  lr_burden_c0.1  피처 1개 · C=0.1   seed42  0.39657
    저   G+B+V+R         피처 15개 · C=1.0  seed42  0.38389

----------------------------------------------------------------------
이 스크립트가 답하는 질문
----------------------------------------------------------------------
1. **제 피처가 C=0.1 에서도 기여하는가?**
   규제가 이미 하던 일을 피처가 대신하고 있었다면 C=0.1 에서는 증분이 사라집니다.
   독립적으로 기여한다면 증분이 유지되고, 둘을 합친 것이 최고가 됩니다.

2. **SDH 님 0.39657 을 제 파이프라인으로 재현할 수 있는가?**
   `G+N` (= 이진화 + log1p(변이 유전자 수) 1개) 이 SDH 님 `lr_burden_c0.1` 과
   같은 구성입니다. 재현되면 규제 조건에서도 두 파이프라인이 비교 가능합니다.

----------------------------------------------------------------------
설계
----------------------------------------------------------------------
블록 5종 × C 2종 × cv_seed 3종 = CV 30회. `max_iter` 는 **양쪽 모두 2000 으로
고정**해 C 만 유일한 변수가 되게 합니다 (SDH 님 전처리 비교 조건과 동일).

    G      이진화만                     ← 공통 기준
    G+N    + log1p(변이 유전자 수) 1개   ← SDH lr_burden 과 동일 구성
    G+B    + 부담 3개 (유전자수/이벤트수/다중)
    G+B+V  + 유형 카운트 6개
    G+B+V+R  + 유형 비율 6개

⚠️ 이 스크립트는 최적 C 를 찾지 않습니다. C 두 값에서 피처 효과가 유지되는지만
   봅니다. C 세부 탐색은 전처리 확정 후에 하는 것이 팀 절차입니다
   (SDH 님 문서 §8: 전처리 확정 → C 탐색 → 다른 모델 → 앙상블).
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

# SDH 님 전처리 비교 조건과 동일. C 만 바꾼다.
BASE_PARAMS = {"solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced"}
C_VALUES = [1.0, 0.1]

BLOCKS = [
    ("G",    "G       이진화만"),
    ("GN",   "G+N     + 유전자수 1개 (SDH lr_burden 동일)"),
    ("GB",   "G+B     + 부담 3개"),
    ("GBV",  "G+B+V   + 유형 카운트"),
    ("GBVR", "G+B+V+R + 유형 비율"),
]

# SDH 님 보고값 (seed 42, 5-fold OOF) — 재현 대조용
SDH_REF = {("G", 1.0): 0.344525, ("G", 0.1): 0.383000, ("GN", 0.1): 0.396571}


def paired_delta(after: dict, before: dict):
    a = {d["cv_seed"]: d["f1_macro"] for d in after["per_seed"]}
    b = {d["cv_seed"]: d["f1_macro"] for d in before["per_seed"]}
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return float(d.mean()), sd, int((d > 0).sum()), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="규제 × 피처 블록 대조")
    ap.add_argument("--root", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = pa.load_cfg()["pipeline"]
    cv_seeds = tuple(cfg["cv"]["seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg["cv"].get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["cv"]["n_splits"]
    n_fits = len(cv_seeds) * n_splits

    print("=" * 84)
    print("  규제(C) × 피처 블록 대조")
    print(f"  features_A {fa.__version__} · cv_seeds {list(cv_seeds)} · "
          f"model_seed {model_seed} · StratifiedKFold-{n_splits}")
    print(f"  고정: solver=lbfgs · max_iter={BASE_PARAMS['max_iter']} · "
          f"class_weight=balanced   /   변수: C {C_VALUES}")
    print("=" * 84)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    cnt_train, _ = pa.parse_all(train, test, gene_cols)

    res, conv = {}, {}
    for C in C_VALUES:
        print(f"\n── C = {C} " + "─" * 60)
        mp = dict(BASE_PARAMS, C=C)
        for blocks, label in BLOCKS:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                r = pa.cross_validate_multi(train, y, cnt_train, gene_cols, tuple(blocks),
                                            model_key="logreg", model_params=mp,
                                            cv_seeds=cv_seeds, model_seed=model_seed,
                                            n_splits=n_splits, v=False)
            conv[(blocks, C)] = sum(1 for x in caught
                                    if issubclass(x.category, ConvergenceWarning))
            res[(blocks, C)] = r
            per = "  ".join(f"{d['f1_macro']:.5f}" for d in r["per_seed"])
            std = f"± {r['f1_macro_std']:.5f}" if r["f1_macro_std"] is not None else ""
            flag = f"  ⚠미수렴 {conv[(blocks, C)]}/{n_fits}" if conv[(blocks, C)] else ""
            ref = SDH_REF.get((blocks, C))
            refs = ""
            if ref is not None and not a.smoke:
                d42 = next((x["f1_macro"] for x in r["per_seed"] if x["cv_seed"] == 42), None)
                if d42 is not None:
                    refs = f"   [SDH {ref:.5f} · 차이 {d42 - ref:+.5f}]"
            print(f"{label:44} F1 {r['f1_macro']:.5f} {std}   [seed별 {per}]{flag}{refs}")

    # ── C 효과 (같은 블록에서 C=0.1 − C=1.0) ─────────────────────────
    print()
    print("C 효과  (같은 블록에서 C=0.1 − C=1.0, paired)")
    c_rows = []
    for blocks, label in BLOCKS:
        m, sd, pos, n = paired_delta(res[(blocks, 0.1)], res[(blocks, 1.0)])
        c_rows.append({"블록": blocks, "평균 차이": round(m, 5),
                       "차이 σ": round(sd, 5), "양수 seed": f"{pos}/{n}"})
    c_tab = pd.DataFrame(c_rows)
    print(c_tab.to_string(index=False))

    # ── 피처 효과 (같은 C 에서 G 대비) ───────────────────────────────
    print()
    print("피처 효과  (같은 C 에서 G 대비 paired 증분)")
    f_rows = []
    for C in C_VALUES:
        for blocks, label in BLOCKS[1:]:
            m, sd, pos, n = paired_delta(res[(blocks, C)], res[("G", C)])
            f_rows.append({"C": C, "블록": blocks, "평균 증분": round(m, 5),
                           "증분 σ": round(sd, 5), "양수 seed": f"{pos}/{n}"})
    f_tab = pd.DataFrame(f_rows)
    print(f_tab.to_string(index=False))

    print()
    print("읽는 법")
    print("  · C=1.0 에서 컸던 증분이 C=0.1 에서도 유지되면 → 피처와 규제는 독립적으로 기여")
    print("  · C=0.1 에서 증분이 사라지면 → 규제가 하던 일을 피처가 대신하고 있었던 것")
    print("  · GN @ C=0.1 이 SDH 0.39657 에 가까우면 → 두 파이프라인이 규제 조건에서도 비교 가능")
    print("※ n=3 의 σ 는 그 자체로 부정확하다. 방향과 대략적 크기만 읽는다.")

    if not a.smoke:
        art = root / "experiments" / "iljun" / "exp_002_variant_type" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        c_tab.to_csv(art / "reg_check_C_effect.csv", index=False, encoding="utf-8-sig")
        f_tab.to_csv(art / "reg_check_feature_effect.csv", index=False, encoding="utf-8-sig")
        summary = {
            "features_version": fa.__version__,
            "features_sha256": pa.sha256(fa.__file__),
            "fixed_params": BASE_PARAMS, "C_values": C_VALUES,
            "cv_seeds": [int(s) for s in cv_seeds], "model_seed": int(model_seed),
            "n_splits": int(n_splits),
            "results": {f"{b}@C{C}": {"f1_macro": res[(b, C)]["f1_macro"],
                                      "f1_macro_std": res[(b, C)]["f1_macro_std"],
                                      "accuracy": res[(b, C)]["accuracy"],
                                      "dim": res[(b, C)]["dim"],
                                      "per_seed": res[(b, C)]["per_seed"],
                                      "nonconvergence": f"{conv[(b, C)]}/{n_fits}"}
                        for b, _ in BLOCKS for C in C_VALUES},
            "C_effect": c_rows, "feature_effect": f_rows,
            "sdh_reference_seed42": {f"{k[0]}@C{k[1]}": v for k, v in SDH_REF.items()},
            "note": "SDH exp_002 의 C=0.1 발견을 내 블록에 적용. max_iter=2000 고정, C 만 변수.",
        }
        (art / "reg_check.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n저장: artifacts/reg_check.json · reg_check_C_effect.csv · "
              "reg_check_feature_effect.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
