"""
클래스 편향 보정 — 재학습 없이 확률만 조정한다
================================================

    .venv/bin/python3 experiments/member_d/exp_002_variant_type/tune_class_bias.py

----------------------------------------------------------------------
무슨 문제를 푸나
----------------------------------------------------------------------
`diagnose_classes.py` 결과에서 **처방이 정반대인 두 그룹**이 나왔다.

    A 과잉예측  THYM · PCPG · PRAD · TGCT · OV     → 덜 찍게 해야 함
       THYM: 정밀도 0.155 / 재현율 0.582 · 마커 0개
       THCA 환자의 19.8%, LAML 의 27.2%, PRAD 의 16.2% 를 빨아들이고 있다

    B 누락      GBMLGG · BLCA · DLBC              → 더 찍게 해야 함
       DLBC: 정밀도 0.813 / 재현율 0.307 · 마커 60개
       찍으면 81% 맞는데 실제 환자의 31% 밖에 못 찾는다

----------------------------------------------------------------------
어떻게 고치나 — 확률에 배수를 곱한다
----------------------------------------------------------------------
LogisticRegression 은 26개 클래스마다 확률을 내고 **가장 큰 것**을 답으로 고른다.

    예측 = argmax( p_THYM, p_THCA, p_DLBC, ... )

여기서 클래스마다 배수를 곱하면 찍는 빈도를 바꿀 수 있다.

    조정 확률 = p_c × w_c
    THYM 의 w 를 0.5 로  →  THYM 이 이기려면 두 배 확신이 필요 → 덜 찍는다
    DLBC 의 w 를 2.0 으로 →  DLBC 가 더 쉽게 이긴다           → 더 찍는다

**재학습이 필요 없다.** OOF 확률을 한 번 구해두면 수천 가지 조합을 즉시 시험할 수 있다.

----------------------------------------------------------------------
과적합을 피하는 방법 (중요)
----------------------------------------------------------------------
같은 데이터로 배수를 고르고 그 데이터로 점수를 매기면 **당연히 좋아 보인다.**
그건 성능이 아니라 착시다. 그래서 이렇게 나눈다.

    cv_seed 42        → 배수를 고르는 데만 사용 (튜닝)
    cv_seed 52, 62    → 고른 배수를 그대로 적용해 확인 (검증)

**52·62 에서도 오르면 진짜**고, 42 에서만 오르면 과적합이다.

자유 파라미터를 최소로 둔다. 26개를 각각 조정하면 26개 파라미터라 반드시 과적합한다.
여기서는 **진단 그룹 단위로 2개**만 쓴다 (A 그룹 배수, B 그룹 배수).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pipeline as pa                                                # noqa: E402

# diagnose_classes.py 결과 (cv_seeds 42/52/62 평균 기준)
GROUP_A = ["THYM", "PCPG", "PRAD", "TGCT", "OV"]        # 과잉예측 → 줄인다
GROUP_B = ["GBMLGG", "BLCA", "DLBC"]                    # 누락     → 늘린다

A_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
B_GRID = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]


def apply_weights(proba, classes, w):
    """확률에 클래스별 배수를 곱하고 argmax. w 는 {클래스: 배수} 딕셔너리."""
    mult = np.array([w.get(c, 1.0) for c in classes], dtype=float)
    return classes[np.argmax(proba * mult, axis=1)]


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="클래스 편향 보정")
    ap.add_argument("--root", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = pa.load_cfg()["pipeline"]
    blocks = tuple(cfg["blocks"])
    cv_seeds = tuple(cfg["cv"]["seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg["cv"].get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["cv"]["n_splits"]
    mp = dict(cfg.get("model_params") or pa.DEFAULT_MODEL_PARAMS[cfg["model"]])

    print("=" * 82)
    print("  클래스 편향 보정 — 재학습 없이 확률 배수만 조정")
    print(f"  피처 {cfg['blocks']} · {mp} · cv_seeds {list(cv_seeds)} · KFold-{n_splits}")
    print(f"  A 과잉예측(줄임) {GROUP_A}")
    print(f"  B 누락(늘림)     {GROUP_B}")
    print("=" * 82)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    cnt_train, _ = pa.parse_all(train, test, gene_cols)

    # ── seed 별 OOF 확률 ─────────────────────────────────────────────
    store = {}
    for s in cv_seeds:
        r = pa.cross_validate(train, y, cnt_train, gene_cols, blocks,
                              model_key=cfg["model"], model_params=mp,
                              cv_seed=s, model_seed=model_seed,
                              n_splits=n_splits, v=False, return_proba=True)
        store[s] = r
        print(f"  cv_seed {s}  보정 전 Macro F1 {r['f1_macro']:.5f}")

    classes = store[cv_seeds[0]]["classes"]
    tune_seed = cv_seeds[0]
    check_seeds = list(cv_seeds[1:])

    base = {s: macro_f1(y, store[s]["oof"]) for s in cv_seeds}
    print(f"\n  튜닝 seed {tune_seed} · 검증 seed {check_seeds}")

    # ── 격자 탐색 (튜닝 seed 에서만) ─────────────────────────────────
    print("\n[1] 배수 탐색 — cv_seed %d 에서만" % tune_seed)
    rows = []
    for wa in A_GRID:
        for wb in B_GRID:
            w = {c: wa for c in GROUP_A} | {c: wb for c in GROUP_B}
            f = macro_f1(y, apply_weights(store[tune_seed]["proba"], classes, w))
            rows.append({"A배수": wa, "B배수": wb, "튜닝seed F1": round(f, 5),
                         "보정전 대비": round(f - base[tune_seed], 5)})
    grid = pd.DataFrame(rows).sort_values("튜닝seed F1", ascending=False)
    print(grid.head(10).to_string(index=False))

    best = grid.iloc[0]
    wa, wb = float(best["A배수"]), float(best["B배수"])
    w_best = {c: wa for c in GROUP_A} | {c: wb for c in GROUP_B}
    print(f"\n  선택: A배수 {wa} · B배수 {wb}   (튜닝 seed 에서 {best['보정전 대비']:+.5f})")

    # ── 검증 seed 에 그대로 적용 ─────────────────────────────────────
    print(f"\n[2] 검증 — 위 배수를 seed {check_seeds} 에 그대로 적용")
    ver = []
    for s in cv_seeds:
        after = macro_f1(y, apply_weights(store[s]["proba"], classes, w_best))
        ver.append({"cv_seed": s, "역할": "튜닝" if s == tune_seed else "검증",
                    "보정 전": round(base[s], 5), "보정 후": round(after, 5),
                    "차이": round(after - base[s], 5)})
    vdf = pd.DataFrame(ver)
    print(vdf.to_string(index=False))

    chk = vdf[vdf["역할"] == "검증"]
    gain = float(chk["차이"].mean()) if len(chk) else 0.0
    sigma = float(np.std(list(base.values()), ddof=1))
    print(f"\n  검증 seed {len(chk)}개 중 {int((chk['차이'] > 0).sum())}개에서 개선")
    print(f"  평균 개선 (검증 seed만) {gain:+.5f}")
    print(f"  보정 전 seed 편차 σ    {sigma:.5f}   →  개선폭은 {abs(gain)/sigma:.2f}σ")

    # 방향이 일관돼도 크기가 σ 안이면 '검출되지 않았다'고 적는다
    consistent = bool((chk["차이"] > 0).all()) if len(chk) else False
    detected = consistent and abs(gain) >= sigma
    ok = detected
    if detected:
        print("  → 방향이 일관되고 크기도 σ 를 넘는다. 실제 효과로 볼 근거가 있다.")
    elif consistent:
        print("  → 방향은 일관되나 크기가 seed 편차 안에 있다.")
        print("     **효과가 있다고 말할 수 없다.** 이 설계로는 검출하지 못했다.")
    else:
        print("  → 검증 seed 에서 재현되지 않았다. 튜닝 seed 과적합으로 봐야 한다.")

    # ── 어떤 클래스가 어떻게 바뀌었나 (검증 seed 하나로) ──────────────
    s_show = check_seeds[0] if check_seeds else tune_seed
    before_pred = store[s_show]["oof"]
    after_pred = apply_weights(store[s_show]["proba"], classes, w_best)
    pb, rb, fb, sup = precision_recall_fscore_support(y, before_pred, labels=classes, zero_division=0)
    pa_, ra, fa_, _ = precision_recall_fscore_support(y, after_pred, labels=classes, zero_division=0)
    dd = pd.DataFrame({"환자수": sup,
                       "정밀도_전": pb.round(3), "정밀도_후": pa_.round(3),
                       "재현율_전": rb.round(3), "재현율_후": ra.round(3),
                       "F1_전": fb.round(4), "F1_후": fa_.round(4),
                       "F1변화": (fa_ - fb).round(4)}, index=classes)
    dd["그룹"] = ["A" if c in GROUP_A else ("B" if c in GROUP_B else "") for c in classes]
    print(f"\n[3] 클래스별 변화 (cv_seed {s_show}, 검증용)")
    print("\n  조정한 클래스")
    print(dd[dd["그룹"] != ""].sort_values("F1변화", ascending=False).to_string())
    other = dd[dd["그룹"] == ""].sort_values("F1변화", ascending=False)
    print("\n  건드리지 않았는데 오른 클래스 상위 5")
    print(other.head(5)[["환자수", "F1_전", "F1_후", "F1변화"]].to_string())
    print("\n  건드리지 않았는데 내린 클래스 하위 5")
    print(other.tail(5)[["환자수", "F1_전", "F1_후", "F1변화"]].to_string())
    print(f"\n  조정 클래스 합계 변화 {dd[dd['그룹']!='']['F1변화'].sum():+.4f}")
    print(f"  나머지    합계 변화 {other['F1변화'].sum():+.4f}")

    if not a.smoke:
        art = root / "experiments" / "member_d" / "exp_002_variant_type" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        grid.to_csv(art / "bias_grid.csv", index=False, encoding="utf-8-sig")
        vdf.to_csv(art / "bias_verification.csv", index=False, encoding="utf-8-sig")
        dd.to_csv(art / "bias_class_change.csv", encoding="utf-8-sig")
        (art / "bias_tuning.json").write_text(json.dumps({
            "blocks": cfg["blocks"], "model_params": mp,
            "group_A_overpredict": GROUP_A, "group_B_underpredict": GROUP_B,
            "tune_seed": int(tune_seed), "check_seeds": [int(s) for s in check_seeds],
            "chosen": {"A": wa, "B": wb},
            "baseline_by_seed": {str(k): v for k, v in base.items()},
            "verification": ver,
            "verified_on_holdout_seeds": ok,
            "note": "튜닝 seed 에서만 배수를 고르고 나머지 seed 로 확인. 자유 파라미터 2개.",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n저장: artifacts/bias_grid.csv · bias_verification.csv · "
              "bias_class_change.csv · bias_tuning.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
