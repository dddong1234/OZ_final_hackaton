"""
exp_004 · LR + LightGBM 앙상블 — 모델 다양화가 도움이 되나
==========================================================

    .venv/bin/python3 experiments/iljun/exp_004_ensemble/ensemble.py
    .venv/bin/python3 experiments/iljun/exp_004_ensemble/ensemble.py --smoke

----------------------------------------------------------------------
왜 하나
----------------------------------------------------------------------
지금까지 모든 실험이 LogisticRegression(선형) 하나였다. 앙상블이 이득을 보려면
두 모델이 **서로 다른 실수**를 해야 한다. LightGBM(트리)은 선형모델과 결정
방식이 달라서, 섞으면(확률 평균) 서로의 실수를 상쇄할 여지가 있다.
팀 타임라인 04단계 = 앙상블. 경수님이 LGBM 단독 0.37대를 이미 확인했다
(LR 0.41보다 낮다 — 그래서 LGBM 이 더 세서가 아니라 '다양성'으로 기여하는지를 본다).

----------------------------------------------------------------------
설계
----------------------------------------------------------------------
같은 피처(GBV, C=0.07 정본 기반)를 두 모델에 똑같이 준다. fold 마다:
  · LR    predict_proba
  · LGBM  predict_proba
두 확률을 가중 평균해 argmax.  가중치 w: 예측 = w·LR + (1−w)·LGBM
  w=1.0 은 LR 단독(정본 재현), w 를 내리며 LGBM 을 섞는다.

측정: LR 단독 · LGBM 단독 · 가중 앙상블(여러 w) 을 **같은 fold** 에서 3-seed.
판정: 앙상블 − LR 단독을 paired 로. 3/3 seed 양수 + σ 초과라야 채택.

----------------------------------------------------------------------
★ CV↔LB 교훈 적용
----------------------------------------------------------------------
exp_003 에서 CV 개선의 ~63% 만 LB 로 갔고 간격이 벌어졌다. 그러니:
  · w 를 CV 최댓값으로 과튜닝하지 않는다 (작은 이득은 LB 에서 사라진다)
  · 앙상블 이득이 σ 안이면 "효과 없음"으로 본다
  · 진짜 판정은 제출로 (슬롯 조율 후)

fold 학습 분할에서만 fit. test 는 열지 않는다.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
for p in (_HERE, _EXP2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline as pa                                                # noqa: E402
import features_A as fa                                              # noqa: E402

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("lightgbm 이 필요합니다:  .venv/bin/pip install lightgbm")
    raise

# 팀 표준 LR
LR_PARAMS = {"solver": "lbfgs", "C": 0.07, "max_iter": 2000, "class_weight": "balanced"}
# LGBM 1차 파라미터 (튜닝 전 — 상수로 고정, 나중에 필요하면 탐색)
LGBM_PARAMS = {"objective": "multiclass", "class_weight": "balanced",
               "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
               "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.6,
               "n_jobs": -1, "verbose": -1}
WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]     # w = LR 비중


def macro(y, p):
    return round(float(f1_score(y, p, average="macro")), 5)


def align(proba_model, model_classes, classes_all):
    """모델별 predict_proba 를 전체 클래스 축으로 정렬한다."""
    out = np.zeros((proba_model.shape[0], len(classes_all)))
    pos = {c: i for i, c in enumerate(classes_all)}
    for j, c in enumerate(model_classes):
        out[:, pos[c]] = proba_model[:, j]
    return out


def paired(after: dict, before: dict):
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LR + LightGBM 앙상블")
    ap.add_argument("--root", default=None)
    ap.add_argument("--blocks", default="GBV", help="공유 피처 (기본 GBV)")
    ap.add_argument("--trees", type=int, default=None,
                    help="LGBM n_estimators (기본 300). 빠른 1차엔 100~150 권장")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    blocks = tuple(a.blocks)
    cv_seeds = pa.CONFIRMATION_CV_SEEDS[: (2 if a.smoke else None)]
    model_seed = pa.MODEL_SEED
    n_splits = 2 if a.smoke else 5
    weights = [1.0, 0.8, 0.6] if a.smoke else WEIGHTS
    if a.trees:
        LGBM_PARAMS["n_estimators"] = a.trees

    print("=" * 86)
    print("  exp_004 · LR + LightGBM 앙상블")
    print(f"  공유 피처 {''.join(blocks)} · LR {LR_PARAMS}")
    print(f"  LGBM {LGBM_PARAMS['n_estimators']}trees lr{LGBM_PARAMS['learning_rate']} "
          f"leaves{LGBM_PARAMS['num_leaves']} (튜닝 전)")
    print(f"  cv_seeds {list(cv_seeds)} · KFold-{n_splits} · 가중치 w(LR비중) {weights}")
    print("=" * 86)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    counts, _ = pa.parse_all(train, test, gene_cols)
    classes_all = np.array(sorted(pd.unique(y)))

    per_seed = {"LR": {}, "LGBM": {}}
    for w in weights:
        per_seed[f"ENS w{w}"] = {}

    for s in cv_seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        pr_lr = np.zeros((len(y), len(classes_all)))
        pr_gb = np.zeros((len(y), len(classes_all)))
        t0 = time.time()
        for i_tr, i_va in cv.split(train, y):
            spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=model_seed)
            Xa, _ = fa.build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, blocks)
            Xb, _ = fa.build_features(train.iloc[i_va], counts.iloc[i_va], spec, blocks)
            ytr = y[i_tr]

            lr = LogisticRegression(random_state=model_seed, **LR_PARAMS).fit(Xa, ytr)
            pr_lr[i_va] = align(lr.predict_proba(Xb), lr.classes_, classes_all)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gb = LGBMClassifier(random_state=model_seed, **LGBM_PARAMS).fit(Xa, ytr)
            pr_gb[i_va] = align(gb.predict_proba(Xb), gb.classes_, classes_all)

        per_seed["LR"][s] = macro(y, classes_all[pr_lr.argmax(1)])
        per_seed["LGBM"][s] = macro(y, classes_all[pr_gb.argmax(1)])
        for w in weights:
            mix = w * pr_lr + (1 - w) * pr_gb
            per_seed[f"ENS w{w}"][s] = macro(y, classes_all[mix.argmax(1)])
        print(f"  cv_seed {s}  LR {per_seed['LR'][s]:.5f}  LGBM {per_seed['LGBM'][s]:.5f}  "
              f"ENS(w0.8) {per_seed['ENS w0.8'][s]:.5f}   ({time.time()-t0:.0f}s)", flush=True)

    def ms(k):
        v = np.array(list(per_seed[k].values()), dtype=float)
        return round(v.mean(), 5), (round(v.std(ddof=1), 5) if len(v) > 1 else None)

    lr_m, lr_s = ms("LR")
    gb_m, gb_s = ms("LGBM")
    print("\n" + "=" * 86)
    print(f"  LR 단독    {lr_m:.5f} ± {lr_s}")
    print(f"  LGBM 단독  {gb_m:.5f} ± {gb_s}   (LR 대비 {gb_m-lr_m:+.5f})")
    print("=" * 86)
    print("  앙상블 (w = LR 비중)")
    rows = []
    for w in weights:
        k = f"ENS w{w}"
        m, sd = ms(k)
        dm, dsd, pos, nn = paired(per_seed[k], per_seed["LR"])
        rows.append({"w(LR)": w, "Macro F1": m, "σ": sd,
                     "LR 대비": round(dm, 5), "양수seed": f"{pos}/{nn}"})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    # 최적 w (w=1.0 은 LR 단독이라 제외하고 판정)
    cand = [w for w in weights if w < 1.0]
    best_w = max(cand, key=lambda w: ms(f"ENS w{w}")[0])
    bm, bsd = ms(f"ENS w{best_w}")
    dm, dsd, pos, nn = paired(per_seed[f"ENS w{best_w}"], per_seed["LR"])
    print(f"\n  최적 앙상블  w={best_w}  {bm:.5f} ± {bsd}")
    print(f"  LR 단독 대비 {dm:+.5f} ± {dsd}  ({pos}/{nn} seed 양수)")
    detected = (pos == nn and nn > 1) and abs(dm) >= (dsd if np.isfinite(dsd) and dsd > 0 else 0)
    if detected:
        print("  → 3/3 seed 양수 + σ 초과. 다양성 이득으로 볼 근거가 있다.")
    elif pos == nn and nn > 1:
        print("  → 방향은 일관되나 σ 안. 이득이 있다고 말하기 어렵다 (LB 로 확인).")
    else:
        print("  → seed 에 따라 갈린다. 앙상블 이득 없음.")
    print("  ※ CV 이득의 ~63%만 LB 로 갔던 exp_003 을 감안 — 작은 이득은 과신 금물.")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        tab.to_csv(art / "ensemble.csv", index=False, encoding="utf-8-sig")
        (art / "ensemble.json").write_text(json.dumps({
            "blocks": "".join(blocks), "lr_params": LR_PARAMS, "lgbm_params": LGBM_PARAMS,
            "validation": pa.validation_spec(n_splits, model_seed, cv_seeds),
            "weights": weights,
            "LR": {"f1_macro": lr_m, "f1_macro_std": lr_s, "per_seed": per_seed["LR"]},
            "LGBM": {"f1_macro": gb_m, "f1_macro_std": gb_s, "per_seed": per_seed["LGBM"]},
            "ensemble": {f"w{w}": {"f1_macro": ms(f'ENS w{w}')[0],
                                   "per_seed": per_seed[f"ENS w{w}"]} for w in weights},
            "best_w": best_w, "best_vs_lr": {"mean": dm, "std": dsd, "pos": f"{pos}/{nn}"},
            "detected": bool(detected),
            "note": "LGBM 파라미터 튜닝 전 1차. CV 이득은 LB 로 확인 필요.",
            "fingerprint": pa.fingerprint(),
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n저장: artifacts/ensemble.json · ensemble.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
