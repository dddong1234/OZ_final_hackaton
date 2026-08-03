"""
제출 파일 생성 — LR + LightGBM 앙상블 (w=0.8, trees=100)
=========================================================

    .venv/bin/python3 experiments/iljun/exp_004_ensemble/submit_ensemble.py
    .venv/bin/python3 experiments/iljun/exp_004_ensemble/submit_ensemble.py --smoke

exp_004 에서 CV 최고였던 구성(w=0.8 · trees=100 · CV 0.42895, 3/3 seed)을
전체 train 으로 학습해 test 를 예측한다. 한 장만 만든다.

----------------------------------------------------------------------
왜 이 한 장인가
----------------------------------------------------------------------
확정된 LB 3점(0.260 / 0.2795 / 0.284)으로 CV→LB 패스스루 ~65%, 간격 ~0.13 이
일관됨을 확인했다. 그럼 지금 가진 것 중 CV 최고(앙상블 0.42895)가 예상 LB
최고(~0.295)다. 토큰과 달리 '모델 다양성'이라 다른 축이기도 하다.
※ 이 앙상블의 LGBM 은 아직 튜닝 전(기본 100트리·wide-sparse). exp_006 로
  LGBM 을 살리면 더 나은 앙상블이 나온다 — 이건 '개념 확인 + 새 최고 확보'용.

----------------------------------------------------------------------
규칙 준수
----------------------------------------------------------------------
· 피처 spec 은 전체 train 으로만 fit, test 는 예측에만 사용(통계 미반영).
· 외부 유전자-암종 지식 미사용. leakage 없음.
"""
from __future__ import annotations

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

import pipeline as pa                                                 # noqa: E402
import features_A as fa                                              # noqa: E402
from ensemble import LR_PARAMS, align                                # noqa: E402

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("lightgbm 필요:  .venv/bin/pip install lightgbm")
    raise

# exp_004 최고 구성
W_LR = 0.8                                    # LR 비중 (LGBM 0.2)
LGBM_PARAMS = {"objective": "multiclass", "class_weight": "balanced",
               "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31,
               "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.6,
               "n_jobs": -1, "verbose": -1}
BLOCKS = ("G", "B", "V")
SEED = 42


def fit_pair(Xtr, ytr):
    lr = LogisticRegression(random_state=SEED, **LR_PARAMS).fit(Xtr, ytr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gb = LGBMClassifier(random_state=SEED, **LGBM_PARAMS).fit(Xtr, ytr)
    return lr, gb


def blend_proba(lr, gb, X, classes_all):
    p_lr = align(lr.predict_proba(X), lr.classes_, classes_all)
    p_gb = align(gb.predict_proba(X), gb.classes_, classes_all)
    return W_LR * p_lr + (1 - W_LR) * p_gb


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="배선 점검(점수 무의미)")
    a = ap.parse_args()

    root = pa.find_project_root()
    n_splits = 2 if a.smoke else 5
    out = root / "experiments" / "iljun" / "results" / "iljun-exp004-ensemble"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  제출 — LR+LGBM 앙상블  w(LR)={W_LR} · LGBM {LGBM_PARAMS['n_estimators']}trees")
    print(f"  LR {LR_PARAMS} · blocks {''.join(BLOCKS)}")
    print("=" * 80)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    counts_tr, counts_te = pa.parse_all(train, test, gene_cols)
    classes_all = np.array(sorted(pd.unique(y)))

    # ── seed42 CV 정합성 (기록 0.42895 와 대조) ────────────────────────
    print("\n[1] seed42 5-fold 앙상블 CV (기록 0.42895 와 대조)")
    t0 = time.time()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.empty(len(y), dtype=object)
    for i_tr, i_va in cv.split(train, y):
        spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=SEED)
        Xa, _ = fa.build_features(train.iloc[i_tr], counts_tr.iloc[i_tr], spec, BLOCKS)
        Xb, _ = fa.build_features(train.iloc[i_va], counts_tr.iloc[i_va], spec, BLOCKS)
        lr, gb = fit_pair(Xa, y[i_tr])
        mix = blend_proba(lr, gb, Xb, classes_all)
        oof[i_va] = classes_all[mix.argmax(1)]
    oof = np.array(list(oof))
    cv_f1 = round(float(f1_score(y, oof, average="macro")), 5)
    cv_acc = round(float(accuracy_score(y, oof)), 5)
    print(f"  seed42 CV Macro F1 {cv_f1:.5f} / Acc {cv_acc:.5f}   ({time.time()-t0:.0f}s)")

    # ── 전체 train 학습 → test 예측 ───────────────────────────────────
    print("\n[2] 전체 train 학습 → test 예측")
    t0 = time.time()
    spec = fa.fit_spec(train, gene_cols, seed=SEED)
    Xtr, _ = fa.build_features(train, counts_tr, spec, BLOCKS)
    Xte, _ = fa.build_features(test, counts_te, spec, BLOCKS)
    lr, gb = fit_pair(Xtr, y)
    mix_te = blend_proba(lr, gb, Xte, classes_all)
    pred = classes_all[mix_te.argmax(1)]
    print(f"  ({time.time()-t0:.0f}s)")

    # ── 검증 + 저장 ───────────────────────────────────────────────────
    ids = test[pa.ID].values
    df = pd.DataFrame({pa.ID: ids, pa.TARGET: pred})
    assert len(df) == len(test), "행 수 불일치"
    assert df[pa.TARGET].notna().all(), "NaN 예측 있음"
    unknown = set(df[pa.TARGET]) - set(y)
    assert not unknown, f"train 에 없는 라벨 예측: {unknown}"

    f = out / f"submission_ENS_w{W_LR}_t{LGBM_PARAMS['n_estimators']}_cv{cv_f1:.5f}.csv"
    df.to_csv(f, index=False)
    print("\n" + "=" * 80)
    print(f"  저장: {f}")
    print(f"  예측 분포 상위: {df[pa.TARGET].value_counts().head(5).to_dict()}")
    print(f"  seed42 CV {cv_f1:.5f} — 기록 0.42895 와 σ 안이면 정상.")
    print("  데이콘 업로드 후 LB 를 TEST log(ENS-004a)에 적어주세요.")
    print(f"  예상 LB ~0.295 (패스스루 65% 가정). 현재 최고 0.284 대비 +0.011.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
