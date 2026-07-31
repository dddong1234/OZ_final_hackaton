"""
팀 공용 단독-모델 평가 하니스
==============================

어떤 모델이든 **같은 CV 프로토콜**(StratifiedKFold-5 × 다중 seed, OOF Macro F1)로
재서, 팀원 누구의 "단독 최고점"이든 같은 잣대로 비교되게 한다.

핵심 두 개를 함수로 주입한다:
  make_model(seed)                         -> 학습 안 된 estimator
  make_rep(train, counts, gene_cols, i_tr, i_va, seed) -> (X_tr, X_va)

make_rep 이 fold 안에서 fit_spec·SVD 등 '학습되는 전처리'를 fold-train 에서만
fit 하도록 책임진다. 그래서 이 하니스 자체는 모델·표현에 무관하다(=공용).

fold 학습 분할에서만 fit. test 는 열지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
if str(_EXP2) not in sys.path:
    sys.path.insert(0, str(_EXP2))

import pipeline as pa                                                 # noqa: E402


def macro(y, p):
    return round(float(f1_score(y, p, average="macro")), 5)


def paired(after: dict, before: dict):
    """같은 seed 끼리 짝지어 차이를 본다."""
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def evaluate(train, y, gene_cols, counts, make_model, make_rep,
             cv_seeds, n_splits, model_seed=42):
    """seed 별 OOF Macro F1 딕셔너리를 돌려준다."""
    per_seed = {}
    for s in cv_seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        oof = np.empty(len(y), dtype=object)
        for i_tr, i_va in cv.split(train, y):
            X_tr, X_va = make_rep(train, counts, gene_cols, i_tr, i_va, model_seed)
            oof[i_va] = make_model(model_seed).fit(X_tr, y[i_tr]).predict(X_va)
        per_seed[s] = macro(y, np.array(list(oof)))
    return per_seed


def summary(per_seed: dict):
    v = np.array(list(per_seed.values()), dtype=float)
    return round(float(v.mean()), 5), (round(float(v.std(ddof=1)), 5) if len(v) > 1 else None)
