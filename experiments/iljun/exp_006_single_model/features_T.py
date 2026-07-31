"""
exp_006 · 트리 친화 표현 (T)
=============================

왜
--
트리(LGBM 등)는 4,235개 넓고 희소한 이진 유전자 피처를 잘 못 쓴다. 트리는 한
번에 피처 하나로만 가르는데, 0/1 이 대부분 0 인 피처 수천 개는 각각 신호가
너무 약하다. 그래서 트리엔 **조밀·저차원** 표현을 줘야 공정하다.

무엇
----
G(유전자 이진) 을 TruncatedSVD 로 압축한 dense 성분 + 이미 조밀한 B(부담)·
V(유형 카운트) 를 이어붙인다. SVD 는 유전자들의 공분산 구조(어떤 유전자들이
함께 변이하나)를 몇십~몇백 축으로 요약해 트리가 쓰기 좋게 만든다.

규칙 준수
---------
· SVD 는 fold-train 에서만 fit 하고 val·test 엔 transform 만 한다.
  = "Train 통계를 Test 에 적용"이라 leakage 아님(운영개요 §5).
· 외부 지식 없음. test 개발 중 미열람.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
if str(_EXP2) not in sys.path:
    sys.path.insert(0, str(_EXP2))

import features_A as fa                                              # noqa: E402

__version__ = "T_v1_svd_dense"


def fit_svd(df_tr, counts_tr, spec, svd_dim, seed=42):
    """fold-train 의 G 에서 SVD 를 학습한다."""
    Xg, _ = fa.build_features(df_tr, counts_tr, spec, ("G",))
    k = int(min(svd_dim, Xg.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    svd.fit(Xg)
    return {"svd": svd, "svd_dim": k,
            "explained": float(svd.explained_variance_ratio_.sum())}


def transform(df, counts, spec, tspec):
    """G→SVD dense 성분 + B·V dense 를 이어붙인 dense 행렬 [n, k+9]."""
    Xg, _ = fa.build_features(df, counts, spec, ("G",))
    comp = tspec["svd"].transform(Xg)                     # dense [n, k]
    Xbv, _ = fa.build_features(df, counts, spec, ("B", "V"))
    bv = Xbv.toarray() if sparse.issparse(Xbv) else np.asarray(Xbv)
    return np.hstack([comp, bv]).astype(np.float32)
