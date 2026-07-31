"""
판별력 기반 변이 토큰 피처 (D 블록)
====================================

토큰이란 (유전자, 변이문자열) 한 쌍이다. 예: (TP53, R248Q).
유전자 이진화(G)는 "TP53 에 변이가 있나"만 보지만, 토큰은 "TP53 의 어떤 변이냐"
까지 본다. 문헌에서 되는 방향이라 한 '변이 패턴'에 가장 가깝다.

──────────────────────────────────────────────────────────────────────
★ 배치 아티팩트 경고 (2026-07-31 데이터로 확인)
──────────────────────────────────────────────────────────────────────
판별력(lift)만으로 토큰을 고르면 ACC 같은 작은 암종의 germline/배치 변이가
뽑힌다. 예: LRIG1:L24V 는 ACC 환자 25명 전원(순도 1.0)에 있지만 LRIG1 은
ACC 드라이버가 아니다 — 배치 아티팩트다.

    train 안에서는 배치 아티팩트와 진짜 드라이버가 구별되지 않는다.
    (둘 다 "이 암종에 100% 몰림"으로 똑같이 보인다)
    차이는 test(다른 배치)에서만 드러난다.

그래서 이 피처가 올린 CV 는 **진짜인지 아티팩트인지 CV 로는 판정 불가**하다.
반드시 리더보드 제출로 확인해야 한다. 이 모듈은 가드를 넣어 위험을 줄이고,
run_tokens.py 가 아티팩트로 의심되는 토큰을 눈에 보이게 표시한다.

가드:
  1. functional 만  — silent(동의변이)·other 제외. 배치 아티팩트의 상당수가 silent.
  2. min_count      — fold-train 에서 M 회 이상 재현된 토큰만 후보.
  3. 후보 풀 공유    — 빈도(freq)와 판별력(disc)이 같은 후보 풀에서 순위만 다르게
                       고른다. 그래야 '선택 기준'만의 차이를 잰다.

모든 계산은 fold 학습 분할에서만 한다. validation 은 적용만 받는다 (대회 규칙 2번).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

# exp_002 의 파서(classify)를 재사용한다
_EXP2 = Path(__file__).resolve().parent.parent / "exp_002_variant_type"
if str(_EXP2) not in sys.path:
    sys.path.insert(0, str(_EXP2))
from features_A import classify, WT                                  # noqa: E402

__version__ = "D_v1_discriminative_tokens"

# functional = 기능적으로 의미 있는 변이. 이 유형만 토큰 후보로 쓴다.
FUNCTIONAL = {"missense", "nonsense", "frameshift", "indel"}


def parse_token_sets(df, gene_cols):
    """환자마다 functional (유전자,변이) 토큰의 집합을 만든다.

    전체 데이터에 대해 한 번만 부르면 된다. fold 는 인덱스로 나눠 쓴다.
    반환: 길이 n 의 리스트, 각 원소는 frozenset{(gene, var), ...}
    """
    raw = df[gene_cols].fillna(WT).values
    mask = raw != WT
    gcols = list(gene_cols)
    out = []
    for i in range(len(df)):
        toks = set()
        for j in np.flatnonzero(mask[i]):
            g = gcols[j]
            for v in set(raw[i][j].split()):        # 칸 내부 중복 제거(팀 표준)
                if classify(v) in FUNCTIONAL:
                    toks.add((g, v))
        out.append(frozenset(toks))
    return out


def fit_tokens(token_sets, y, idx_train, top_k, min_count=10, method="disc"):
    """fold 학습 분할에서 토큰을 고른다. y 를 쓰므로 fold-train 만 넣는다.

    token_sets : parse_token_sets 결과 (전체)
    idx_train  : 이 fold 의 학습 분할 위치 인덱스
    method     : 'freq'(빈도순) | 'disc'(판별력 lift 순)
    반환 spec  : {"order": [(gene,var), ...], "method", "diag": {...}}
                 order 는 상위 top_k. 실제 사용 시 앞에서 K 개를 자른다.
    """
    yt = np.asarray(y)[idx_train]
    n_tr = len(idx_train)

    # 클래스 사전확률 (fold-train)
    classes, cnts = np.unique(yt, return_counts=True)
    base = dict(zip(classes, cnts / n_tr))
    class_size = dict(zip(classes, cnts))

    # 토큰별 등장 수(n_t) 와 클래스별 등장 수
    n_t = {}
    per_class = {}
    for pos in idx_train:
        c = np.asarray(y)[pos]
        for tok in token_sets[pos]:
            n_t[tok] = n_t.get(tok, 0) + 1
            d = per_class.setdefault(tok, {})
            d[c] = d.get(c, 0) + 1

    # 후보 = min_count 이상 재현된 토큰 (freq·disc 공용 풀)
    cand = [t for t, n in n_t.items() if n >= min_count]

    def dominant(t):
        d = per_class[t]
        c = max(d, key=d.get)
        return c, d[c]

    def lift(t):
        c, kc = dominant(t)
        return (kc / n_t[t]) / base[c]

    if method == "freq":
        cand.sort(key=lambda t: (n_t[t], lift(t)), reverse=True)
    elif method == "disc":
        cand.sort(key=lambda t: (lift(t), n_t[t]), reverse=True)
    else:
        raise ValueError("method 는 'freq' 또는 'disc'")

    order = cand[:top_k]

    # 배치 아티팩트 진단 — 작은 암종에 100% 몰린 토큰 수를 센다
    flagged = []
    for t in order:
        c, kc = dominant(t)
        purity = kc / n_t[t]
        if purity >= 0.99 and class_size.get(c, 0) <= 120:
            flagged.append((f"{t[0]}:{t[1]}", int(n_t[t]), str(c), round(purity, 2)))

    return {
        "version": __version__, "method": method,
        "min_count": int(min_count), "n_candidates": len(cand),
        "order": order,
        "diag": {"n_flagged_artifact": len(flagged), "flagged_top": flagged[:10]},
    }


def transform_tokens(token_sets, idx, spec, k):
    """선택된 상위 k 토큰의 존재 여부를 이진 CSR [len(idx) × k] 로 만든다."""
    cols = spec["order"][:k]
    col_of = {t: j for j, t in enumerate(cols)}
    rows, cs = [], []
    for r, pos in enumerate(idx):
        for tok in token_sets[pos]:
            j = col_of.get(tok)
            if j is not None:
                rows.append(r); cs.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    M = sparse.csr_matrix((data, (rows, cs)), shape=(len(idx), len(cols)),
                          dtype=np.float32)
    names = [f"D_tok__{g}:{v}" for (g, v) in cols]
    return M, names
