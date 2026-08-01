"""
exp_007 · 아미노산 치환쌍(A_pair) + 표기 구조(S) — P 모듈
==========================================================

왜
--
팀 상위(홍주 biodomain02 LB 0.351, SDH case_06 LB 0.342)가 쓴 '이기는 축'을
우리 파이프라인에서 재현한다. 변이 표기 `R132H` 를 [ref R][pos 132][alt H] 로
읽어, **아미노산 치환의 방향**을 피처로 만든다. 이게 우리가 여태 안 판 축이다.

블록
----
  P_pair : ref→alt 치환쌍 380종(20x20-20) 을 환자별로 센다.  ← SDH CV 1위 블록
  P_marg : ref / alt 아미노산 각각의 카운트 (20+20)
  S      : 표기 유형 다양도·엔트로피·dominant share + 유전자별 이벤트 topology

규칙 준수
---------
· 380쌍·20AA 는 데이터/라벨과 무관하게 고정 → **fit 불필요 = leakage 원천봉쇄**.
  (전체 train 에서 한 번 만들어 fold 인덱스로 잘라 써도 안전하다.)
· test 는 예측에만. 외부 유전자-암종 지식 없음. 표기 문자열과 train label 만.
· SDH 와 같은 AA 집합·정규식을 써서 380쌍 정의를 정확히 맞춘다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
if str(_EXP2) not in sys.path:
    sys.path.insert(0, str(_EXP2))

from features_A import classify, WT                                   # noqa: E402

__version__ = "P_v1_aa_pairs"

# SDH exp_009 와 동일한 정의
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
SUBSTITUTION = re.compile(r"^([A-Z])(-?\d+)([A-Z])$")     # 미스센스: 단문자 ref/alt
PAIRS = tuple(f"{r}>{a}" for r in AA for a in AA if r != a)   # 380
PAIR_IDX = {p: i for i, p in enumerate(PAIRS)}
AA_IDX = {a: i for i, a in enumerate(AA)}


def _events_per_row(df, gene_cols):
    """행마다 (gene, variant) 리스트를 만든다. WT 제외, 칸 내부는 공백 분리."""
    raw = df[gene_cols].fillna(WT).values
    mask = raw != WT
    gcols = list(gene_cols)
    for i in range(len(df)):
        row = []
        for j in np.flatnonzero(mask[i]):
            g = gcols[j]
            for v in raw[i][j].split():
                row.append((g, v))
        yield i, row


def pair_matrix(df, gene_cols):
    """환자별 ref→alt 치환쌍 380종 카운트 CSR [n x 380]."""
    n = len(df)
    rows, cols, vals = [], [], []
    for i, row in _events_per_row(df, gene_cols):
        c = {}
        for _g, v in row:
            m = SUBSTITUTION.match(v)
            if not m:
                continue
            ref, _pos, alt = m.groups()
            if ref == alt:                       # silent 은 방향 없음 → 제외
                continue
            idx = PAIR_IDX.get(f"{ref}>{alt}")
            if idx is not None:
                c[idx] = c.get(idx, 0) + 1
        for idx, cnt in c.items():
            rows.append(i); cols.append(idx); vals.append(cnt)
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(PAIRS)), dtype=np.float32)
    return M, [f"P_pair__{p}" for p in PAIRS]


def marg_matrix(df, gene_cols):
    """ref / alt 아미노산 각각의 카운트 CSR [n x 40]."""
    n = len(df); k = len(AA)
    rows, cols, vals = [], [], []
    for i, row in _events_per_row(df, gene_cols):
        c = {}
        for _g, v in row:
            m = SUBSTITUTION.match(v)
            if not m:
                continue
            ref, _pos, alt = m.groups()
            ri, ai = AA_IDX.get(ref), AA_IDX.get(alt)
            if ri is not None:
                c[ri] = c.get(ri, 0) + 1
            if ai is not None:
                c[k + ai] = c.get(k + ai, 0) + 1
        for idx, cnt in c.items():
            rows.append(i); cols.append(idx); vals.append(cnt)
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(n, 2 * k), dtype=np.float32)
    return M, [f"P_ref__{a}" for a in AA] + [f"P_alt__{a}" for a in AA]


def s_matrix(df, gene_cols):
    """표기 구조 S — 유전자별 이벤트 topology + 유형 분포(다양도·엔트로피·dominant)."""
    names = ["S__one_event_genes", "S__two_event_genes", "S__threeplus_event_genes",
             "S__multi_type_genes", "S__max_events_one_gene",
             "S__type_diversity", "S__type_entropy", "S__dominant_type_share",
             "S__total_events"]
    n = len(df)
    feats = np.zeros((n, len(names)), dtype=np.float32)
    for i, row in _events_per_row(df, gene_cols):
        if not row:
            continue
        per_gene = {}                       # gene -> [event_count, {types}]
        type_counts = {}
        for g, v in row:
            kind = classify(v)
            d = per_gene.setdefault(g, [0, set()])
            d[0] += 1; d[1].add(kind)
            type_counts[kind] = type_counts.get(kind, 0) + 1
        counts = [d[0] for d in per_gene.values()]
        feats[i, 0] = sum(1 for c in counts if c == 1)
        feats[i, 1] = sum(1 for c in counts if c == 2)
        feats[i, 2] = sum(1 for c in counts if c >= 3)
        feats[i, 3] = sum(1 for d in per_gene.values() if len(d[1]) >= 2)
        feats[i, 4] = max(counts)
        total = sum(type_counts.values())
        props = np.array([v / total for v in type_counts.values()], dtype=float)
        feats[i, 5] = len(type_counts)
        feats[i, 6] = float(-(props * np.log(props)).sum())
        feats[i, 7] = float(props.max())
        feats[i, 8] = total
    return sparse.csr_matrix(feats), names


def build_P(df, gene_cols, blocks="pair", log1p=True):
    """blocks 문자에 따라 P 블록들을 이어붙인다. 'p'air 'm'arg 's'.
       log1p=True 면 카운트 블록(pair·marg)에 log1p 를 씌운다 — GBV 이진과 스케일을
       맞춰 lbfgs 수렴을 돕는다(B/V 블록과 동일한 처리). S 는 이미 파생지표라 그대로.
       (leakage 없음 — 380쌍·20AA 고정, 전체 df 로 만들어 인덱스로 잘라도 됨)"""
    parts, names = [], []
    if "p" in blocks:
        M, nm = pair_matrix(df, gene_cols)
        if log1p:
            M = M.copy(); M.data = np.log1p(M.data)
        parts.append(M); names += nm
    if "m" in blocks:
        M, nm = marg_matrix(df, gene_cols)
        if log1p:
            M = M.copy(); M.data = np.log1p(M.data)
        parts.append(M); names += nm
    if "s" in blocks:
        M, nm = s_matrix(df, gene_cols); parts.append(M); names += nm
    X = sparse.hstack(parts, format="csr") if parts else None
    return X, names
