"""
트랙 A — 변이 표기(mutation notation) 활용 피처
================================================

Owner : member_d (iljun)
Track : A
Spec  : A_v1_variant_type
기준선: member-d-logreg-001 · Macro F1 0.36305 (유전자 이진화만)

----------------------------------------------------------------------
설계 철학
----------------------------------------------------------------------
1) **행 안에서 끝낸다.**
   모든 변환은 한 샘플 내부 연산이다. 열 평균·분산·빈도 같은
   "행 사이 통계"를 쓰지 않는다. 그래서 부분집합 불변성이 성립하고
   (앞 100행만 변환 == 전체의 앞 100행), 단일 행 추론이 가능하며,
   Leakage 가 구조적으로 불가능하다. 나중에 API 로 뽑을 때도 그대로 쓴다.

2) **fit 이 필요한 것은 spec 하나뿐.**
   상수열 제거 인덱스(keep_idx)만 학습 대상이다. fold 안에서
   fit_spec(train_fold) 로 다시 만들어 넣으면 CV 가 정직해진다.

3) **블록으로 쪼갠다.**
   G(유전자) / B(부담) / V(유형 카운트) / R(유형 비율) 을 따로 두어
   ablation 으로 기여도를 분리 측정한다. 합쳐놓으면 뭐가 효과인지 모른다.

4) **팀 표준안으로 파싱을 고정한다.**
   같은 데이터를 두 사람이 다르게 파싱하면 트랙 병합에서 피처가 안 맞는다.
   - 규칙 ① 한 칸 안의 중복 토큰은 1개로 센다 ("R248Q R248Q" -> 1건).
     전사체가 여러 개라 중복 표기된 경우가 6,100건. 세면 카운트가 부풀려진다.
   - 규칙 ② 판정 순서 고정: ">" -> "fs" -> "del"/"ins" -> 나머지.
     정규식은 ^([A-Z*]+)(\\d+)(.*)$ . 앞 아미노산을 1글자로만 받으면
     TP469fs 같은 두 글자 접두 1,295건이 통째로 빠진다(실제로 겪은 버그).

----------------------------------------------------------------------
배제한 것과 이유  ← 나중의 나와 팀원이 같은 삽질을 반복하지 않도록
----------------------------------------------------------------------
* **열 단위 표준화(StandardScaler)** — 열 평균/표준편차는 행 사이 통계다.
  fold 안에서 fit 하면 쓸 수는 있으나, 단일 행 추론이 깨지고 대회 규정상
  전처리 spec 이 하나 더 늘어난다. 희소 이진 행렬이라 이득도 거의 없다.
  log1p 로 스케일을 눌러 대체한다.

* **test 를 본 상수열 제거 / 어휘 사전** — 대회 규정 위반. spec 은
  오로지 train(또는 train fold)만으로 만든다. test 는 컬럼 정합성과
  결측 유무 같은 '구조 점검'까지만 본다.

* **per-sample 비율의 중앙값 집계** — 저-TMB 샘플에서 2/3, 0/3 처럼
  이산화되어 클래스 대푯값이 왜곡된다(LAML silent 가 0.0% 로 읽혔던 사고).
  비율이 필요하면 클래스 단위로 '합산 후 나눗셈' 한다.
  단, 피처로 쓰는 R 블록은 샘플 개별 비율이므로 이 함정과 무관하다.

* **dense ndarray** — 4,384 열 × 6,201 행 float32 = 105 MB, fit 27.8 s.
  CSR 로 0.9 MB, 3.7 s. 결과는 완전히 동일하다. 무조건 sparse 를 쓴다.

* **변이 위치(숫자) 자체를 수치 피처로** — 468 이라는 위치값은 순서는
  있어도 거리가 의미 없다. 핫스팟은 별도 실험(11x/12x)에서 '특정
  (유전자, 변이) 조합의 이진 지시자'로 다룬다. 여기서는 유형만 본다.

* **indel 을 del/ins 로 분리** — 전체 3건. 분리해도 정보가 없다.

----------------------------------------------------------------------
사용법
----------------------------------------------------------------------
    from features.features_A import (
        KINDS, BLOCKS, parse_sample_counts, fit_spec, build_features,
    )

    gene_cols = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    cnt_train = parse_sample_counts(train, gene_cols)
    spec      = fit_spec(train, gene_cols, seed=42)
    X, names  = build_features(train, cnt_train, spec, blocks=("G", "B", "V", "R"))

`build_features` 는 (scipy.sparse.csr_matrix, list[str]) 를 반환한다 —
팀 규약 `build_features(df) -> (X, feature_names)` 와 같은 인터페이스이며
피처 이름에는 전부 `A_` 접두가 붙어 다른 트랙과 충돌하지 않는다.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

__version__ = "A_v1_variant_type"

# ----------------------------------------------------------------------
# 파서
# ----------------------------------------------------------------------
PATTERN = re.compile(r"^([A-Z*]+)(\d+)(.*)$")
KINDS = ["missense", "silent", "nonsense", "frameshift", "indel", "other"]
BURDEN = ["n_mut_genes", "n_events", "n_multi_genes"]
BLOCKS = ("G", "B", "V", "R")

WT = "WT"


def classify(token: str) -> str:
    """변이 토큰 하나를 기능적 유형으로 분류한다. 판정 순서가 중요하다."""
    if ">" in token:
        return "other"                        # 468_469LG>F* 복합 치환
    if "fs" in token:
        return "frameshift"
    if "del" in token or "ins" in token:
        return "indel"
    m = PATTERN.match(token)
    if not m:
        return "other"
    ref, _, alt = m.groups()
    if ref == "*" and alt == "*":
        return "other"                        # *261* 종결 -> 종결
    if alt == "*":
        return "nonsense"
    if alt == ref:
        return "silent"
    if alt == "":
        return "other"
    return "missense"


def parse_sample_counts(df: pd.DataFrame, gene_cols: list[str]) -> pd.DataFrame:
    """샘플별 [유형 카운트 6 + 부담 지표 3] 표를 만든다.

    전부 '한 행 안에서' 끝나므로 Leakage 가 아니다.
    """
    raw = df[gene_cols].fillna(WT).values
    mask = raw != WT
    n = len(df)
    out = np.zeros((n, len(KINDS) + len(BURDEN)), dtype=np.float32)
    for i in range(n):
        c = Counter()
        n_events = 0
        n_multi = 0
        for s in raw[i][mask[i]]:
            toks = set(s.split())             # <- 규칙 (1) 칸 내부 중복 제거
            if len(toks) > 1:
                n_multi += 1
            n_events += len(toks)
            for t in toks:
                c[classify(t)] += 1
        for j, k in enumerate(KINDS):
            out[i, j] = c[k]
        out[i, len(KINDS) + 0] = mask[i].sum()   # 변이 유전자 수
        out[i, len(KINDS) + 1] = n_events        # 총 이벤트 수
        out[i, len(KINDS) + 2] = n_multi         # 다중 이벤트 유전자 수
    return pd.DataFrame(out, columns=KINDS + BURDEN, index=df.index)


# ----------------------------------------------------------------------
# 전처리 spec
# ----------------------------------------------------------------------
def fit_spec(df_fit: pd.DataFrame, gene_cols: list[str], seed: int = 42) -> dict:
    """전처리 규칙을 학습한다. df_fit 에는 절대 test 를 넣지 않는다.

    학습되는 것은 '이 fit 범위에서 한 번도 변이가 없던 열'을 버리는
    keep_idx 하나뿐이다.
    """
    mask = df_fit[gene_cols].fillna(WT).values != WT
    return {
        "version": __version__,
        "gene_cols": list(gene_cols),
        "keep_idx": np.flatnonzero(mask.any(axis=0)).tolist(),
        "seed": seed,
    }


# ----------------------------------------------------------------------
# 피처 조립
# ----------------------------------------------------------------------
def build_features(df: pd.DataFrame, counts: pd.DataFrame, spec: dict,
                   blocks=BLOCKS):
    """blocks 에 지정된 블록만 이어붙인 CSR 행렬과 피처 이름을 반환한다.

    | 블록 | 내용                                       | 차원   |
    |------|--------------------------------------------|--------|
    | G    | 유전자 이진화 (fit 범위 상수열 제거)        | ~4,230 |
    | B    | 변이 부담 log1p (유전자수/이벤트수/다중)    | 3      |
    | V    | 유형별 카운트 log1p                         | 6      |
    | R    | 유형별 비율 (총 이벤트 대비)                | 6      |

    R 을 따로 두는 이유 — 카운트는 TMB 와 강하게 얽혀 있다(lift 9.96배).
    비율은 '변이가 많고 적고'를 지우고 구성만 남긴다. 둘이 다른 정보를
    담는지 ablation 으로 확인한다.
    """
    gene_cols = spec["gene_cols"]
    parts, names = [], []

    if "G" in blocks:
        keep = np.array(spec["keep_idx"], dtype=int)
        m = (df[gene_cols].fillna(WT).values != WT)[:, keep]
        parts.append(sparse.csr_matrix(m.astype(np.float32)))
        names += [f"A_gene__{gene_cols[i]}" for i in keep]

    if "B" in blocks:
        b = np.log1p(counts[BURDEN].values)
        parts.append(sparse.csr_matrix(b.astype(np.float32)))
        names += ["A_burden__log_genes", "A_burden__log_events", "A_burden__log_multi"]

    if "V" in blocks:
        v = np.log1p(counts[KINDS].values)
        parts.append(sparse.csr_matrix(v.astype(np.float32)))
        names += [f"A_vcount__{k}" for k in KINDS]

    if "R" in blocks:
        vals = counts[KINDS].values
        tot = vals.sum(axis=1, keepdims=True)
        r = np.divide(vals, tot, out=np.zeros_like(vals), where=tot > 0)
        parts.append(sparse.csr_matrix(r.astype(np.float32)))
        names += [f"A_vratio__{k}" for k in KINDS]

    if not parts:
        raise ValueError("blocks 가 비었습니다. G/B/V/R 중 하나 이상을 고르세요.")

    return sparse.hstack(parts, format="csr"), names


def spec_to_json(spec: dict, X_shape, blocks: str, feature_names: list[str]) -> dict:
    """artifacts 로 저장할 요약본. gene_cols 4,384개는 빼서 파일을 작게 유지한다."""
    return {
        "version": spec["version"],
        "seed": spec["seed"],
        "blocks": blocks,
        "n_features": int(X_shape[1]),
        "n_genes_kept": len(spec["keep_idx"]),
        "feature_names_head": feature_names[:5],
        "feature_names_tail": feature_names[-12:],
        "keep_idx": spec["keep_idx"],
    }
