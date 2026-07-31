"""
exp_005 · 코돈(잔기 위치) 핫스팟 — H 블록
==========================================

무엇인가
--------
변이 표기 `R132H` 는 [참조AA R][위치 132][대체AA H] 다. 정확토큰(exp_003 의 D
블록)은 `IDH1:R132H` 를 통째로 쓴다. 여기서는 **대체AA 를 버리고** `IDH1:R132`
로 묶는다. 그러면 같은 코돈의 서로 다른 치환(R132H · R132C · R132L)이 한
피처가 된다.

왜 홍주님(biodomain-01)과 안 겹치나
-----------------------------------
홍주님: 아미노산 20종 치환 '방향' + '고정' 위치구간 + 표기 복잡도를, 데이터·
라벨과 무관하게 고정한 규칙으로 '환자 단위 합계' 한다.
여기(H): 어느 (유전자, 코돈)이 fold-train 에서 재현되는지를 '데이터로 학습'해
그 특정 코돈의 존재를 '변이 단위'로 표시한다. 집계 단위·학습 여부·해상도가 다르다.

왜 내 정확토큰(D)과도 다른 축인가
---------------------------------
D 는 정확변이(대체AA 포함), H 는 코돈(대체AA 무시). H 는 D 가 쪼개던 핫스팟을
한 피처로 모아 더 조밀하고 재현적이다.

★ LB 격차 가설 (검증 대상)
---------------------------
정확토큰은 배치 아티팩트에 가장 취약하다 — 특정 정확변이 하나가 시퀀싱 배치
서명일 수 있다(CV 는 오르나 LB 는 안 오른다). 코돈으로 묶으면 재현되는 핫스팟만
남아 실제 생물학일 확률이 높다 → LB 로 더 잘 전달될 수 있다. CV 로는 못 가른다.
제출로만 판정한다.

규칙 준수
---------
· 핫스팟 사전은 외부 DB(COSMIC 등)가 아니라 fold-train 라벨에서만 학습한다(규칙 2).
· 외부 유전자-암종 지식을 모델 입력에 넣지 않는다(규칙 1).
· test 는 개발 중 열지 않는다. 제출 시 transform 만 적용한다.

선택·집계 로직은 features_D 의 fit_tokens / transform_tokens 를 그대로 재사용한다.
그래야 'D vs H' 비교가 오직 키의 해상도(정확변이 vs 코돈) 차이만 남는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
_EXP3 = _HERE.parent / "exp_003_discriminative_tokens"
for p in (_EXP2, _EXP3):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from features_A import classify, WT, PATTERN                          # noqa: E402
import features_D as fd                                               # noqa: E402

__version__ = "H_v1_codon_hotspots"

# 코돈 핫스팟은 '한 잔기의 점 치환'만 대상으로 한다.
# missense(R132H) · nonsense(R132*) 는 위치가 명확해 코돈으로 묶기 깨끗하다.
# frameshift · indel 은 V 블록(유형 카운트)이 이미 잡으므로 제외한다.
POINT = {"missense", "nonsense"}


def codon_of(gene: str, token: str):
    """functional 점 치환 하나 → (gene, 'R132') 코돈 키. 아니면 None."""
    if classify(token) not in POINT:
        return None
    m = PATTERN.match(token)
    if not m:
        return None
    ref, pos, _alt = m.groups()          # 대체AA(_alt)를 버려 같은 코돈을 묶는다
    return (gene, f"{ref}{pos}")


def parse_codon_sets(df, gene_cols):
    """환자마다 (유전자, 코돈) 키의 집합을 만든다. D 의 parse_token_sets 와 같은
    형태(길이 n 리스트, 원소는 frozenset)라 fd.fit_tokens/transform_tokens 에
    그대로 넣을 수 있다."""
    raw = df[gene_cols].fillna(WT).values
    mask = raw != WT
    gcols = list(gene_cols)
    out = []
    for i in range(len(df)):
        keys = set()
        for j in np.flatnonzero(mask[i]):
            g = gcols[j]
            for v in set(raw[i][j].split()):     # 칸 내부 중복 제거(팀 표준)
                k = codon_of(g, v)
                if k is not None:
                    keys.add(k)
        out.append(frozenset(keys))
    return out


def fit_codons(codon_sets, y, idx_train, top_k, min_count=10, method="freq"):
    """fold-train 에서 코돈 핫스팟을 고른다. D 의 fit_tokens 를 재사용한다.
    (freq = 재현수 순. exp_003 에서 freq 가 disc 를 이겼으므로 기본 freq.)"""
    return fd.fit_tokens(codon_sets, y, idx_train, top_k,
                         min_count=min_count, method=method)


def transform_codons(codon_sets, idx, spec, k):
    """상위 k 코돈의 존재 여부를 이진 CSR 로. 이름만 H_hot__ 로 바꾼다."""
    M, names = fd.transform_tokens(codon_sets, idx, spec, k)
    names = [n.replace("D_tok__", "H_hot__") for n in names]
    return M, names
