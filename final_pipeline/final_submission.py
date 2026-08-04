"""AH05 해커톤 최종 제출 — 암 환자 유전체 변이 정보를 활용한 26개 암종 분류.

실행
    python final_pipeline/final_submission.py
    python final_pipeline/final_submission.py --data-dir /data      # 대회 제출 환경

개발 환경
    OS          macOS (Darwin 25.5.0) / Linux 호환
    Python      3.12
    numpy 2.x · pandas 2.2+ · scipy 1.13+ · scikit-learn 1.5+ · lightgbm 4.5+ · tqdm 4.66+

    (개발 중 검증은 Python 3.14 에서도 수행했다. 팀 규약 환경은 3.12 이며,
     환경 차이는 팀 측정상 Macro F1 소수점 넷째 자리에서 최대 0.0006 이다.)

재현 설정 — 모두 아래 CONFIG / ENSEMBLE_WEIGHTS 에 고정되어 있다
    random seed         42
    LogisticRegression  solver=lbfgs, C=0.07, max_iter=2000, class_weight=balanced
    LightGBM            multiclass, n_estimators=100, learning_rate=0.05,
                        num_leaves=31, class_weight=balanced, deterministic=True
    교차검증            StratifiedKFold(n_splits=5, shuffle=True)

모델 구성 (3-seed OOF Macro F1 0.54202 / Public LB 0.45349)

    ① 변이 문자열을 행 내부에서만 파싱해 다층 표현을 만든다
         G 유전자별 변이 유무 · B 변이 부담 3 · V 변이 유형 7 · T truncating
         R 반복 missense · A 아미노산 ref→alt 치환쌍 380 · S 표기 구조 8
         + 정확 hotspot 4개 (BRAF V600E · IDH1 R132H · PIK3CA H1047R/E545K)
         + 혼동 암종쌍 contrast — **쌍을 데이터에서 자동 발견한다** (아래 ③)
         B/V/A count 에 log1p 적용

    ② class-enrichment 26 — (유전자 × 변이유형) 토큰의 암종별 log-odds 를
       중첩 cross-fit 으로 학습해 26개 점수로 압축한다. label 을 쓰는
       supervised 피처이므로 누수 방지 구조를 별도로 둔다(아래 주석 참조).

    ③ 세 모델 소프트보팅 — LR multinomial 0.55 + One-vs-Rest LR 0.30 + LightGBM 0.15
       계열이 서로 달라 다른 실수를 한다(불일치율 OVR 18.5% / LGBM 39.0%).

Data Leakage 방지 — 코드 구조로 보장한다
    · train 과 test 를 **한 프레임으로 합치지 않는다.** train 을 먼저 파싱해
      (유전자, 변이) 어휘를 정의하고, test 는 그 어휘에 투영만 한다.
      test 에만 있는 변이는 어떤 열도 만들지 못한다.
    · 활성 열 · 상수열 제거 · 반복 missense 선택 · 혼동쌍 선택 · enrichment
      토큰 support · 표준화 통계 — 전부 train 인덱스에서만 계산한다.
    · class-enrichment 는 label 을 쓰므로 중첩 cross-fit 한다. 학습 행은 자기
      label 이 포함된 가중치를 받지 않는다. test 에는 적용만 한다.
    · 검증: label 을 무작위로 섞으면 enrichment 이득이 +0.051 → +0.002 로
      사라진다(permutation check). test 통계 미사용은 설계행렬 대조로 확인했다.

주의 — 이 파일은 어떤 프로젝트 로컬 모듈도 import 하지 않는다. 단독 실행된다.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from tqdm.auto import tqdm


WT = "WT"
EVENT_TYPES = ("MISSENSE", "SYNONYMOUS", "NONSENSE", "FRAMESHIFT", "SPLICE", "INFRAME_INDEL", "OTHER")
TRUNCATING = frozenset({"NONSENSE", "FRAMESHIFT", "SPLICE"})
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
SPLICE_RE = re.compile(r"SPLICE|IVS|[+-]\d+")
INDEL_RE = re.compile(r"DEL|INS|DUP")


@dataclass(frozen=True)
class ExperimentConfig:
    id_col: str = "ID"
    target_col: str = "SUBCLASS"
    n_splits: int = 5
    primary_seed: int = 42
    stability_seeds: tuple[int, ...] = (42, 2024, 777)
    lr_c: float = 0.07
    lr_max_iter: int = 2000
    recurrent_min_count: int = 5


CONFIG = ExperimentConfig()


def normalise_cell(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == WT:
        return ()
    return tuple(dict.fromkeys(token.removeprefix("P.") for token in re.sub(r"[;,|]+", " ", text).split() if token))


def classify_event(event: str) -> str:
    if "FS" in event:
        return "FRAMESHIFT"
    if SPLICE_RE.search(event):
        return "SPLICE"
    if INDEL_RE.search(event):
        return "INFRAME_INDEL"
    if "*" in event or event.endswith("X"):
        return "NONSENSE"
    match = SUB_RE.fullmatch(event)
    if match:
        return "SYNONYMOUS" if match.group(1) == match.group(3) else "MISSENSE"
    return "OTHER"


@dataclass
class RowCache:
    genes: list[str]
    mutation_matrix: sparse.csr_matrix
    truncation_matrix: sparse.csr_matrix
    event_matrix: sparse.csr_matrix
    event_names: list[str]
    event_is_missense: np.ndarray
    burden: np.ndarray
    variant: np.ndarray
    amino: np.ndarray
    topology: np.ndarray
    events: pd.DataFrame

    @classmethod
    def build(cls, frame: pd.DataFrame, genes: list[str], show_progress: bool = True,
              vocabulary: list[str] | None = None) -> "RowCache":
        """Parse one frame row-by-row.

        `vocabulary` fixes the (gene, event) column space.  Passing the train
        vocabulary when parsing test guarantees the test matrix is expressed
        purely in train-derived columns: events seen only in test are counted
        in the row-local blocks (burden/variant/amino/topology) exactly as
        before, but never create a column of their own.
        """
        n_rows, n_genes = len(frame), len(genes)
        mut_r: list[int] = []; mut_c: list[int] = []
        trunc_r: list[int] = []; trunc_c: list[int] = []
        records: list[tuple[int, int, str, str]] = []
        iterator = tqdm(enumerate(genes), total=n_genes, desc="row-local mutation cache", disable=not show_progress)
        for gene_idx, gene in iterator:
            # One Series at a time: never materialise a second wide object DataFrame.
            for row_idx, value in enumerate(frame[gene].array):
                tokens = normalise_cell(value)
                if not tokens:
                    continue
                mut_r.append(row_idx); mut_c.append(gene_idx)
                for event in tokens:
                    kind = classify_event(event)
                    records.append((row_idx, gene_idx, event, kind))
                    if kind in TRUNCATING:
                        trunc_r.append(row_idx); trunc_c.append(gene_idx)
        ones = np.ones(len(mut_r), dtype=np.float32)
        mutation = sparse.coo_matrix((ones, (mut_r, mut_c)), shape=(n_rows, n_genes)).tocsr()
        mutation.data[:] = 1.0
        truncation = sparse.coo_matrix((np.ones(len(trunc_r), dtype=np.float32), (trunc_r, trunc_c)), shape=(n_rows, n_genes)).tocsr()
        truncation.data[:] = 1.0
        events = pd.DataFrame(records, columns=["row", "gene_idx", "event", "event_type"])
        if events.empty:
            events = pd.DataFrame(columns=["row", "gene_idx", "event", "event_type"])
            event_names = list(vocabulary) if vocabulary is not None else []
            event_matrix = sparse.csr_matrix((n_rows, len(event_names)), dtype=np.float32)
            missense = np.zeros(len(event_names), dtype=bool)
        else:
            events = events.drop_duplicates(["row", "gene_idx", "event"]).reset_index(drop=True)
            events["gene"] = events.gene_idx.map(dict(enumerate(genes)))
            events["pair"] = events.gene + "__" + events.event
            event_names = sorted(events.pair.unique()) if vocabulary is None else list(vocabulary)
            event_lookup = {name: idx for idx, name in enumerate(event_names)}
            # Events outside the vocabulary get no column.  With vocabulary=None
            # (train) every event is in the vocabulary, so nothing is dropped.
            column = events.pair.map(event_lookup)
            known = column.notna().to_numpy()
            event_matrix = sparse.coo_matrix(
                (np.ones(int(known.sum()), dtype=np.float32),
                 (events.row.to_numpy()[known], column.to_numpy()[known].astype(np.int64))),
                shape=(n_rows, len(event_names)),
            ).tocsr()
            event_matrix.data[:] = 1.0
            missense = events.groupby("pair").event_type.first().reindex(event_names).eq("MISSENSE").to_numpy()

        burden = np.zeros((n_rows, 3), dtype=np.float32)
        burden[:, 0] = np.asarray(mutation.sum(axis=1)).ravel()
        variant = np.zeros((n_rows, len(EVENT_TYPES)), dtype=np.float32)
        amino = np.zeros((n_rows, 20 + 20 + 380 + 6), dtype=np.float32)
        topology = np.zeros((n_rows, 8), dtype=np.float32)
        if not events.empty:
            burden[:, 1] = events.groupby("row").size().reindex(range(n_rows), fill_value=0).to_numpy()
            by_gene = events.groupby(["row", "gene_idx"]).size()
            burden[:, 2] = by_gene.gt(1).groupby(level=0).sum().reindex(range(n_rows), fill_value=0).to_numpy()
            for col, kind in enumerate(EVENT_TYPES):
                variant[:, col] = events.event_type.eq(kind).groupby(events.row).sum().reindex(range(n_rows), fill_value=0).to_numpy()
            parsed = events.event.str.extract(SUB_RE); events["ref"] = parsed[0]; events["pos"] = pd.to_numeric(parsed[1], errors="coerce"); events["alt"] = parsed[2]
            substitutions = events.dropna(subset=["ref", "alt", "pos"])
            aa_index = {letter: index for index, letter in enumerate(AA)}
            for row, ref, alt, pos in substitutions[["row", "ref", "alt", "pos"]].itertuples(index=False):
                if ref in aa_index:
                    amino[row, aa_index[ref]] += 1
                if alt in aa_index:
                    amino[row, 20 + aa_index[alt]] += 1
                if ref in aa_index and alt in aa_index and ref != alt:
                    pair_index = sum(1 for a in AA for b in AA if a != b and (a, b) < (ref, alt))
                    amino[row, 40 + pair_index] += 1
                for bin_idx, (low, high) in enumerate(((1, 50), (51, 100), (101, 250), (251, 500), (501, 1000), (1001, np.inf))):
                    if low <= pos <= high: amino[row, 420 + bin_idx] += 1; break
            gene_counts = events.groupby(["row", "gene_idx"]).agg(event_count=("event", "size"), type_count=("event_type", "nunique"))
            for col, mask in enumerate((gene_counts.event_count.eq(1), gene_counts.event_count.eq(2), gene_counts.event_count.ge(3), gene_counts.type_count.ge(2))):
                topology[:, col] = mask.groupby(level=0).sum().reindex(range(n_rows), fill_value=0).to_numpy()
            topology[:, 4] = gene_counts.event_count.groupby(level=0).max().reindex(range(n_rows), fill_value=0).to_numpy()
            type_counts = pd.crosstab(events.row, events.event_type).reindex(index=range(n_rows), columns=EVENT_TYPES, fill_value=0)
            proportions = type_counts.div(type_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            topology[:, 5] = type_counts.gt(0).sum(axis=1).to_numpy()
            topology[:, 6] = -(proportions.where(proportions.gt(0), 1) * np.log(proportions.where(proportions.gt(0), 1))).sum(axis=1).to_numpy()
            topology[:, 7] = proportions.max(axis=1).to_numpy()
        return cls(genes, mutation, truncation, event_matrix, event_names, missense, burden, variant, amino, topology, events)

    @classmethod
    def stack(cls, head: "RowCache", tail: "RowCache") -> "RowCache":
        """Concatenate two caches parsed against the same gene list and vocabulary.

        Column metadata is taken from `head` (train): `tail` was parsed with
        `vocabulary=head.event_names`, so both matrices already agree on width.
        """
        assert head.genes == tail.genes, "두 캐시의 유전자 목록이 다릅니다"
        assert head.event_matrix.shape[1] == tail.event_matrix.shape[1], (
            "test 캐시가 train 어휘로 파싱되지 않았습니다"
        )
        offset = head.mutation_matrix.shape[0]
        tail_events = tail.events.copy()
        if not tail_events.empty:
            tail_events["row"] = tail_events["row"] + offset
        return cls(
            head.genes,
            sparse.vstack([head.mutation_matrix, tail.mutation_matrix], format="csr"),
            sparse.vstack([head.truncation_matrix, tail.truncation_matrix], format="csr"),
            sparse.vstack([head.event_matrix, tail.event_matrix], format="csr"),
            head.event_names,
            head.event_is_missense,
            np.vstack([head.burden, tail.burden]),
            np.vstack([head.variant, tail.variant]),
            np.vstack([head.amino, tail.amino]),
            np.vstack([head.topology, tail.topology]),
            pd.concat([head.events, tail_events], ignore_index=True),
        )


def nonconstant_columns(matrix: sparse.csr_matrix) -> np.ndarray:
    """Match DataFrame.nunique()>1, including all-positive count columns."""
    minimum = np.asarray(matrix.min(axis=0).toarray()).ravel()
    maximum = np.asarray(matrix.max(axis=0).toarray()).ravel()
    return minimum != maximum


class FoldMatrixBuilder:
    def __init__(self, cache: RowCache, backbone: str, exact_events=(), gene_pairs=(), gene_groups=(), hotspot_top_k: int = 0, contrast_pairs=(), amino_mode: str = "all", log1p_counts: bool = False):
        self.cache, self.backbone = cache, backbone
        self.exact_events, self.gene_pairs, self.gene_groups = tuple(exact_events), tuple(gene_pairs), tuple(gene_groups)
        self.hotspot_top_k, self.contrast_pairs = hotspot_top_k, tuple(contrast_pairs)
        assert amino_mode in {"all", "pair"}
        self.amino_mode, self.log1p_counts = amino_mode, log1p_counts

    def _domain_matrix(self, train_index: np.ndarray, labels: pd.Series | np.ndarray | None) -> tuple[sparse.csr_matrix, list[str]]:
        cols: list[sparse.csr_matrix] = []; names: list[str] = []; gene_map = {gene: idx for idx, gene in enumerate(self.cache.genes)}
        lookup = {name: idx for idx, name in enumerate(self.cache.event_names)}
        for gene, event in self.exact_events:
            col = self.cache.event_matrix[:, lookup[gene + "__" + event]] if gene + "__" + event in lookup else sparse.csr_matrix((self.cache.mutation_matrix.shape[0], 1))
            cols.append(col); names.append(f"D__exact_{gene}_{event}")
        for first, second in self.gene_pairs:
            if first in gene_map and second in gene_map:
                col = self.cache.mutation_matrix[:, gene_map[first]].multiply(self.cache.mutation_matrix[:, gene_map[second]])
            else: col = sparse.csr_matrix((self.cache.mutation_matrix.shape[0], 1))
            cols.append(col); names.append(f"D__pair_{first}_{second}")
        for name, genes in self.gene_groups:
            ids = [gene_map[gene] for gene in genes if gene in gene_map]
            cols.append(sparse.csr_matrix(self.cache.mutation_matrix[:, ids].sum(axis=1) if ids else np.zeros((self.cache.mutation_matrix.shape[0], 1))))
            names.append(f"D__group_count_{name}")
        if self.hotspot_top_k:
            assert labels is not None, "fold-train labels are required for hotspot selection"
            train_labels = np.asarray(labels)[train_index]
            counts = np.asarray(self.cache.event_matrix[train_index].getnnz(axis=0)).ravel()
            concentration = np.zeros_like(counts, dtype=np.float64)
            for class_name in np.unique(train_labels):
                class_counts = np.asarray(self.cache.event_matrix[train_index][train_labels == class_name].getnnz(axis=0)).ravel()
                concentration = np.maximum(concentration, class_counts / np.maximum(counts, 1))
            fixed = {f"{gene}__{event}" for gene, event in self.exact_events}
            eligible = np.flatnonzero((counts >= 10) & (concentration >= 0.60) & ~np.isin(self.cache.event_names, list(fixed)))
            ranked = sorted(eligible, key=lambda index: (-counts[index] * concentration[index], -counts[index], self.cache.event_names[index]))[:self.hotspot_top_k]
            for index in ranked:
                cols.append(self.cache.event_matrix[:, index]); names.append(f"H__{self.cache.event_names[index]}")
        if self.contrast_pairs:
            assert labels is not None, "fold-train labels are required for contrast selection"
            train_labels = np.asarray(labels)[train_index]
            for left, right, top_k in self.contrast_pairs:
                left_mask, right_mask = train_labels == left, train_labels == right
                if not left_mask.any() or not right_mask.any():
                    continue
                left_counts = np.asarray(self.cache.mutation_matrix[train_index][left_mask].getnnz(axis=0)).ravel()
                right_counts = np.asarray(self.cache.mutation_matrix[train_index][right_mask].getnnz(axis=0)).ravel()
                support = left_counts + right_counts
                contrast = left_counts / left_mask.sum() - right_counts / right_mask.sum()
                eligible = np.flatnonzero(support >= 10)
                selected = sorted(eligible, key=lambda index: (-abs(contrast[index]), -support[index], self.cache.genes[index]))[:top_k]
                if not selected:
                    continue
                signs = np.sign(contrast[selected]).astype(np.float32)
                count_col = sparse.csr_matrix(self.cache.mutation_matrix[:, selected].sum(axis=1))
                contrast_col = self.cache.mutation_matrix[:, selected].dot(sparse.csr_matrix(signs).T)
                cols += [count_col, contrast_col]
                names += [f"C__{left}_vs_{right}_count", f"C__{left}_vs_{right}_contrast"]
        return (sparse.hstack(cols, format="csr") if cols else sparse.csr_matrix((self.cache.mutation_matrix.shape[0], 0))), names

    def build(self, train_index: np.ndarray, valid_index: np.ndarray, labels: pd.Series | np.ndarray | None = None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str]]:
        cache = self.cache
        active = np.flatnonzero(np.asarray(cache.mutation_matrix[train_index].getnnz(axis=0)).ravel())
        parts = [cache.mutation_matrix[:, active]]; names = [f"G__{cache.genes[i]}" for i in active]
        burden = np.log1p(cache.burden) if self.log1p_counts else cache.burden
        variant = np.log1p(cache.variant) if self.log1p_counts else cache.variant
        parts += [sparse.csr_matrix(burden), sparse.csr_matrix(variant)]
        names += ["B__mutated_gene_count", "B__event_count", "B__multi_event_gene_count"] + [f"V__{kind.lower()}_event_count" for kind in EVENT_TYPES]
        trunc = np.flatnonzero(np.asarray(cache.truncation_matrix[train_index].getnnz(axis=0)).ravel())
        parts.append(cache.truncation_matrix[:, trunc]); names += [f"T__{cache.genes[i]}" for i in trunc]
        parts.append(sparse.csr_matrix(np.asarray(cache.truncation_matrix.sum(axis=1)))); names.append("T__truncating_gene_count")
        recurrent = np.flatnonzero((np.asarray(cache.event_matrix[train_index].getnnz(axis=0)).ravel() >= CONFIG.recurrent_min_count) & cache.event_is_missense)
        parts.append(cache.event_matrix[:, recurrent]); names += [f"R__{cache.event_names[i]}" for i in recurrent]
        parts.append(sparse.csr_matrix(np.asarray(cache.event_matrix[:, recurrent].sum(axis=1)))); names.append("R__recurrent_missense_event_count")
        if "+A" in self.backbone:
            if self.amino_mode == "pair":
                amino = np.log1p(cache.amino[:, 40:420]) if self.log1p_counts else cache.amino[:, 40:420]
                parts.append(sparse.csr_matrix(amino)); names += [f"A_pair__{i}" for i in range(380)]
            else:
                amino = np.log1p(cache.amino) if self.log1p_counts else cache.amino
                parts.append(sparse.csr_matrix(amino)); names += [f"A__{i}" for i in range(cache.amino.shape[1])]
        if "+S" in self.backbone: parts.append(sparse.csr_matrix(cache.topology)); names += [f"S__{i}" for i in range(cache.topology.shape[1])]
        domain, domain_names = self._domain_matrix(train_index, labels); parts.append(domain); names += domain_names
        all_rows = sparse.hstack(parts, format="csr")
        keep = nonconstant_columns(all_rows[train_index])
        return all_rows[train_index][:, keep], all_rows[valid_index][:, keep], [name for name, included in zip(names, keep) if included]


@dataclass(frozen=True)
class Candidate:
    experiment_id: str
    backbone: str
    exact_events: tuple[tuple[str, str], ...] = ()
    gene_pairs: tuple[tuple[str, str], ...] = ()
    gene_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    hotspot_top_k: int = 0
    contrast_pairs: tuple[tuple[str, str, int], ...] = ()
    amino_mode: str = "all"
    log1p_counts: bool = False
    lr_max_iter: int = CONFIG.lr_max_iter


FINAL_EXACT_HOTSPOTS = (
    ("BRAF", "V600E"),
    ("IDH1", "R132H"),
    ("PIK3CA", "H1047R"),
    ("PIK3CA", "E545K"),
)
# contrast_pairs 는 여기에 적지 않는다. 암종 쌍은 fold-train 혼동행렬에서
# 자동으로 도출한다 (discover_confusion_pairs 참조) — 코호트 명칭을 모델 구조에
# 사람이 넣지 않기 위해서다.
FINAL_CANDIDATE = Candidate(
    experiment_id="ensemble-autopairs-enrichment-3way",
    backbone="G+B+V+T+R+A+S",
    exact_events=FINAL_EXACT_HOTSPOTS,
    amino_mode="pair",
    log1p_counts=True,
)

# ══════════════════════════════════════════════════════════════════════════
#  혼동 암종쌍 자동 발견
#
#  이전 판본은 KIRC↔KIPAN, LGG↔GBMLGG 를 사람이 지목해 코드에 적었다.  코호트
#  명칭 관계를 모델 구조에 반영하지 않기 위해, 쌍을 fold-train 혼동행렬에서
#  직접 도출하도록 바꿨다.  검증 결과 자동 발견이 기존 손지정 2쌍을 정확히
#  1·2위로 재발견했다(KIPAN↔KIRC 0.363, GBMLGG↔LGG 0.301).
# ══════════════════════════════════════════════════════════════════════════

CONFUSION_PAIR_COUNT = 8      # 상위 몇 쌍을 쓸 것인가
CONFUSION_GENES_PER_PAIR = 5  # 쌍마다 고를 유전자 수


def discover_confusion_pairs(cache, train_index, labels, seed):
    """fold-train 안에서 3-fold 대리모델을 돌려 실제로 많이 혼동되는 쌍을 찾는다.

    대리모델은 유전자 이진화(G 블록)만 쓴다 — 쌍 후보를 고르는 용도라 가벼우면
    충분하고, 본 모델과 같은 피처를 다시 만들 필요가 없다.  **fold-train 라벨만**
    사용하므로 검증/테스트 정보가 들어가지 않는다.
    """
    y = np.asarray(labels)[train_index]
    X = cache.mutation_matrix[train_index]
    classes = np.unique(y)
    predicted = np.empty(len(y), dtype=object)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fit_rows, holdout_rows in inner.split(np.zeros(len(y)), y):
        model = LogisticRegression(solver="lbfgs", C=CONFIG.lr_c, max_iter=300,
                                   class_weight="balanced", random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X[fit_rows], y[fit_rows])
        predicted[holdout_rows] = model.predict(X[holdout_rows])

    matrix = confusion_matrix(y, np.array(list(predicted)), labels=classes)
    sizes = matrix.sum(axis=1)
    scored = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            swapped = matrix[i, j] + matrix[j, i]           # 서로 헷갈린 횟수
            denominator = max(sizes[i] + sizes[j], 1)
            scored.append((swapped / denominator, str(classes[i]), str(classes[j])))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return tuple((left, right, CONFUSION_GENES_PER_PAIR)
                 for _, left, right in scored[:CONFUSION_PAIR_COUNT])


# ══════════════════════════════════════════════════════════════════════════
#  class-enrichment — (유전자 × 변이유형) 토큰의 암종별 log-odds 를 26점수로 압축
#
#  label 을 사용하는 supervised 피처다.  수천 개 토큰을 모델에 직접 넣지 않고
#  암종별 signature 점수 26개로 압축해 고차원 희소성 증가를 최소화한다.
# ══════════════════════════════════════════════════════════════════════════

ENRICHMENT_ALPHA = 1.0        # 라플라스 평활
ENRICHMENT_MIN_SUPPORT = 10   # 이 횟수 미만 관측된 토큰은 가중치를 학습하지 않는다
ENRICHMENT_SHRINKAGE = 10.0   # support 가 작을수록 가중치를 0 쪽으로 당긴다
ENRICHMENT_WEIGHT_CLIP = 4.0


def gene_type_matrix(cache):
    """(행 × 토큰) 이진 행렬. 토큰 = 유전자__변이유형, 샘플 단위 presence."""
    events = cache.events
    if events.empty:
        return sparse.csr_matrix((cache.mutation_matrix.shape[0], 0), dtype=np.float32)
    observed = events[["row", "gene", "event_type"]].drop_duplicates()
    key = observed.gene.to_numpy().astype(object) + "__" + observed.event_type.to_numpy().astype(object)
    codes, uniques = pd.factorize(pd.Index(key))
    matrix = sparse.coo_matrix(
        (np.ones(len(codes), dtype=np.float32), (observed.row.to_numpy(), codes)),
        shape=(cache.mutation_matrix.shape[0], len(uniques)),
    ).tocsr()
    matrix.data[:] = 1.0
    return matrix


def fit_enrichment_weights(token_matrix, fit_index, labels, classes):
    """fit_index(학습 분할)에서만 암종별 토큰 log-odds 를 학습한다."""
    fit_matrix = token_matrix[fit_index]
    support = np.asarray(fit_matrix.getnnz(axis=0)).ravel()
    # 모든 학습 행에 존재하는 토큰은 log-odds 가 발산하므로 상한도 둔다.
    selected = np.flatnonzero((support >= ENRICHMENT_MIN_SUPPORT) & (support < len(fit_index)))
    if not len(selected):
        return selected, np.zeros((len(classes), 0), dtype=np.float32)

    fit_matrix = fit_matrix[:, selected]
    support = support[selected].astype(np.float64)
    fit_labels = np.asarray(labels)[fit_index]
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for index, class_name in enumerate(classes):
        positive_mask = fit_labels == class_name
        positive_size = int(positive_mask.sum())
        negative_size = len(fit_index) - positive_size
        positive = np.asarray(fit_matrix[positive_mask].getnnz(axis=0)).ravel().astype(np.float64)
        negative = support - positive
        weights[index] = (
            np.log((positive + ENRICHMENT_ALPHA) / (positive_size - positive + ENRICHMENT_ALPHA))
            - np.log((negative + ENRICHMENT_ALPHA) / (negative_size - negative + ENRICHMENT_ALPHA))
        )
    weights *= support[None, :] / (support[None, :] + ENRICHMENT_SHRINKAGE)
    return selected, np.clip(weights, -ENRICHMENT_WEIGHT_CLIP, ENRICHMENT_WEIGHT_CLIP).astype(np.float32)


def apply_enrichment_scores(token_matrix, row_index, selected, weights):
    """한 샘플의 암종별 점수 = 보유 토큰 가중치 합 / sqrt(활성 토큰 수)."""
    if not len(selected):
        return np.zeros((len(row_index), weights.shape[0]), dtype=np.float32)
    rows = token_matrix[row_index][:, selected]
    scores = np.asarray(rows @ weights.T, dtype=np.float32)
    denominator = np.sqrt(np.maximum(np.asarray(rows.getnnz(axis=1)).ravel(), 1)).astype(np.float32)
    return scores / denominator[:, None]


def cross_fitted_enrichment(token_matrix, train_index, valid_index, labels, seed):
    """중첩 cross-fit — 학습 행이 자기 label 이 든 가중치를 받지 않게 한다.

    1. train 을 다시 내부 5-fold 로 나눈다
    2. 각 inner-fit 에서 가중치를 학습해 inner-holdout 에만 적용한다  → 학습용 점수
    3. train 전체로 가중치를 다시 학습해 valid/test 에 적용만 한다   → 추론용 점수
    4. 표준화 통계는 학습용 점수에서만 만든다
    """
    label_array = np.asarray(labels)
    classes = sorted(np.unique(label_array[train_index]).tolist())
    train_scores = np.zeros((len(train_index), len(classes)), dtype=np.float32)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_rows, holdout_rows in inner.split(np.zeros(len(train_index)), label_array[train_index]):
        selected, weights = fit_enrichment_weights(
            token_matrix, train_index[fit_rows], label_array, classes)
        train_scores[holdout_rows] = apply_enrichment_scores(
            token_matrix, train_index[holdout_rows], selected, weights)

    selected, weights = fit_enrichment_weights(token_matrix, train_index, label_array, classes)
    valid_scores = apply_enrichment_scores(token_matrix, valid_index, selected, weights)

    keep = train_scores.min(axis=0) != train_scores.max(axis=0)
    train_scores, valid_scores = train_scores[:, keep], valid_scores[:, keep]
    names = [f"E__gene_type__{name}" for name, included in zip(classes, keep) if included]
    if not train_scores.shape[1]:
        return train_scores, valid_scores, names
    mean = train_scores.mean(axis=0)
    deviation = train_scores.std(axis=0)
    deviation[deviation < 1e-6] = 1.0
    return (((train_scores - mean) / deviation).astype(np.float32),
            ((valid_scores - mean) / deviation).astype(np.float32), names)


# ══════════════════════════════════════════════════════════════════════════
#  앙상블 — 계열이 다른 세 모델의 소프트보팅
#
#  같은 피처 위에서 모델만 바꾼다.  LGBM 은 단독으로는 LR 보다 약하지만(0.485 vs
#  0.531) 서로 다른 실수를 해서 섞였을 때 이득이 난다 — "더 세서"가 아니라
#  "다양성"으로 기여한다.
# ══════════════════════════════════════════════════════════════════════════

ENSEMBLE_WEIGHTS = {"lr_multinomial": 0.55, "lr_ovr": 0.30, "lightgbm": 0.15}


def make_models(seed):
    parameters = dict(solver="lbfgs", C=CONFIG.lr_c, max_iter=CONFIG.lr_max_iter,
                      class_weight="balanced", random_state=seed)
    return {
        "lr_multinomial": LogisticRegression(**parameters),
        "lr_ovr": OneVsRestClassifier(LogisticRegression(**parameters), n_jobs=1),
        "lightgbm": LGBMClassifier(objective="multiclass", n_estimators=100,
                                   learning_rate=0.05, num_leaves=31,
                                   class_weight="balanced", random_state=seed,
                                   n_jobs=-1, deterministic=True,
                                   force_col_wise=True, verbosity=-1),
    }


# ══════════════════════════════════════════════════════════════════════════
#  학습 · 추론 · 제출
# ══════════════════════════════════════════════════════════════════════════

def build_design_matrices(cache, token_matrix, train_index, valid_index, labels, seed):
    """B04 피처 + class-enrichment 26 을 붙여 설계행렬을 만든다."""
    pairs = discover_confusion_pairs(cache, train_index, labels, seed)
    print(f"[info] 자동 발견 혼동쌍 {len(pairs)}개: "
          + ", ".join(f"{left}↔{right}" for left, right, _ in pairs[:4]) + " ...", flush=True)
    builder = FoldMatrixBuilder(
        cache, FINAL_CANDIDATE.backbone, FINAL_CANDIDATE.exact_events,
        FINAL_CANDIDATE.gene_pairs, FINAL_CANDIDATE.gene_groups,
        FINAL_CANDIDATE.hotspot_top_k, pairs,
        FINAL_CANDIDATE.amino_mode, FINAL_CANDIDATE.log1p_counts,
    )
    train_matrix, valid_matrix, names = builder.build(train_index, valid_index, labels)
    train_scores, valid_scores, score_names = cross_fitted_enrichment(
        token_matrix, train_index, valid_index, labels, seed)
    if train_scores.shape[1]:
        train_matrix = sparse.hstack([train_matrix, sparse.csr_matrix(train_scores)], format="csr")
        valid_matrix = sparse.hstack([valid_matrix, sparse.csr_matrix(valid_scores)], format="csr")
        names = names + score_names
    return train_matrix, valid_matrix, names


def make_submission(train, test, genes):
    """train 으로만 학습하고, 그 고정된 규칙을 test 행에 적용한다.

    train 과 test 는 **별도로 파싱**되며 하나의 프레임으로 합쳐지지 않는다.
    train 파싱이 (유전자, 변이) 어휘를 정의하고 test 는 그 어휘에 투영된다.
    """
    train_cache = RowCache.build(train[genes], genes)
    test_cache = RowCache.build(test[genes], genes, vocabulary=train_cache.event_names)
    cache = RowCache.stack(train_cache, test_cache)
    token_matrix = gene_type_matrix(cache)
    train_index = np.arange(len(train))
    test_index = np.arange(len(train), len(train) + len(test))

    train_matrix, test_matrix, names = build_design_matrices(
        cache, token_matrix, train_index, test_index, train[CONFIG.target_col], CONFIG.primary_seed)
    print(f"[info] 설계행렬 {train_matrix.shape[1]:,}열", flush=True)

    probabilities, warnings_seen, classes = {}, 0, None
    for name, model in make_models(CONFIG.primary_seed).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, train[CONFIG.target_col])
        warnings_seen += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        probabilities[name] = model.predict_proba(test_matrix)
        classes = model.classes_
        print(f"[info] {name} 학습 완료", flush=True)

    blended = sum(ENSEMBLE_WEIGHTS[name] * probabilities[name] for name in ENSEMBLE_WEIGHTS)
    submission = pd.DataFrame({CONFIG.id_col: test[CONFIG.id_col],
                               CONFIG.target_col: classes[blended.argmax(axis=1)]})
    metadata = {
        "seed": CONFIG.primary_seed,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "confusion_pairs": CONFUSION_PAIR_COUNT,
        "enrichment_shrinkage": ENRICHMENT_SHRINKAGE,
        "feature_count": len(names),
        "convergence_warning_count": warnings_seen,
        "train_rows": len(train),
        "test_rows": len(test),
    }
    return submission, metadata


def find_root(start: Path) -> Path:
    """data/raw 를 가진 프로젝트 루트를 찾는다.

    못 찾으면 현재 디렉터리를 쓴다 — 대회 환경처럼 `--data-dir /data` 로 경로를
    직접 주는 경우 루트 탐색이 실패해도 실행이 멈추지 않아야 한다.
    """
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    return start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="train.csv / test.csv 가 있는 경로. 대회 환경에서는 /data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--submission-name", default="submission.csv")
    args = parser.parse_args()

    root = find_root(Path.cwd())
    data_dir = args.data_dir or root / "data" / "raw"
    output_dir = args.output_dir or root / "experiments" / "iljun" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    genes = [column for column in train if column not in (CONFIG.id_col, CONFIG.target_col)]
    assert list(test.columns) == [CONFIG.id_col, *genes], "test 컬럼 구조가 train 과 다릅니다"
    assert int(train[genes].isna().sum().sum()) == 0, "train 에 예상치 못한 결측이 있습니다"
    # test 결측 개수는 평가 환경에서 달라질 수 있으므로 assert 하지 않고 참고만 한다.
    print(f"[info] train {train.shape} · test {test.shape} · 유전자 {len(genes):,}", flush=True)
    print(f"[info] test 결측 셀 {int(test[genes].isna().sum().sum()):,} (NaN 은 변이로 해석하지 않는다)",
          flush=True)

    submission, metadata = make_submission(train, test, genes)

    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        assert list(sample.columns) == [CONFIG.id_col, CONFIG.target_col]
        assert list(sample[CONFIG.id_col]) == list(test[CONFIG.id_col])
        sample[CONFIG.target_col] = submission[CONFIG.target_col]
        submission = sample

    assert len(submission) == len(test)
    assert list(submission.columns) == [CONFIG.id_col, CONFIG.target_col]
    assert submission[CONFIG.id_col].equals(test[CONFIG.id_col])
    assert int(submission.isna().sum().sum()) == 0
    assert submission[CONFIG.id_col].duplicated().sum() == 0

    path = output_dir / args.submission_name
    submission.to_csv(path, index=False)
    metadata["submission_path"] = str(path)
    metadata["distinct_classes"] = int(submission[CONFIG.target_col].nunique())
    (output_dir / f"{path.stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
