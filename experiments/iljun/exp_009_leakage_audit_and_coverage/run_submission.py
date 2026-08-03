"""Final single-file submission runner for 26-class cancer classification.

This file is self-contained: it imports no project-local Python module.  It
uses the fixed final configuration validated in exp-gs-002-08:
H-AS backbone + four exact hotspots + confusion-pair contrast + A_pair-only
+ log1p counts.  Test rows are parsed row-by-row only; all learned feature
selection is derived from the complete training data and its labels.
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
FINAL_CONTRAST_PAIRS = (
    ("KIRC", "KIPAN", 5),
    ("LGG", "GBMLGG", 5),
)
FINAL_CANDIDATE = Candidate(
    experiment_id="H-AS-LR-exact-confusion-pairs-Apair-log1p",
    backbone="G+B+V+T+R+A+S",
    exact_events=FINAL_EXACT_HOTSPOTS,
    contrast_pairs=FINAL_CONTRAST_PAIRS,
    amino_mode="pair",
    log1p_counts=True,
)


def make_model() -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        C=CONFIG.lr_c,
        max_iter=CONFIG.lr_max_iter,
        class_weight="balanced",
        random_state=CONFIG.primary_seed,
    )


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists() or (path / "experiments" / "gs").exists():
            return path
    raise FileNotFoundError("data/raw 또는 experiments/gs를 가진 프로젝트 루트를 찾지 못했습니다.")


def nan_as_mutation_count(frame: pd.DataFrame, genes: list[str]) -> int:
    """Audit only missing cells; normalise_cell(None/NaN) must always be empty."""
    count = 0
    for gene in genes:
        missing = frame[gene][frame[gene].isna()]
        count += sum(bool(normalise_cell(value)) for value in missing)
    return count


def make_submission(train: pd.DataFrame, test: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, dict]:
    """Fit on train only, then apply its fixed feature rules to test rows.

    train and test are parsed in two separate passes and are never concatenated
    as raw frames.  The train pass defines the (gene, event) vocabulary; the test
    pass is projected onto that vocabulary, so no test row can introduce a
    feature column.  Every learned selection below is additionally restricted to
    `train_index`.
    """
    train_cache = RowCache.build(train[genes], genes)
    test_cache = RowCache.build(test[genes], genes, vocabulary=train_cache.event_names)
    cache = RowCache.stack(train_cache, test_cache)
    train_index = np.arange(len(train))
    test_index = np.arange(len(train), len(train) + len(test))
    builder = FoldMatrixBuilder(
        cache,
        FINAL_CANDIDATE.backbone,
        FINAL_CANDIDATE.exact_events,
        FINAL_CANDIDATE.gene_pairs,
        FINAL_CANDIDATE.gene_groups,
        FINAL_CANDIDATE.hotspot_top_k,
        FINAL_CANDIDATE.contrast_pairs,
        FINAL_CANDIDATE.amino_mode,
        FINAL_CANDIDATE.log1p_counts,
    )
    train_matrix, test_matrix, names = builder.build(train_index, test_index, train[CONFIG.target_col])

    # Measured leakage check (previously a hardcoded True).  Rebuild the train
    # design matrix from a cache that contains no test rows at all.  If any test
    # row had influenced vocabulary, selection or scaling, these two matrices
    # would differ.  Reuses the already-parsed train cache, so no re-parsing.
    train_only_builder = FoldMatrixBuilder(
        train_cache,
        FINAL_CANDIDATE.backbone,
        FINAL_CANDIDATE.exact_events,
        FINAL_CANDIDATE.gene_pairs,
        FINAL_CANDIDATE.gene_groups,
        FINAL_CANDIDATE.hotspot_top_k,
        FINAL_CANDIDATE.contrast_pairs,
        FINAL_CANDIDATE.amino_mode,
        FINAL_CANDIDATE.log1p_counts,
    )
    solo_matrix, _, solo_names = train_only_builder.build(train_index, train_index, train[CONFIG.target_col])
    leakage_check = bool(solo_names == names and (solo_matrix != train_matrix).nnz == 0)
    assert leakage_check, "test 행의 존재가 train 설계행렬을 바꿨습니다 — 누수 점검 실패"
    test_only_events = sum(1 for name in test_cache.events.get("pair", pd.Series(dtype=str)).unique()
                           if name not in set(train_cache.event_names))

    model = make_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_matrix, train[CONFIG.target_col])
    prediction = model.predict(test_matrix)
    warnings_seen = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    submission = pd.DataFrame({CONFIG.id_col: test[CONFIG.id_col], CONFIG.target_col: prediction})
    metadata = {
        "experiment_id": FINAL_CANDIDATE.experiment_id,
        "model": "logistic",
        "seed": CONFIG.primary_seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(names),
        "convergence_warning_count": warnings_seen,
        "leakage_check": leakage_check,
        "leakage_check_method": "train-only rebuild vs stacked build, exact matrix equality",
        "vocabulary_source": "train",
        "test_only_events_dropped": test_only_events,
    }
    return submission, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--submission-name",
        default="submission_exp-gs-002-final_single_run_seed42.csv",
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert normalise_cell(np.nan) == () and normalise_cell("WT") == (); print("self-check: parser/NaN contract passed"); return
    root = find_root(Path.cwd())
    data_dir = args.data_dir or root / "data" / "raw"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    genes = [col for col in train if col not in (CONFIG.id_col, CONFIG.target_col)]
    assert list(test.columns) == [CONFIG.id_col, *genes]
    # NOTE: test 결측 개수는 평가 환경에서 달라질 수 있으므로 assert 하지 않고
    # 참고용으로만 출력한다. 특정 값을 코드에 고정하면 (1) 다른 test 파일에서
    # AssertionError 로 실행이 멈추고, (2) test 를 사전 관찰했다는 근거가 된다.
    print(f"[info] test 결측 셀 수: {int(test[genes].isna().sum().sum())}")
    assert int(train[genes].isna().sum().sum()) == 0
    # NaN 은 어떤 경우에도 mutation event 로 해석되지 않아야 한다 (파서 계약).
    assert nan_as_mutation_count(train, genes) == 0
    assert nan_as_mutation_count(test, genes) == 0
    output = root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    submission, metadata = make_submission(train, test, genes)
    assert metadata["convergence_warning_count"] == 0, "최종 전체 train 학습에서 수렴 경고가 발생했습니다."

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
    submission_path = output / args.submission_name
    submission.to_csv(submission_path, index=False)
    metadata_path = output / f"{submission_path.stem}_metadata.csv"
    pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
    print(json.dumps({**metadata, "submission_path": str(submission_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
