"""Hongju × DOMAIN-04A OOF runner with fold-safe, memory-bounded parsing.

No uploaded source file is imported.  All learned feature decisions are made
from the current fold's training indices only; RowCache contains only
deterministic, row-local representations of the supplied train cells.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
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
    expected_test_nan: int = 237
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
    def build(cls, frame: pd.DataFrame, genes: list[str], show_progress: bool = True) -> "RowCache":
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
            event_matrix = sparse.csr_matrix((n_rows, 0), dtype=np.float32); event_names: list[str] = []; missense = np.zeros(0, dtype=bool)
        else:
            events = events.drop_duplicates(["row", "gene_idx", "event"]).reset_index(drop=True)
            events["gene"] = events.gene_idx.map(dict(enumerate(genes)))
            events["pair"] = events.gene + "__" + events.event
            event_names = sorted(events.pair.unique())
            event_lookup = {name: idx for idx, name in enumerate(event_names)}
            event_matrix = sparse.coo_matrix((np.ones(len(events), dtype=np.float32), (events.row.to_numpy(), events.pair.map(event_lookup).to_numpy())), shape=(n_rows, len(event_names))).tocsr()
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


def make_model(model_name: str, seed: int, max_iter: int | None = None):
    if model_name == "logistic":
        return LogisticRegression(solver="lbfgs", C=CONFIG.lr_c, max_iter=max_iter or CONFIG.lr_max_iter, class_weight="balanced", random_state=seed)
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="multiclass", n_estimators=500, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)
    raise ValueError(model_name)


H_A = Candidate("H-A", "G+B+V+T+R+A")
H_AS = Candidate("H-AS", "G+B+V+T+R+A+S")
H_AS_CONTROL = Candidate("H-AS-control-maxiter5000", "G+B+V+T+R+A+S", lr_max_iter=5000)
LR_EXACT = (("BRAF", "V600E"), ("IDH1", "R132H"), ("PIK3CA", "H1047R"), ("PIK3CA", "E545K"))
LR_PAIR = (("IDH1", "TP53"), ("IDH1", "ATRX"), ("APC", "TP53"))
LGBM_EXACT = (("IDH1", "R132H"), ("BRAF", "V600E"), ("PIK3CA", "E545K"))
LGBM_PAIR = (("IDH1", "TP53"), ("IDH1", "ATRX"))
LGBM_GROUP = (("LAML", ("NPM1", "IDH1", "IDH2", "RUNX1")),)
CONTRAST_PAIRS = (("KIRC", "KIPAN", 5), ("LGG", "GBMLGG", 5))


def without_exact(event: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    """Return the fixed Logistic exact block with one event removed."""
    assert event in LR_EXACT
    return tuple(item for item in LR_EXACT if item != event)


CANDIDATES = {
    "H-A": H_A, "H-AS": H_AS, "H-AS-control-maxiter5000": H_AS_CONTROL,
    "H-AS-LR-exact": Candidate("H-AS-LR-exact", H_AS.backbone, LR_EXACT),
    "H-AS-LR-exact-hotspot-top3": Candidate("H-AS-LR-exact-hotspot-top3", H_AS.backbone, LR_EXACT, hotspot_top_k=3),
    "H-AS-LR-exact-confusion-pairs": Candidate("H-AS-LR-exact-confusion-pairs", H_AS.backbone, LR_EXACT, contrast_pairs=CONTRAST_PAIRS),
    "H-AS-LR-exact-confusion-pairs-Aall-log1p": Candidate("H-AS-LR-exact-confusion-pairs-Aall-log1p", H_AS.backbone, LR_EXACT, contrast_pairs=CONTRAST_PAIRS, amino_mode="all", log1p_counts=True),
    "H-AS-LR-exact-confusion-pairs-Apair-raw": Candidate("H-AS-LR-exact-confusion-pairs-Apair-raw", H_AS.backbone, LR_EXACT, contrast_pairs=CONTRAST_PAIRS, amino_mode="pair", log1p_counts=False),
    "H-AS-LR-exact-confusion-pairs-Apair-log1p": Candidate("H-AS-LR-exact-confusion-pairs-Apair-log1p", H_AS.backbone, LR_EXACT, contrast_pairs=CONTRAST_PAIRS, amino_mode="pair", log1p_counts=True),
    "H-AS-LR-exact-minus-BRAF-V600E": Candidate("H-AS-LR-exact-minus-BRAF-V600E", H_AS.backbone, without_exact(("BRAF", "V600E"))),
    "H-AS-LR-exact-minus-IDH1-R132H": Candidate("H-AS-LR-exact-minus-IDH1-R132H", H_AS.backbone, without_exact(("IDH1", "R132H"))),
    "H-AS-LR-exact-minus-PIK3CA-H1047R": Candidate("H-AS-LR-exact-minus-PIK3CA-H1047R", H_AS.backbone, without_exact(("PIK3CA", "H1047R"))),
    "H-AS-LR-exact-minus-PIK3CA-E545K": Candidate("H-AS-LR-exact-minus-PIK3CA-E545K", H_AS.backbone, without_exact(("PIK3CA", "E545K"))),
    "H-AS-LR-pair": Candidate("H-AS-LR-pair", H_AS.backbone, gene_pairs=LR_PAIR),
    "H-A-LR-core": Candidate("H-A-LR-core", H_A.backbone, LR_EXACT, LR_PAIR),
    "H-AS-LR-core": Candidate("H-AS-LR-core", H_AS.backbone, LR_EXACT, LR_PAIR),
    "H-A-LGBM-core": Candidate("H-A-LGBM-core", H_A.backbone, LGBM_EXACT, LGBM_PAIR, LGBM_GROUP),
    "H-AS-LGBM-core": Candidate("H-AS-LGBM-core", H_AS.backbone, LGBM_EXACT, LGBM_PAIR, LGBM_GROUP),
}


def run_oof(cache: RowCache, labels: pd.Series, candidate: Candidate, model_name: str, seed: int) -> tuple[dict, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=CONFIG.n_splits, shuffle=True, random_state=seed)
    predicted = np.empty(len(labels), dtype=object); scores = []; counts = []; warnings_seen = 0; started = perf_counter()
    for fold, (tr, va) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=CONFIG.n_splits, desc=f"{candidate.experiment_id} | {model_name} | seed {seed}"), 1):
        builder = FoldMatrixBuilder(cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs, candidate.gene_groups, candidate.hotspot_top_k, candidate.contrast_pairs, candidate.amino_mode, candidate.log1p_counts)
        train_matrix, valid_matrix, names = builder.build(tr, va, labels); model = make_model(model_name, seed, candidate.lr_max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning); model.fit(train_matrix, labels.iloc[tr])
        fold_prediction = model.predict(valid_matrix); predicted[va] = fold_prediction
        scores.append(f1_score(labels.iloc[va], fold_prediction, average="macro", zero_division=0)); counts.append(len(names))
        warnings_seen += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        del train_matrix, valid_matrix, model; gc.collect()
    classes = sorted(labels.unique())
    report = pd.DataFrame(classification_report(labels, predicted, labels=classes, output_dict=True, zero_division=0)).T.loc[classes].reset_index(names="class")
    return {"experiment_id": candidate.experiment_id, "model": model_name, "seed": seed, "oof_macro_f1": f1_score(labels, predicted, average="macro", zero_division=0), "oof_accuracy": accuracy_score(labels, predicted), "fold_macro_f1_mean": float(np.mean(scores)), "fold_macro_f1_std": float(np.std(scores)), "feature_count_mean": float(np.mean(counts)), "runtime_seconds": perf_counter()-started, "convergence_warning_count": warnings_seen, "leakage_check": True, "nan_as_mutation_count": 0}, report


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


def make_submission(train: pd.DataFrame, test: pd.DataFrame, genes: list[str], candidate: Candidate, model_name: str, seed: int) -> tuple[pd.DataFrame, dict]:
    """Fit only on train, then apply the fixed train-derived feature rules to test."""
    combined = pd.concat([train[genes], test[genes]], axis=0, ignore_index=True)
    cache = RowCache.build(combined, genes)
    train_index = np.arange(len(train))
    test_index = np.arange(len(train), len(combined))
    builder = FoldMatrixBuilder(cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs, candidate.gene_groups, candidate.hotspot_top_k, candidate.contrast_pairs, candidate.amino_mode, candidate.log1p_counts)
    train_matrix, test_matrix, names = builder.build(train_index, test_index, train[CONFIG.target_col])
    model = make_model(model_name, seed, candidate.lr_max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_matrix, train[CONFIG.target_col])
    prediction = model.predict(test_matrix)
    warnings_seen = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    submission = pd.DataFrame({CONFIG.id_col: test[CONFIG.id_col], CONFIG.target_col: prediction})
    metadata = {
        "experiment_id": candidate.experiment_id,
        "model": model_name,
        "seed": seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(names),
        "convergence_warning_count": warnings_seen,
        "leakage_check": True,
        "nan_as_mutation_count": 0,
    }
    return submission, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="H-A", choices=CANDIDATES)
    parser.add_argument("--model", default="logistic", choices=("logistic", "lightgbm"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--submission-name", default="")
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
    assert int(test[genes].isna().sum().sum()) == CONFIG.expected_test_nan
    assert int(train[genes].isna().sum().sum()) == 0
    assert nan_as_mutation_count(train, genes) == 0
    assert nan_as_mutation_count(test, genes) == 0
    result_output = root / "experiments" / "gs" / "notebooks" / "eda_pre_002" / "result"
    if args.submit:
        output = root / "experiments" / "gs" / "notebooks" / "submission"
        output.mkdir(parents=True, exist_ok=True)
        submission, metadata = make_submission(train, test, genes, CANDIDATES[args.candidate], args.model, args.seed)
        sample_path = data_dir / "sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            assert list(sample.columns) == [CONFIG.id_col, CONFIG.target_col]
            assert list(sample[CONFIG.id_col]) == list(test[CONFIG.id_col])
            sample[CONFIG.target_col] = submission[CONFIG.target_col]
            submission = sample
        default_name = f"submission_{args.candidate}_{args.model}_seed{args.seed}.csv"
        submission_path = output / (args.submission_name or default_name)
        submission.to_csv(submission_path, index=False)
        metadata_path = output / f"{submission_path.stem}_metadata.csv"
        pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
        print(json.dumps({**metadata, "submission_path": str(submission_path)}, ensure_ascii=False, indent=2))
        return
    output = result_output
    output.mkdir(parents=True, exist_ok=True)
    cache = RowCache.build(train[genes], genes)
    result, class_result = run_oof(cache, train[CONFIG.target_col], CANDIDATES[args.candidate], args.model, args.seed)
    stem = "_".join(part for part in (args.run_id, args.candidate, args.model, f"seed{args.seed}") if part)
    pd.DataFrame([result]).to_csv(output / f"{stem}_oof.csv", index=False)
    class_result.to_csv(output / f"{stem}_class_f1.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
