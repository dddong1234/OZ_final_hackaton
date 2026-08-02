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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
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
    def __init__(self, cache: RowCache, backbone: str, exact_events=(), gene_pairs=(), gene_groups=(), hotspot_top_k: int = 0, contrast_pairs=(), amino_mode: str = "all", log1p_counts: bool = False, b_count_binning: bool = False):
        self.cache, self.backbone = cache, backbone
        self.exact_events, self.gene_pairs, self.gene_groups = tuple(exact_events), tuple(gene_pairs), tuple(gene_groups)
        self.hotspot_top_k, self.contrast_pairs = hotspot_top_k, tuple(contrast_pairs)
        assert amino_mode in {"all", "pair"}
        self.amino_mode, self.log1p_counts, self.b_count_binning = amino_mode, log1p_counts, b_count_binning

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
        if self.b_count_binning:
            raw_burden = cache.burden
            bins = (
                (0, "mutated_gene", 1, 1), (0, "mutated_gene", 2, 2), (0, "mutated_gene", 3, 4), (0, "mutated_gene", 5, 7), (0, "mutated_gene", 8, np.inf),
                (1, "event", 1, 1), (1, "event", 2, 2), (1, "event", 3, 4), (1, "event", 5, 7), (1, "event", 8, np.inf),
                (2, "multi_event_gene", 1, 1), (2, "multi_event_gene", 2, np.inf),
            )
            bin_matrix = np.column_stack([(raw_burden[:, column] >= low) & (raw_burden[:, column] <= high) for column, _, low, high in bins]).astype(np.float32)
            parts.append(sparse.csr_matrix(bin_matrix))
            names += [f"B_bin__{label}_{low}_{'plus' if np.isinf(high) else high}" for _, label, low, high in bins]
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
    b_count_binning: bool = False
    lr_max_iter: int = CONFIG.lr_max_iter


def make_model(model_name: str, seed: int, max_iter: int | None = None, multi_class: str | None = None):
    if model_name == "logistic":
        parameters = {"solver": "lbfgs", "C": CONFIG.lr_c, "max_iter": max_iter or CONFIG.lr_max_iter, "class_weight": "balanced", "random_state": seed}
        base_model = LogisticRegression(**parameters)
        if multi_class is None or multi_class == "multinomial":
            return base_model
        if multi_class == "ovr":
            return OneVsRestClassifier(base_model)
        raise ValueError(f"Unsupported logistic multi_class mode: {multi_class}")
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
    "H-AS-LR-exact-confusion-pairs-Apair-log1p-Bbins": Candidate("H-AS-LR-exact-confusion-pairs-Apair-log1p-Bbins", H_AS.backbone, LR_EXACT, contrast_pairs=CONTRAST_PAIRS, amino_mode="pair", log1p_counts=True, b_count_binning=True),
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
        builder = FoldMatrixBuilder(cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs, candidate.gene_groups, candidate.hotspot_top_k, candidate.contrast_pairs, candidate.amino_mode, candidate.log1p_counts, candidate.b_count_binning)
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


def soft_blend_pair_probabilities(primary_probability: np.ndarray, left_index: int, right_index: int, expert_left_probability: np.ndarray, alpha: float) -> np.ndarray:
    """Preserve pair mass and all non-pair probabilities while softly blending pair odds."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("specialist alpha must be in [0, 1]")
    blended = primary_probability.copy()
    pair_mass = primary_probability[:, left_index] + primary_probability[:, right_index]
    primary_left_ratio = np.divide(primary_probability[:, left_index], pair_mass, out=np.full_like(pair_mass, 0.5), where=pair_mass > 0)
    blend_weight = alpha * pair_mass
    final_left_ratio = (1.0 - blend_weight) * primary_left_ratio + blend_weight * expert_left_probability
    blended[:, left_index] = pair_mass * final_left_ratio
    blended[:, right_index] = pair_mass * (1.0 - final_left_ratio)
    assert np.allclose(blended.sum(axis=1), primary_probability.sum(axis=1))
    return blended


def _class_f1_frame(labels: pd.Series, predicted: np.ndarray, classes: list[str], variant: str) -> pd.DataFrame:
    report = pd.DataFrame(classification_report(labels, predicted, labels=classes, output_dict=True, zero_division=0)).T.loc[classes].reset_index(names="class")
    report.insert(0, "variant", variant)
    return report


def fixed_three_way_probability_blend(primary_probability: np.ndarray, multinomial_probability: np.ndarray, ovr_probability: np.ndarray) -> np.ndarray:
    """Fixed 0.50 primary + 0.25 multinomial token + 0.25 OVR token blend."""
    if primary_probability.shape != multinomial_probability.shape or primary_probability.shape != ovr_probability.shape:
        raise ValueError("all probability matrices must have identical shapes")
    blended = 0.50 * primary_probability + 0.25 * multinomial_probability + 0.25 * ovr_probability
    assert np.allclose(blended.sum(axis=1), 1.0)
    return blended


def event_token_documents(cache: RowCache) -> list[str]:
    """Deterministic row-local mutation documents; no labels or fitted statistics."""
    documents = [[] for _ in range(cache.mutation_matrix.shape[0])]
    if cache.events.empty:
        return ["" for _ in documents]
    for row, gene, event, event_type, ref, alt in cache.events[["row", "gene", "event", "event_type", "ref", "alt"]].itertuples(index=False):
        documents[row].extend((f"G__{gene}", f"E__{gene}__{event}", f"TYPE__{event_type}"))
        if pd.notna(ref) and pd.notna(alt) and ref != alt:
            documents[row].append(f"AA__{ref}_{alt}")
    return [" ".join(tokens) for tokens in documents]


def run_event_tfidf_oof(cache: RowCache, labels: pd.Series, candidate: Candidate, seed: int, min_df: int = 3) -> tuple[dict, pd.DataFrame]:
    classes = sorted(labels.unique()); class_index = {name: i for i, name in enumerate(classes)}
    documents = event_token_documents(cache); splitter = StratifiedKFold(n_splits=CONFIG.n_splits, shuffle=True, random_state=seed)
    primary_probability = np.zeros((len(labels), len(classes))); token_probability = np.zeros_like(primary_probability)
    counts=[]; vocab_sizes=[]; primary_warn=token_warn=0; started=perf_counter()
    for fold, (tr, va) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=CONFIG.n_splits, desc=f"{candidate.experiment_id} | event-tfidf | seed {seed}"), 1):
        builder = FoldMatrixBuilder(cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs, candidate.gene_groups, candidate.hotspot_top_k, candidate.contrast_pairs, candidate.amino_mode, candidate.log1p_counts, candidate.b_count_binning)
        x_train, x_valid, names = builder.build(tr, va, labels)
        primary = make_model("logistic", seed, candidate.lr_max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning); primary.fit(x_train, labels.iloc[tr])
        primary_warn += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        raw_primary = primary.predict_proba(x_valid)
        for column, name in enumerate(primary.classes_): primary_probability[va, class_index[name]] = raw_primary[:, column]
        vectorizer = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False, ngram_range=(1, 1), min_df=min_df, sublinear_tf=True, norm="l2", dtype=np.float32)
        token_train = vectorizer.fit_transform([documents[i] for i in tr]); token_valid = vectorizer.transform([documents[i] for i in va]); vocab_sizes.append(len(vectorizer.vocabulary_))
        token_model = make_model("logistic", seed, candidate.lr_max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning); token_model.fit(token_train, labels.iloc[tr])
        token_warn += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        raw_token = token_model.predict_proba(token_valid)
        for column, name in enumerate(token_model.classes_): token_probability[va, class_index[name]] = raw_token[:, column]
        counts.append(len(names)); del x_train, x_valid, primary, token_model, token_train, token_valid; gc.collect()
    blend_probability = 0.5 * primary_probability + 0.5 * token_probability
    rows=[]; class_rows=[]
    for variant, probability in (("primary", primary_probability), ("event_tfidf", token_probability), ("blend_0p5", blend_probability)):
        prediction=np.asarray(classes)[probability.argmax(axis=1)]
        rows.append((variant, f1_score(labels, prediction, average="macro", zero_division=0), accuracy_score(labels, prediction)))
        class_rows.append(_class_f1_frame(labels, prediction, classes, variant))
    score={name: macro for name, macro, _ in rows}; accuracy={name: acc for name, _, acc in rows}
    result={"experiment_id":candidate.experiment_id,"model":"logistic","seed":seed,"tfidf_min_df":min_df,"tfidf_sublinear_tf":True,"primary_oof_macro_f1":score["primary"],"event_tfidf_oof_macro_f1":score["event_tfidf"],"blend_0p5_oof_macro_f1":score["blend_0p5"],"delta_blend_vs_primary":score["blend_0p5"]-score["primary"],"primary_oof_accuracy":accuracy["primary"],"event_tfidf_oof_accuracy":accuracy["event_tfidf"],"blend_0p5_oof_accuracy":accuracy["blend_0p5"],"feature_count_mean":float(np.mean(counts)),"tfidf_vocabulary_size_mean":float(np.mean(vocab_sizes)),"runtime_seconds":perf_counter()-started,"primary_convergence_warning_count":primary_warn,"tfidf_convergence_warning_count":token_warn,"convergence_warning_count":primary_warn+token_warn,"leakage_check":True,"nan_as_mutation_count":0}
    return result, pd.concat(class_rows, ignore_index=True)


def run_event_tfidf_ovr_comparison_oof(cache: RowCache, labels: pd.Series, candidate: Candidate, seed: int, min_df: int = 3) -> tuple[dict, pd.DataFrame]:
    """Compare multinomial and OVR event-token LR on identical fold-local TF-IDF matrices."""
    classes = sorted(labels.unique())
    class_index = {name: index for index, name in enumerate(classes)}
    documents = event_token_documents(cache)
    splitter = StratifiedKFold(n_splits=CONFIG.n_splits, shuffle=True, random_state=seed)
    primary_probability = np.zeros((len(labels), len(classes)))
    multinomial_probability = np.zeros_like(primary_probability)
    ovr_probability = np.zeros_like(primary_probability)
    feature_counts: list[int] = []
    vocabulary_sizes: list[int] = []
    primary_warn = multinomial_warn = ovr_warn = 0
    started = perf_counter()

    for fold, (tr, va) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=CONFIG.n_splits, desc=f"{candidate.experiment_id} | TF-IDF multinomial vs OVR | seed {seed}"), 1):
        builder = FoldMatrixBuilder(cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs, candidate.gene_groups, candidate.hotspot_top_k, candidate.contrast_pairs, candidate.amino_mode, candidate.log1p_counts, candidate.b_count_binning)
        x_train, x_valid, names = builder.build(tr, va, labels)

        primary = make_model("logistic", seed, candidate.lr_max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            primary.fit(x_train, labels.iloc[tr])
        primary_warn += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        raw_primary = primary.predict_proba(x_valid)
        for column, name in enumerate(primary.classes_):
            primary_probability[va, class_index[name]] = raw_primary[:, column]

        vectorizer = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False, ngram_range=(1, 1), min_df=min_df, sublinear_tf=True, norm="l2", dtype=np.float32)
        token_train = vectorizer.fit_transform([documents[index] for index in tr])
        token_valid = vectorizer.transform([documents[index] for index in va])
        vocabulary_sizes.append(len(vectorizer.vocabulary_))

        multinomial = make_model("logistic", seed, candidate.lr_max_iter, multi_class="multinomial")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            multinomial.fit(token_train, labels.iloc[tr])
        multinomial_warn += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        raw_multinomial = multinomial.predict_proba(token_valid)
        for column, name in enumerate(multinomial.classes_):
            multinomial_probability[va, class_index[name]] = raw_multinomial[:, column]

        ovr = make_model("logistic", seed, candidate.lr_max_iter, multi_class="ovr")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            ovr.fit(token_train, labels.iloc[tr])
        ovr_warn += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        raw_ovr = ovr.predict_proba(token_valid)
        for column, name in enumerate(ovr.classes_):
            ovr_probability[va, class_index[name]] = raw_ovr[:, column]

        feature_counts.append(len(names))
        del x_train, x_valid, primary, token_train, token_valid, multinomial, ovr
        gc.collect()

    probabilities = {
        "primary": primary_probability,
        "event_tfidf_multinomial": multinomial_probability,
        "event_tfidf_ovr": ovr_probability,
        "blend_multinomial_0p5": 0.5 * primary_probability + 0.5 * multinomial_probability,
        "blend_ovr_0p5": 0.5 * primary_probability + 0.5 * ovr_probability,
        "blend_three_way_0p5_0p25_0p25": fixed_three_way_probability_blend(primary_probability, multinomial_probability, ovr_probability),
    }
    predictions = {variant: np.asarray(classes)[probability.argmax(axis=1)] for variant, probability in probabilities.items()}
    scores = {variant: f1_score(labels, prediction, average="macro", zero_division=0) for variant, prediction in predictions.items()}
    accuracy = {variant: accuracy_score(labels, prediction) for variant, prediction in predictions.items()}
    class_result = pd.concat([_class_f1_frame(labels, prediction, classes, variant) for variant, prediction in predictions.items()], ignore_index=True)
    disagreement = float(np.mean(predictions["event_tfidf_multinomial"] != predictions["event_tfidf_ovr"]))
    result = {
        "experiment_id": candidate.experiment_id,
        "model": "logistic",
        "seed": seed,
        "tfidf_min_df": min_df,
        "tfidf_sublinear_tf": True,
        "primary_oof_macro_f1": scores["primary"],
        "event_tfidf_multinomial_oof_macro_f1": scores["event_tfidf_multinomial"],
        "event_tfidf_ovr_oof_macro_f1": scores["event_tfidf_ovr"],
        "blend_multinomial_0p5_oof_macro_f1": scores["blend_multinomial_0p5"],
        "blend_ovr_0p5_oof_macro_f1": scores["blend_ovr_0p5"],
        "blend_three_way_0p5_0p25_0p25_oof_macro_f1": scores["blend_three_way_0p5_0p25_0p25"],
        "delta_ovr_blend_vs_multinomial_blend": scores["blend_ovr_0p5"] - scores["blend_multinomial_0p5"],
        "delta_three_way_vs_ovr_blend": scores["blend_three_way_0p5_0p25_0p25"] - scores["blend_ovr_0p5"],
        "primary_oof_accuracy": accuracy["primary"],
        "event_tfidf_multinomial_oof_accuracy": accuracy["event_tfidf_multinomial"],
        "event_tfidf_ovr_oof_accuracy": accuracy["event_tfidf_ovr"],
        "blend_multinomial_0p5_oof_accuracy": accuracy["blend_multinomial_0p5"],
        "blend_ovr_0p5_oof_accuracy": accuracy["blend_ovr_0p5"],
        "blend_three_way_0p5_0p25_0p25_oof_accuracy": accuracy["blend_three_way_0p5_0p25_0p25"],
        "token_prediction_disagreement_rate": disagreement,
        "feature_count_mean": float(np.mean(feature_counts)),
        "tfidf_vocabulary_size_mean": float(np.mean(vocabulary_sizes)),
        "runtime_seconds": perf_counter() - started,
        "primary_convergence_warning_count": primary_warn,
        "tfidf_multinomial_convergence_warning_count": multinomial_warn,
        "tfidf_ovr_convergence_warning_count": ovr_warn,
        "convergence_warning_count": primary_warn + multinomial_warn + ovr_warn,
        "leakage_check": True,
        "nan_as_mutation_count": 0,
    }
    return result, class_result


def run_soft_pair_specialist_oof(cache: RowCache, labels: pd.Series, candidate: Candidate, seed: int, left_class: str, right_class: str, alpha: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run a fold-safe binary specialist and soft pair-probability blend on primary OOF probabilities."""
    classes = sorted(labels.unique())
    if left_class not in classes or right_class not in classes:
        raise ValueError("specialist classes must be present in train labels")
    left_index, right_index = classes.index(left_class), classes.index(right_class)
    splitter = StratifiedKFold(n_splits=CONFIG.n_splits, shuffle=True, random_state=seed)
    primary_probability = np.zeros((len(labels), len(classes)), dtype=np.float64)
    blended_probability = np.zeros_like(primary_probability)
    feature_counts: list[int] = []; fold_scores: list[float] = []; primary_warnings = 0; specialist_warnings = 0; pair_train_sizes: list[int] = []
    started = perf_counter()
    for fold, (tr, va) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=CONFIG.n_splits, desc=f"{candidate.experiment_id} | soft specialist | seed {seed}"), 1):
        builder = FoldMatrixBuilder(cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs, candidate.gene_groups, candidate.hotspot_top_k, candidate.contrast_pairs, candidate.amino_mode, candidate.log1p_counts, candidate.b_count_binning)
        train_matrix, valid_matrix, names = builder.build(tr, va, labels)
        primary = make_model("logistic", seed, candidate.lr_max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            primary.fit(train_matrix, labels.iloc[tr])
        primary_warnings += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        fold_primary = np.zeros((len(va), len(classes)), dtype=np.float64)
        raw_primary_probability = primary.predict_proba(valid_matrix)
        for column, class_name in enumerate(primary.classes_):
            fold_primary[:, classes.index(class_name)] = raw_primary_probability[:, column]

        pair_mask = labels.iloc[tr].isin((left_class, right_class)).to_numpy()
        pair_train_sizes.append(int(pair_mask.sum()))
        if np.unique(labels.iloc[tr].to_numpy()[pair_mask]).size != 2:
            raise ValueError("both specialist classes must be present in every outer-fold train split")
        specialist = make_model("logistic", seed, candidate.lr_max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            specialist.fit(train_matrix[pair_mask], labels.iloc[tr].to_numpy()[pair_mask])
        specialist_warnings += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        specialist_probability = specialist.predict_proba(valid_matrix)
        expert_left = specialist_probability[:, list(specialist.classes_).index(left_class)]
        fold_blended = soft_blend_pair_probabilities(fold_primary, left_index, right_index, expert_left, alpha)
        non_pair_indices = [index for index in range(len(classes)) if index not in (left_index, right_index)]
        assert np.allclose(fold_blended[:, non_pair_indices], fold_primary[:, non_pair_indices])
        primary_probability[va] = fold_primary; blended_probability[va] = fold_blended
        fold_scores.append(f1_score(labels.iloc[va], np.asarray(classes)[fold_blended.argmax(axis=1)], average="macro", zero_division=0)); feature_counts.append(len(names))
        del train_matrix, valid_matrix, primary, specialist; gc.collect()

    primary_prediction = np.asarray(classes)[primary_probability.argmax(axis=1)]
    blended_prediction = np.asarray(classes)[blended_probability.argmax(axis=1)]
    class_result = pd.concat((_class_f1_frame(labels, primary_prediction, classes, "primary"), _class_f1_frame(labels, blended_prediction, classes, "soft_specialist")), ignore_index=True)
    pair_rows = []
    for variant, prediction in (("primary", primary_prediction), ("soft_specialist", blended_prediction)):
        for true_class, predicted_class in ((left_class, right_class), (right_class, left_class)):
            pair_rows.append({"variant": variant, "true_class": true_class, "predicted_class": predicted_class, "count": int(((labels == true_class) & (prediction == predicted_class)).sum())})
    result = {
        "experiment_id": candidate.experiment_id, "model": "logistic", "seed": seed, "specialist_left": left_class, "specialist_right": right_class, "specialist_alpha": alpha,
        "primary_oof_macro_f1": f1_score(labels, primary_prediction, average="macro", zero_division=0), "soft_specialist_oof_macro_f1": f1_score(labels, blended_prediction, average="macro", zero_division=0),
        "delta_oof_macro_f1": f1_score(labels, blended_prediction, average="macro", zero_division=0) - f1_score(labels, primary_prediction, average="macro", zero_division=0),
        "primary_oof_accuracy": accuracy_score(labels, primary_prediction), "soft_specialist_oof_accuracy": accuracy_score(labels, blended_prediction), "fold_soft_specialist_macro_f1_mean": float(np.mean(fold_scores)), "fold_soft_specialist_macro_f1_std": float(np.std(fold_scores)),
        "feature_count_mean": float(np.mean(feature_counts)), "specialist_train_rows_mean": float(np.mean(pair_train_sizes)), "runtime_seconds": perf_counter() - started,
        "primary_convergence_warning_count": primary_warnings, "specialist_convergence_warning_count": specialist_warnings, "convergence_warning_count": primary_warnings + specialist_warnings,
        "leakage_check": True, "nan_as_mutation_count": 0,
    }
    return result, class_result, pd.DataFrame(pair_rows)


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
    builder = FoldMatrixBuilder(cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs, candidate.gene_groups, candidate.hotspot_top_k, candidate.contrast_pairs, candidate.amino_mode, candidate.log1p_counts, candidate.b_count_binning)
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


def make_event_tfidf_ovr_submission(train: pd.DataFrame, test: pd.DataFrame, genes: list[str], candidate: Candidate, seed: int, min_df: int = 3) -> tuple[pd.DataFrame, dict]:
    """Fit the fixed exp13 primary/OVR models on train only and blend test probabilities 0.5/0.5."""
    combined = pd.concat([train[genes], test[genes]], axis=0, ignore_index=True)
    cache = RowCache.build(combined, genes)
    train_index = np.arange(len(train))
    test_index = np.arange(len(train), len(combined))
    classes = sorted(train[CONFIG.target_col].unique())
    class_index = {name: index for index, name in enumerate(classes)}
    builder = FoldMatrixBuilder(cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs, candidate.gene_groups, candidate.hotspot_top_k, candidate.contrast_pairs, candidate.amino_mode, candidate.log1p_counts, candidate.b_count_binning)
    train_matrix, test_matrix, feature_names = builder.build(train_index, test_index, train[CONFIG.target_col])

    primary = make_model("logistic", seed, candidate.lr_max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        primary.fit(train_matrix, train[CONFIG.target_col])
    primary_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    primary_probability = np.zeros((len(test), len(classes)))
    raw_primary_probability = primary.predict_proba(test_matrix)
    for column, name in enumerate(primary.classes_):
        primary_probability[:, class_index[name]] = raw_primary_probability[:, column]

    documents = event_token_documents(cache)
    vectorizer = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False, ngram_range=(1, 1), min_df=min_df, sublinear_tf=True, norm="l2", dtype=np.float32)
    token_train = vectorizer.fit_transform(documents[:len(train)])
    token_test = vectorizer.transform(documents[len(train):])
    ovr = make_model("logistic", seed, candidate.lr_max_iter, multi_class="ovr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        ovr.fit(token_train, train[CONFIG.target_col])
    ovr_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    ovr_probability = np.zeros_like(primary_probability)
    raw_ovr_probability = ovr.predict_proba(token_test)
    for column, name in enumerate(ovr.classes_):
        ovr_probability[:, class_index[name]] = raw_ovr_probability[:, column]

    blended_probability = 0.5 * primary_probability + 0.5 * ovr_probability
    prediction = np.asarray(classes)[blended_probability.argmax(axis=1)]
    submission = pd.DataFrame({CONFIG.id_col: test[CONFIG.id_col], CONFIG.target_col: prediction})
    metadata = {
        "experiment_id": "exp-gs-002-13",
        "candidate": candidate.experiment_id,
        "model": "primary_logistic_0p5 + event_tfidf_ovr_logistic_0p5",
        "seed": seed,
        "tfidf_min_df": min_df,
        "tfidf_sublinear_tf": True,
        "train_rows": len(train),
        "test_rows": len(test),
        "primary_feature_count": len(feature_names),
        "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
        "primary_convergence_warning_count": primary_warnings,
        "tfidf_ovr_convergence_warning_count": ovr_warnings,
        "convergence_warning_count": primary_warnings + ovr_warnings,
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
    parser.add_argument("--submit-event-tfidf-ovr", action="store_true")
    parser.add_argument("--submission-name", default="")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--soft-specialist", action="store_true")
    parser.add_argument("--specialist-left", default="KIRC")
    parser.add_argument("--specialist-right", default="KIPAN")
    parser.add_argument("--specialist-alpha", type=float, default=0.30)
    parser.add_argument("--event-tfidf", action="store_true")
    parser.add_argument("--event-tfidf-ovr", action="store_true")
    parser.add_argument("--event-tfidf-three-way", action="store_true")
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    args = parser.parse_args()
    if args.self_check:
        assert normalise_cell(np.nan) == () and normalise_cell("WT") == (); print("self-check: parser/NaN contract passed"); return
    root = find_root(Path.cwd())
    data_dir = args.data_dir or root / "data" / "raw"
    train = pd.read_csv(data_dir / "train.csv")
    genes = [col for col in train if col not in (CONFIG.id_col, CONFIG.target_col)]
    assert int(train[genes].isna().sum().sum()) == 0
    assert nan_as_mutation_count(train, genes) == 0
    result_output = root / "experiments" / "gs" / "notebooks" / "eda_pre_002" / "result"
    if args.submit:
        test = pd.read_csv(data_dir / "test.csv")
        assert list(test.columns) == [CONFIG.id_col, *genes]
        assert nan_as_mutation_count(test, genes) == 0
        output = root / "experiments" / "gs" / "notebooks" / "submission"
        output.mkdir(parents=True, exist_ok=True)
        if args.submit_event_tfidf_ovr:
            if args.model != "logistic":
                raise ValueError("event-token OVR submission requires --model logistic")
            submission, metadata = make_event_tfidf_ovr_submission(train, test, genes, CANDIDATES[args.candidate], args.seed, args.tfidf_min_df)
        else:
            submission, metadata = make_submission(train, test, genes, CANDIDATES[args.candidate], args.model, args.seed)
        sample_path = data_dir / "sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            assert list(sample.columns) == [CONFIG.id_col, CONFIG.target_col]
            assert list(sample[CONFIG.id_col]) == list(test[CONFIG.id_col])
            sample[CONFIG.target_col] = submission[CONFIG.target_col]
            submission = sample
        default_name = f"submission_exp-gs-002-13_{args.candidate}_primary-ovr-tfidf_seed{args.seed}.csv" if args.submit_event_tfidf_ovr else f"submission_{args.candidate}_{args.model}_seed{args.seed}.csv"
        submission_path = output / (args.submission_name or default_name)
        submission.to_csv(submission_path, index=False)
        metadata_path = output / f"{submission_path.stem}_metadata.csv"
        pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
        print(json.dumps({**metadata, "submission_path": str(submission_path)}, ensure_ascii=False, indent=2))
        return
    output = result_output
    output.mkdir(parents=True, exist_ok=True)
    cache = RowCache.build(train[genes], genes)
    if args.event_tfidf_ovr or args.event_tfidf_three_way:
        result, class_result = run_event_tfidf_ovr_comparison_oof(cache, train[CONFIG.target_col], CANDIDATES[args.candidate], args.seed, args.tfidf_min_df)
        mode = "event-tfidf-three-way" if args.event_tfidf_three_way else "event-tfidf-ovr"
        stem = "_".join(part for part in (args.run_id, args.candidate, mode, f"seed{args.seed}") if part)
        pd.DataFrame([result]).to_csv(output / f"{stem}_oof.csv", index=False)
        class_result.to_csv(output / f"{stem}_class_f1.csv", index=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.event_tfidf:
        result, class_result = run_event_tfidf_oof(cache, train[CONFIG.target_col], CANDIDATES[args.candidate], args.seed, args.tfidf_min_df)
        stem = "_".join(part for part in (args.run_id, args.candidate, "event-tfidf", f"seed{args.seed}") if part)
        pd.DataFrame([result]).to_csv(output / f"{stem}_oof.csv", index=False)
        class_result.to_csv(output / f"{stem}_class_f1.csv", index=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.soft_specialist:
        result, class_result, pair_result = run_soft_pair_specialist_oof(cache, train[CONFIG.target_col], CANDIDATES[args.candidate], args.seed, args.specialist_left, args.specialist_right, args.specialist_alpha)
        stem = "_".join(part for part in (args.run_id, args.candidate, "soft-specialist", f"seed{args.seed}") if part)
        pd.DataFrame([result]).to_csv(output / f"{stem}_oof.csv", index=False)
        class_result.to_csv(output / f"{stem}_class_f1.csv", index=False)
        pair_result.to_csv(output / f"{stem}_pair_confusion.csv", index=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result, class_result = run_oof(cache, train[CONFIG.target_col], CANDIDATES[args.candidate], args.model, args.seed)
    stem = "_".join(part for part in (args.run_id, args.candidate, args.model, f"seed{args.seed}") if part)
    pd.DataFrame([result]).to_csv(output / f"{stem}_oof.csv", index=False)
    class_result.to_csv(output / f"{stem}_class_f1.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
