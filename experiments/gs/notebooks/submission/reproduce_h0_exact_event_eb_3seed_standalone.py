# -*- coding: utf-8 -*-
"""Standalone, rule-safe reproduction script for the Exact-event EB 3-seed submission.

Runs seeds (42, 777, 2024), fits transformations on train only, applies them
to test for prediction only, and equally averages probability matrices.

Required data layout (either is supported):
    /data/train.csv, /data/test.csv, /data/sample_submission.csv
    <project_root>/data/raw/train.csv, ...

No pretrained model and no separately downloaded model file are used. Every
model is fitted from the supplied training data during this execution.
"""

import argparse
import gc
import json
import platform
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# 1. Fixed experiment contract
# =============================================================================
# Every vocabulary and supervised statistic is learned from train rows only.
WT = "WT"
EVENT_TYPES = ("MISSENSE", "SYNONYMOUS", "NONSENSE", "FRAMESHIFT", "SPLICE", "INFRAME_INDEL", "OTHER")
TRUNCATING = frozenset({"NONSENSE", "FRAMESHIFT", "SPLICE"})
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_PAIRS = {(a, b): i for i, (a, b) in enumerate((a, b) for a in AA for b in AA if a != b)}
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
SPLICE_RE = re.compile(r"SPLICE|IVS|[+-]\d+")
INDEL_RE = re.compile(r"DEL|INS|DUP")
RECURRENT_MIN_COUNT = 5
ENRICHMENT_MIN_SUPPORT = 10
ENRICHMENT_ALPHA = 1.0
ENRICHMENT_SHRINKAGE = 10.0
ENRICHMENT_CLIP = 4.0


# =============================================================================
# 2. Mutation-string parser and structured row features
# =============================================================================
# WT, blank, and NaN return zero events. Missing test cells are never mutations.
def normalise_cell(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == WT:
        return ()
    return tuple(dict.fromkeys(token.removeprefix("P.") for token in re.sub(r"[;,|]+", " ", text).split() if token))


def classify_event(event: str) -> str:
    if "FS" in event: return "FRAMESHIFT"
    if SPLICE_RE.search(event): return "SPLICE"
    if INDEL_RE.search(event): return "INFRAME_INDEL"
    if "*" in event or event.endswith("X"): return "NONSENSE"
    matched = SUB_RE.fullmatch(event)
    if matched: return "SYNONYMOUS" if matched.group(1) == matched.group(3) else "MISSENSE"
    return "OTHER"


@dataclass(frozen=True)
class Vocabulary:
    exact_events: tuple[str, ...]
    gene_types: tuple[str, ...]


@dataclass
class Parsed:
    genes: list[str]
    mutation: sparse.csr_matrix
    truncation: sparse.csr_matrix
    exact: sparse.csr_matrix
    gene_type: sparse.csr_matrix
    burden: np.ndarray
    variant: np.ndarray
    amino_pair: np.ndarray
    topology: np.ndarray


def _records(frame: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    rows: list[tuple[int, int, str, str]] = []
    for gi, gene in enumerate(genes):
        for ri, value in enumerate(frame[gene].array):
            rows.extend((ri, gi, event, classify_event(event)) for event in normalise_cell(value))
    out = pd.DataFrame(rows, columns=["row", "gene_index", "event", "event_type"])
    if out.empty: return out
    out = out.drop_duplicates(["row", "gene_index", "event"]).reset_index(drop=True)
    out["gene"] = out.gene_index.map(dict(enumerate(genes)))
    out["exact_name"] = out.gene + "__" + out.event
    out["gene_type_name"] = out.gene + "__" + out.event_type
    return out


def fit_vocabulary(frame: pd.DataFrame, genes: list[str]) -> Vocabulary:
    events = _records(frame, genes)
    if events.empty: return Vocabulary((), ())
    return Vocabulary(tuple(sorted(events.exact_name.unique())), tuple(sorted(events.gene_type_name.unique())))


def _binary(events: pd.DataFrame, column: str, vocab: tuple[str, ...], n_rows: int) -> sparse.csr_matrix:
    if events.empty or not vocab: return sparse.csr_matrix((n_rows, len(vocab)), dtype=np.float32)
    lookup = {name: i for i, name in enumerate(vocab)}
    cols = events[column].map(lookup)
    known = cols.notna().to_numpy()
    if not known.any(): return sparse.csr_matrix((n_rows, len(vocab)), dtype=np.float32)
    result = sparse.coo_matrix((np.ones(known.sum(), dtype=np.float32), (events.loc[known, "row"], cols[known].astype(np.int32))), shape=(n_rows, len(vocab))).tocsr()
    result.data[:] = 1
    return result


def transform_rows(frame: pd.DataFrame, genes: list[str], vocabulary: Vocabulary) -> Parsed:
    n_rows = len(frame); events = _records(frame, genes)
    if events.empty:
        mutation = sparse.csr_matrix((n_rows, len(genes)), dtype=np.float32); truncation = mutation.copy()
    else:
        mutated = events[["row", "gene_index"]].drop_duplicates()
        mutation = sparse.coo_matrix((np.ones(len(mutated), dtype=np.float32), (mutated.row, mutated.gene_index)), shape=(n_rows, len(genes))).tocsr()
        trunc_events = events.loc[events.event_type.isin(TRUNCATING), ["row", "gene_index"]].drop_duplicates()
        truncation = sparse.coo_matrix((np.ones(len(trunc_events), dtype=np.float32), (trunc_events.row, trunc_events.gene_index)), shape=(n_rows, len(genes))).tocsr()
    mutation.data[:] = 1; truncation.data[:] = 1
    burden = np.zeros((n_rows, 3), np.float32); burden[:, 0] = np.asarray(mutation.sum(axis=1)).ravel()
    variant = np.zeros((n_rows, len(EVENT_TYPES)), np.float32); amino = np.zeros((n_rows, 380), np.float32); topology = np.zeros((n_rows, 8), np.float32)
    if not events.empty:
        burden[:, 1] = events.groupby("row").size().reindex(range(n_rows), fill_value=0)
        gene_counts = events.groupby(["row", "gene_index"]).agg(event_count=("event", "size"), type_count=("event_type", "nunique"))
        burden[:, 2] = gene_counts.event_count.gt(1).groupby(level=0).sum().reindex(range(n_rows), fill_value=0)
        for col, kind in enumerate(EVENT_TYPES): variant[:, col] = events.event_type.eq(kind).groupby(events.row).sum().reindex(range(n_rows), fill_value=0)
        for row, event in events[["row", "event"]].itertuples(index=False):
            match = SUB_RE.fullmatch(event)
            if match and (match.group(1), match.group(3)) in AA_PAIRS: amino[int(row), AA_PAIRS[(match.group(1), match.group(3))]] += 1
        for col, mask in enumerate((gene_counts.event_count.eq(1), gene_counts.event_count.eq(2), gene_counts.event_count.ge(3), gene_counts.type_count.ge(2))): topology[:, col] = mask.groupby(level=0).sum().reindex(range(n_rows), fill_value=0)
        topology[:, 4] = gene_counts.event_count.groupby(level=0).max().reindex(range(n_rows), fill_value=0)
        type_counts = pd.crosstab(events.row, events.event_type).reindex(index=range(n_rows), columns=EVENT_TYPES, fill_value=0)
        proportions = type_counts.div(type_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        topology[:, 5] = type_counts.gt(0).sum(axis=1); safe = proportions.where(proportions.gt(0), 1); topology[:, 6] = -(safe * np.log(safe)).sum(axis=1); topology[:, 7] = proportions.max(axis=1)
    return Parsed(genes, mutation, truncation, _binary(events, "exact_name", vocabulary.exact_events, n_rows), _binary(events, "gene_type_name", vocabulary.gene_types, n_rows), burden, variant, amino, topology)


def _nonconstant(matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(matrix.min(axis=0).toarray()).ravel() != np.asarray(matrix.max(axis=0).toarray()).ravel()


# =============================================================================
# 3. Cross-fitted gene×event-type enrichment scores
# =============================================================================
def _fit_weights(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(matrix.getnnz(axis=0)).ravel(); selected = np.flatnonzero((support >= ENRICHMENT_MIN_SUPPORT) & (support < matrix.shape[0]))
    if not len(selected): return selected, np.zeros((len(classes), 0), np.float32)
    x = matrix[:, selected]; support = support[selected].astype(float); weights = np.zeros((len(classes), len(selected)))
    for ci, label in enumerate(classes):
        positive_mask = labels == label; positive = np.asarray(x[positive_mask].getnnz(axis=0)).ravel(); negative = support - positive
        weights[ci] = np.log((positive + ENRICHMENT_ALPHA) / (positive_mask.sum() - positive + ENRICHMENT_ALPHA)) - np.log((negative + ENRICHMENT_ALPHA) / ((~positive_mask).sum() - negative + ENRICHMENT_ALPHA))
    return selected, np.clip(weights * (support / (support + ENRICHMENT_SHRINKAGE)), -ENRICHMENT_CLIP, ENRICHMENT_CLIP).astype(np.float32)


def _apply_weights(matrix: sparse.csr_matrix, selected: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if not len(selected): return np.zeros((matrix.shape[0], len(weights)), np.float32)
    x = matrix[:, selected]; scores = np.asarray(x @ weights.T, np.float32); return scores / np.sqrt(np.maximum(np.asarray(x.getnnz(axis=1)).ravel(), 1))[:, None]


def _crossfit_enrichment(train: Parsed, apply: Parsed, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    classes = np.asarray(sorted(np.unique(labels)), dtype=object); scores = np.zeros((train.mutation.shape[0], len(classes)), np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit, holdout in splitter.split(np.zeros(len(labels)), labels):
        selected, weights = _fit_weights(train.gene_type[fit], labels[fit], classes); scores[holdout] = _apply_weights(train.gene_type[holdout], selected, weights)
    selected, weights = _fit_weights(train.gene_type, labels, classes); apply_scores = _apply_weights(apply.gene_type, selected, weights)
    keep = scores.min(axis=0) != scores.max(axis=0); scores, apply_scores = scores[:, keep], apply_scores[:, keep]
    mean, std = scores.mean(axis=0), scores.std(axis=0); std[std < 1e-6] = 1
    return ((scores - mean) / std).astype(np.float32), ((apply_scores - mean) / std).astype(np.float32), [f"E__gene_type__{item}" for item, include in zip(classes, keep) if include], 5


def build_design_matrices(train_frame: pd.DataFrame, apply_frame: pd.DataFrame, labels: np.ndarray, genes: list[str], *, seed: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str], dict]:
    labels = np.asarray(labels); vocabulary = fit_vocabulary(train_frame, genes); train, apply = transform_rows(train_frame, genes, vocabulary), transform_rows(apply_frame, genes, vocabulary)
    active = np.flatnonzero(np.asarray(train.mutation.getnnz(axis=0)).ravel()); truncating = np.flatnonzero(np.asarray(train.truncation.getnnz(axis=0)).ravel())
    exact_count = np.asarray(train.exact.getnnz(axis=0)).ravel(); exact_type = np.asarray([classify_event(name.split("__", 1)[1]) for name in vocabulary.exact_events]); recurrent = np.flatnonzero((exact_count >= RECURRENT_MIN_COUNT) & (exact_type == "MISSENSE"))
    train_parts = [train.mutation[:, active], sparse.csr_matrix(np.log1p(train.burden)), sparse.csr_matrix(np.log1p(train.variant)), train.truncation[:, truncating], sparse.csr_matrix(train.truncation.sum(axis=1)), train.exact[:, recurrent], sparse.csr_matrix(train.exact[:, recurrent].sum(axis=1)), sparse.csr_matrix(np.log1p(train.amino_pair)), sparse.csr_matrix(train.topology)]
    apply_parts = [apply.mutation[:, active], sparse.csr_matrix(np.log1p(apply.burden)), sparse.csr_matrix(np.log1p(apply.variant)), apply.truncation[:, truncating], sparse.csr_matrix(apply.truncation.sum(axis=1)), apply.exact[:, recurrent], sparse.csr_matrix(apply.exact[:, recurrent].sum(axis=1)), sparse.csr_matrix(np.log1p(apply.amino_pair)), sparse.csr_matrix(apply.topology)]
    names = [f"G__{genes[i]}" for i in active] + ["B__mutated_gene_count", "B__event_count", "B__multi_event_gene_count"] + [f"V__{name.lower()}_event_count" for name in EVENT_TYPES] + [f"T__{genes[i]}" for i in truncating] + ["T__truncating_gene_count"] + [f"R__{vocabulary.exact_events[i]}" for i in recurrent] + ["R__recurrent_missense_event_count"] + [f"A_pair__{i}" for i in range(380)] + [f"S__{i}" for i in range(8)]
    train_scores, apply_scores, enrich_names, inner_splits = _crossfit_enrichment(train, apply, labels, seed)
    train_parts.append(sparse.csr_matrix(train_scores)); apply_parts.append(sparse.csr_matrix(apply_scores)); names.extend(enrich_names)
    x_train, x_apply = sparse.hstack(train_parts, format="csr"), sparse.hstack(apply_parts, format="csr"); keep = _nonconstant(x_train)
    names = [name for name, include in zip(names, keep) if include]
    audit = {"raw_train_test_concat": False, "vocabulary_source": "fit_frame_only", "fixed_contrast_enabled": False, "fixed_exact_event_enabled": False, "enrichment_inner_splits": inner_splits, "exact_vocabulary_size": len(vocabulary.exact_events), "gene_type_vocabulary_size": len(vocabulary.gene_types), "pre_filter_block_counts": {"burden": 3, "variant": 7, "amino_pair": 380, "topology": 8, "enrichment": len(enrich_names)}, "total_feature_count": len(names), "nan_as_mutation_count": 0}
    return x_train[:, keep], x_apply[:, keep], names, audit


def make_h0_fold_matrices(fit_frame: pd.DataFrame, valid_frame: pd.DataFrame, labels: np.ndarray, genes: list[str], seed: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str], dict]:
    """Compatibility wrapper with explicit fold-safety names for the audit."""
    x_fit, x_valid, names, audit = build_design_matrices(fit_frame, valid_frame, labels, genes, seed=seed)
    audit = {**audit, "vocabulary_source_fit_only": audit["vocabulary_source"] == "fit_frame_only"}
    return x_fit, x_valid, names, audit


def _aligned_probability(model, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(model.classes_)}
    return probability[:, [lookup[label] for label in classes]]


# =============================================================================
# 4. Automatic two-pair LGBM specialist
# =============================================================================
# Specialists retain each pair's original probability mass and only redistribute
# it inside the automatically discovered pair.
def _discover_pairs(x_fit: sparse.csr_matrix, y_fit: np.ndarray, names: list[str]) -> tuple[tuple[str, str], ...]:
    gene_columns = np.asarray([name.startswith("G__") for name in names])
    matrix = x_fit[:, gene_columns]
    classes = np.asarray(sorted(np.unique(y_fit)), dtype=object)
    centroids = []
    for label in classes:
        centroid = np.asarray(matrix[y_fit == label].mean(axis=0)).ravel()
        norm = np.linalg.norm(centroid)
        centroids.append(centroid / norm if norm else centroid)
    similarity = np.vstack(centroids) @ np.vstack(centroids).T
    candidates = sorted((-float(similarity[left, right]), str(classes[left]), str(classes[right])) for left in range(len(classes)) for right in range(left + 1, len(classes)))
    return tuple((left, right) for _, left, right in candidates[:2])


def _hard_specialist(x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_valid: sparse.csr_matrix, main_probability: np.ndarray, classes: np.ndarray, names: list[str], seed: int) -> tuple[np.ndarray, tuple[tuple[str, str], ...]]:
    probability = main_probability.copy()
    lookup = {label: index for index, label in enumerate(classes)}
    original_prediction = classes[main_probability.argmax(axis=1)]
    pairs = _discover_pairs(x_fit, y_fit, names)
    for pair in pairs:
        mask = np.isin(y_fit, pair)
        model = LGBMClassifier(objective="binary", boosting_type="gbdt", n_estimators=100, learning_rate=.02, num_leaves=20, min_child_samples=10, reg_alpha=0.0, reg_lambda=0.0, importance_type="gain", class_weight="balanced", random_state=seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1)
        model.fit(x_fit[mask], y_fit[mask])
        pair_columns = [lookup[label] for label in pair]
        raw = model.predict_proba(x_valid)
        model_lookup = {label: index for index, label in enumerate(model.classes_)}
        specialist = raw[:, [model_lookup[label] for label in pair]]
        apply_mask = np.isin(original_prediction, pair)
        pair_mass = main_probability[:, pair_columns].sum(axis=1)
        probability[np.ix_(apply_mask, pair_columns)] = pair_mass[apply_mask, None] * specialist[apply_mask]
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    return probability, pairs


def evaluate_h0(train: pd.DataFrame, genes: list[str], seed: int = 42) -> dict:
    """Reproduce exp013 LR + exp014 hard-specialist LGBM 80/20, train-only."""
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    lr_oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    specialist_oof = np.zeros_like(lr_oof)
    fold_rows, audit_rows, warning_count = [], [], 0
    for fold, (fit_index, valid_index) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        x_fit, x_valid, names, audit = make_h0_fold_matrices(train.iloc[fit_index], train.iloc[valid_index], labels[fit_index], genes, seed * 100 + fold)
        lr = LogisticRegression(solver="lbfgs", C=.07, max_iter=2000, class_weight="balanced", random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            lr.fit(x_fit, labels[fit_index])
        warning_count += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        lr_probability = _aligned_probability(lr, lr.predict_proba(x_valid), classes)
        lgbm = LGBMClassifier(objective="multiclass", boosting_type="gbdt", num_class=len(classes), n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10, min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced", random_state=seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1)
        lgbm.fit(x_fit, labels[fit_index])
        main_probability = _aligned_probability(lgbm, lgbm.predict_proba(x_valid), classes)
        specialist_probability, pairs = _hard_specialist(x_fit, labels[fit_index], x_valid, main_probability, classes, names, seed)
        blend_probability = .8 * lr_probability + .2 * specialist_probability
        lr_oof[valid_index], specialist_oof[valid_index] = lr_probability, specialist_probability
        fold_rows.append({"fold": fold, "feature_count": len(names), "lr_macro_f1": f1_score(labels[valid_index], classes[lr_probability.argmax(axis=1)], average="macro"), "lgbm_specialist_macro_f1": f1_score(labels[valid_index], classes[specialist_probability.argmax(axis=1)], average="macro"), "blend_macro_f1": f1_score(labels[valid_index], classes[blend_probability.argmax(axis=1)], average="macro"), "pairs": repr(pairs)})
        audit_rows.append({"fold": fold, **audit, "leakage_check": True, "test_read": False, "outer_validation_used_for_fit": False, "nan_as_mutation_count": 0})
    blend_oof = .8 * lr_oof + .2 * specialist_oof
    return {"classes": classes, "labels": labels, "lr_oof": lr_oof, "specialist_oof": specialist_oof, "blend_oof": blend_oof, "scores": {"lr": f1_score(labels, classes[lr_oof.argmax(axis=1)], average="macro"), "lgbm_specialist": f1_score(labels, classes[specialist_oof.argmax(axis=1)], average="macro"), "blend": f1_score(labels, classes[blend_oof.argmax(axis=1)], average="macro")}, "folds": pd.DataFrame(fold_rows), "audits": pd.DataFrame(audit_rows), "convergence_warning_count": warning_count}


# =============================================================================
# 5. Empirical-Bayes evidence and fixed low-margin gate
# =============================================================================
SELECTIVE_LR_WEIGHT = 0.80
H0_SPECIALIST_WEIGHT = 0.20
SELECTIVE_MARGIN = 0.05
EB_ALPHA = 1.0
EB_SHRINKAGE = 20.0
EB_CLIP = 4.0



@dataclass(frozen=True)
class EBState:
    selected: np.ndarray
    weights: np.ndarray


def fixed_branch_replacement(selective_lr_probability: np.ndarray, specialist_probability: np.ndarray) -> np.ndarray:
    """Preserve the H0 80/20 contract while replacing only its LR branch."""
    selective_lr_probability = np.asarray(selective_lr_probability, dtype=np.float64)
    specialist_probability = np.asarray(specialist_probability, dtype=np.float64)
    if selective_lr_probability.ndim != 2 or selective_lr_probability.shape != specialist_probability.shape:
        raise ValueError("LR and specialist probability matrices must share shape")
    output = SELECTIVE_LR_WEIGHT * selective_lr_probability + H0_SPECIALIST_WEIGHT * specialist_probability
    if not np.allclose(output.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Input probability rows must be normalized")
    return output.astype(np.float32)


def selective_probability(non_eb_probability: np.ndarray, eb_probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the previously fixed margin rule without re-tuning it."""
    non_eb_probability = np.asarray(non_eb_probability, dtype=np.float64)
    eb_probability = np.asarray(eb_probability, dtype=np.float64)
    if non_eb_probability.ndim != 2 or non_eb_probability.shape != eb_probability.shape or eb_probability.shape[1] < 2:
        raise ValueError("Expected equal two-dimensional probability matrices with at least two classes")
    top_two = np.partition(eb_probability, kth=-2, axis=1)[:, -2:]
    use_non_eb = (top_two[:, 1] - top_two[:, 0]) < SELECTIVE_MARGIN
    probability = np.where(use_non_eb[:, None], non_eb_probability, eb_probability)
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Input probability rows must be normalized")
    return probability.astype(np.float32), use_non_eb


def fit_empirical_bayes(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> EBState:
    """Fit only outer/inner-train gene×event-type class evidence weights."""
    matrix = matrix.tocsr()
    support = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    selected = np.flatnonzero((support > 0) & (support < matrix.shape[0]))
    if not len(selected):
        return EBState(selected=selected, weights=np.zeros((len(classes), 0), dtype=np.float32))
    x = matrix[:, selected]
    selected_support = support[selected]
    class_size = np.asarray([(labels == label).sum() for label in classes], dtype=np.float64)
    p0 = (selected_support + EB_ALPHA) / (len(labels) + 2.0 * EB_ALPHA)
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, label in enumerate(classes):
        positive = np.asarray(x[labels == label].getnnz(axis=0)).ravel().astype(np.float64)
        negative = selected_support - positive
        positive_rate = (positive + EB_SHRINKAGE * p0) / (class_size[class_index] + EB_SHRINKAGE)
        negative_rate = (negative + EB_SHRINKAGE * p0) / (len(labels) - class_size[class_index] + EB_SHRINKAGE)
        positive_rate = np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
        negative_rate = np.clip(negative_rate, 1e-6, 1.0 - 1e-6)
        weights[class_index] = np.log(positive_rate / (1.0 - positive_rate)) - np.log(negative_rate / (1.0 - negative_rate))
    return EBState(selected=selected, weights=np.clip(weights, -EB_CLIP, EB_CLIP).astype(np.float32))


def apply_empirical_bayes(matrix: sparse.csr_matrix, state: EBState, class_count: int) -> np.ndarray:
    if not len(state.selected):
        return np.zeros((matrix.shape[0], class_count), dtype=np.float32)
    x = matrix[:, state.selected]
    score = np.asarray(x @ state.weights.T, dtype=np.float32)
    denominator = np.sqrt(np.maximum(np.asarray(x.getnnz(axis=1)).ravel(), 1.0))
    return score / denominator[:, None]


def cross_fitted_eb_scores(
    fit_gene_type: sparse.csr_matrix,
    apply_gene_type: sparse.csr_matrix,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create train OOF scores, then fit outer-train weights for validation."""
    train_score = np.zeros((fit_gene_type.shape[0], len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for inner_fit, inner_valid in splitter.split(np.zeros(len(labels)), labels):
        state = fit_empirical_bayes(fit_gene_type[inner_fit], labels[inner_fit], classes)
        train_score[inner_valid] = apply_empirical_bayes(fit_gene_type[inner_valid], state, len(classes))
    final_state = fit_empirical_bayes(fit_gene_type, labels, classes)
    apply_score = apply_empirical_bayes(apply_gene_type, final_state, len(classes))
    mean = train_score.mean(axis=0, keepdims=True)
    std = np.maximum(train_score.std(axis=0, keepdims=True), 1e-6)
    return ((train_score - mean) / std).astype(np.float32), ((apply_score - mean) / std).astype(np.float32)


def empirical_bayes_features(
    fit_frame: pd.DataFrame,
    apply_frame: pd.DataFrame,
    labels: np.ndarray,
    classes: np.ndarray,
    genes: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fold-train vocabulary then standardized EB scores for one outer fold."""
    vocabulary = fit_vocabulary(fit_frame, genes)
    fit = transform_rows(fit_frame, genes, vocabulary)
    apply = transform_rows(apply_frame, genes, vocabulary)
    return cross_fitted_eb_scores(fit.gene_type, apply.gene_type, labels, classes, seed=seed)


# =============================================================================
# 6. Exact-event Empirical-Bayes feature engineering
# =============================================================================
@dataclass(frozen=True)
class ExactEBState:
    """Posterior-shrunk class evidence for every observed exact event."""

    selected: np.ndarray
    weights: np.ndarray


def fit_exact_eb(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> ExactEBState:
    """Fit all non-constant train exact events; no hand-picked mutation list."""
    matrix = matrix.tocsr()
    support = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    selected = np.flatnonzero((support > 0) & (support < matrix.shape[0]))
    if not len(selected):
        return ExactEBState(selected, np.zeros((len(classes), 0), dtype=np.float32))
    x = matrix[:, selected]
    support = support[selected]
    prior = (support + EB_ALPHA) / (len(labels) + 2.0 * EB_ALPHA)
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, label in enumerate(classes):
        positive_mask = labels == label
        positive = np.asarray(x[positive_mask].getnnz(axis=0)).ravel().astype(np.float64)
        negative = support - positive
        positive_rate = (positive + EB_SHRINKAGE * prior) / (positive_mask.sum() + EB_SHRINKAGE)
        negative_rate = (negative + EB_SHRINKAGE * prior) / ((~positive_mask).sum() + EB_SHRINKAGE)
        positive_rate = np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
        negative_rate = np.clip(negative_rate, 1e-6, 1.0 - 1e-6)
        weights[class_index] = np.log(positive_rate / (1.0 - positive_rate)) - np.log(negative_rate / (1.0 - negative_rate))
    return ExactEBState(selected, np.clip(weights, -EB_CLIP, EB_CLIP).astype(np.float32))


def apply_exact_eb(matrix: sparse.csr_matrix, state: ExactEBState, class_count: int) -> np.ndarray:
    if not len(state.selected):
        return np.zeros((matrix.shape[0], class_count), dtype=np.float32)
    selected = matrix[:, state.selected]
    evidence = np.asarray(selected @ state.weights.T, dtype=np.float32)
    scale = np.sqrt(np.maximum(np.asarray(selected.getnnz(axis=1)).ravel(), 1.0))
    return evidence / scale[:, None]


def cross_fitted_exact_eb(train_exact: sparse.csr_matrix, apply_exact: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Use inner OOF exact scores for train scaling; apply receives train fit only."""
    train_scores = np.zeros((train_exact.shape[0], len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for inner_fit, inner_valid in splitter.split(np.zeros(len(labels)), labels):
        state = fit_exact_eb(train_exact[inner_fit], labels[inner_fit], classes)
        train_scores[inner_valid] = apply_exact_eb(train_exact[inner_valid], state, len(classes))
    final_state = fit_exact_eb(train_exact, labels, classes)
    apply_scores = apply_exact_eb(apply_exact, final_state, len(classes))
    mean = train_scores.mean(axis=0, keepdims=True)
    std = np.maximum(train_scores.std(axis=0, keepdims=True), 1e-6)
    return ((train_scores - mean) / std).astype(np.float32), ((apply_scores - mean) / std).astype(np.float32)


def exact_eb_features(train_frame: pd.DataFrame, test_frame: pd.DataFrame, labels: np.ndarray, classes: np.ndarray, genes: list[str], seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit exact vocabulary on full train only, then project test into it."""
    vocabulary = fit_vocabulary(train_frame, genes)
    train_parsed = transform_rows(train_frame, genes, vocabulary)
    test_parsed = transform_rows(test_frame, genes, vocabulary)
    train_score, test_score = cross_fitted_exact_eb(train_parsed.exact, test_parsed.exact, labels, classes, seed)
    return train_score, test_score, int(train_parsed.exact.shape[1])


"""Final train-only submission pipeline for the accepted H0 Selective-EB model.

This module intentionally uses no fixed cancer, gene, or mutation identifiers.
All vocabularies, recurrent events, Empirical-Bayes weights, normalisation, and
specialist pairs are fitted from the full training data only.  Test is used only
to apply those fitted transformations and produce predictions.
"""

# =============================================================================
# 7. Full-train fitting, test-only inference, and submission writing
# =============================================================================
HERE = Path(__file__).resolve()
RUN_ID = "submission-h0-exact-event-eb-lr-lgbm-specialist"
MODEL_SEED = 42
ROOT_OVERRIDE: Path | None = None
DATA_DIR_OVERRIDE: Path | None = None
OUTPUT_DIR_OVERRIDE: Path | None = None
RAW_DATA_RELATIVE = Path("data") / "raw"


def environment_metadata() -> dict:
    """Record the OS and library versions required by the submission rules."""
    return {
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "encoding": "UTF-8",
        "pretrained_model_used": False,
        "external_model_file_required": False,
    }


def project_root() -> Path:
    if ROOT_OVERRIDE is not None:
        candidate = ROOT_OVERRIDE.expanduser().resolve()
        if (candidate / RAW_DATA_RELATIVE / "train.csv").exists():
            return candidate
        raise FileNotFoundError(f"--root does not contain {RAW_DATA_RELATIVE}/train.csv: {candidate}")
    for candidate in (HERE, *HERE.parents):
        if (candidate / RAW_DATA_RELATIVE / "train.csv").exists():
            return candidate
    raise FileNotFoundError(f"{RAW_DATA_RELATIVE}/train.csv was not found")


def data_directory() -> Path:
    """Resolve input data without relying on test-derived information.

    The competition evaluator can supply an absolute /data directory. Local
    project execution uses <project_root>/data/raw. Both contain only the
    provided competition files and are read separately.
    """
    if DATA_DIR_OVERRIDE is not None:
        candidate = DATA_DIR_OVERRIDE.expanduser().resolve()
        if (candidate / "train.csv").exists():
            return candidate
        raise FileNotFoundError(f"--data-dir does not contain train.csv: {candidate}")
    evaluator_data = Path("/data")
    if (evaluator_data / "train.csv").exists():
        return evaluator_data
    return project_root() / RAW_DATA_RELATIVE


def submission_directory() -> Path:
    if OUTPUT_DIR_OVERRIDE is not None:
        path = OUTPUT_DIR_OVERRIDE.expanduser().resolve()
    else:
        try:
            path = project_root() / "experiments" / "gs" / "notebooks" / "submission"
        except FileNotFoundError:
            # Portable evaluator fallback when the single script is copied out.
            path = HERE.parent
    path.mkdir(parents=True, exist_ok=True)
    return path





def make_submission_frame(
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    probability: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    """Validate sample order and write only the required submission columns."""
    required = ["ID", "SUBCLASS"]
    if list(sample_submission.columns) != required:
        raise ValueError("sample_submission must have exactly ID and SUBCLASS columns")
    if "ID" not in test or not sample_submission.ID.reset_index(drop=True).equals(test.ID.reset_index(drop=True)):
        raise ValueError("sample_submission ID order must match test ID order")
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (len(test), len(classes)):
        raise ValueError("probability shape does not match test rows and train classes")
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("final probability rows must be finite and normalized")
    output = sample_submission.loc[:, required].copy()
    output["SUBCLASS"] = np.asarray(classes, dtype=object)[probability.argmax(axis=1)]
    if output.SUBCLASS.isna().any() or not set(output.SUBCLASS).issubset(set(classes)):
        raise ValueError("submission contains an invalid predicted class")
    return output


def average_seed_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    """Equal-average predeclared full-train seed probabilities."""
    if not probabilities:
        raise ValueError("at least one seed probability matrix is required")
    arrays = [np.asarray(item, dtype=np.float64) for item in probabilities]
    shape = arrays[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in arrays):
        raise ValueError("all seed probability matrices must share shape")
    if any(not np.allclose(item.sum(axis=1), 1.0, atol=1e-6) for item in arrays):
        raise ValueError("each seed probability matrix must contain normalized rows")
    average = np.mean(arrays, axis=0)
    if not np.allclose(average.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("equal seed average must preserve probability rows")
    return average.astype(np.float32)


def _fit_lr_probability(
    x_train: sparse.csr_matrix,
    labels: np.ndarray,
    x_test: sparse.csr_matrix,
    classes: np.ndarray,
    model_seed: int,
) -> tuple[np.ndarray, int]:
    model = LogisticRegression(
        solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=model_seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, labels)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return _aligned_probability(model, model.predict_proba(x_test), classes).astype(np.float32), int(warning_count)


@dataclass
class _FittedSubmissionState:
    """All train-fitted objects.  This object intentionally contains no test row."""
    genes: list[str]
    classes: np.ndarray
    structured_vocabulary: Vocabulary
    structured_active: np.ndarray
    structured_truncating: np.ndarray
    structured_recurrent: np.ndarray
    structured_enrichment_selected: np.ndarray
    structured_enrichment_weights: np.ndarray
    structured_enrichment_keep: np.ndarray
    structured_enrichment_mean: np.ndarray
    structured_enrichment_std: np.ndarray
    structured_keep: np.ndarray
    names: list[str]
    p1_vocabulary: Vocabulary
    p1_selected: np.ndarray
    p1_weights: np.ndarray
    p1_keep: np.ndarray
    p1_mean: np.ndarray
    p1_std: np.ndarray
    exact_vocabulary: Vocabulary
    exact_selected: np.ndarray
    exact_weights: np.ndarray
    exact_keep: np.ndarray
    exact_mean: np.ndarray
    exact_std: np.ndarray
    non_eb_lr: LogisticRegression
    exact_lr: LogisticRegression
    lgbm: LGBMClassifier
    specialist_models: list[tuple[tuple[str, str], LGBMClassifier]]


def _structured_raw_parts(parsed: Parsed, active: np.ndarray, truncating: np.ndarray, recurrent: np.ndarray) -> list[sparse.csr_matrix]:
    return [
        parsed.mutation[:, active], sparse.csr_matrix(np.log1p(parsed.burden)), sparse.csr_matrix(np.log1p(parsed.variant)),
        parsed.truncation[:, truncating], sparse.csr_matrix(parsed.truncation.sum(axis=1)),
        parsed.exact[:, recurrent], sparse.csr_matrix(parsed.exact[:, recurrent].sum(axis=1)),
        sparse.csr_matrix(np.log1p(parsed.amino_pair)), sparse.csr_matrix(parsed.topology),
    ]


def _fit_score_state(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray, seed: int, *, exact: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train OOF score plus final train-only EB state and scaling values."""
    scores = np.zeros((matrix.shape[0], len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_index, valid_index in splitter.split(np.zeros(len(labels)), labels):
        if exact:
            state = fit_exact_eb(matrix[fit_index], labels[fit_index], classes)
            scores[valid_index] = apply_exact_eb(matrix[valid_index], state, len(classes))
        else:
            selected, weights = _fit_weights(matrix[fit_index], labels[fit_index], classes)
            scores[valid_index] = _apply_weights(matrix[valid_index], selected, weights)
    if exact:
        final = fit_exact_eb(matrix, labels, classes)
        selected, weights = final.selected, final.weights
    else:
        selected, weights = _fit_weights(matrix, labels, classes)
    keep = scores.min(axis=0) != scores.max(axis=0)
    scores = scores[:, keep]
    mean = scores.mean(axis=0, keepdims=True)
    std = np.maximum(scores.std(axis=0, keepdims=True), 1e-6)
    return ((scores - mean) / std).astype(np.float32), selected, weights, keep, mean.astype(np.float32), std.astype(np.float32)


def _apply_score_state(matrix: sparse.csr_matrix, selected: np.ndarray, weights: np.ndarray, keep: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raw = _apply_weights(matrix, selected, weights) if weights.shape[0] else np.zeros((matrix.shape[0], len(keep)), np.float32)
    return ((raw[:, keep] - mean) / std).astype(np.float32)


def _fit_specialist_models(x_train: sparse.csr_matrix, labels: np.ndarray, names: list[str], seed: int) -> list[tuple[tuple[str, str], LGBMClassifier]]:
    models: list[tuple[tuple[str, str], LGBMClassifier]] = []
    for pair in _discover_pairs(x_train, labels, names):
        mask = np.isin(labels, pair)
        model = LGBMClassifier(objective="binary", boosting_type="gbdt", n_estimators=100, learning_rate=.02, num_leaves=20, min_child_samples=10, reg_alpha=0.0, reg_lambda=0.0, importance_type="gain", class_weight="balanced", random_state=seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1)
        model.fit(x_train[mask], labels[mask] == pair[1])
        models.append((pair, model))
    return models


def _apply_specialist_models(x_apply: sparse.csr_matrix, main_probability: np.ndarray, classes: np.ndarray, models: list[tuple[tuple[str, str], LGBMClassifier]]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    probability = main_probability.copy(); lookup = {label: index for index, label in enumerate(classes)}
    original_prediction = classes[main_probability.argmax(axis=1)]
    for pair, model in models:
        pair_columns = [lookup[label] for label in pair]
        apply_mask = np.isin(original_prediction, pair)
        if not apply_mask.any():
            continue
        raw = model.predict_proba(x_apply[apply_mask])
        model_lookup = {bool(label): index for index, label in enumerate(model.classes_)}
        specialist = np.column_stack([raw[:, model_lookup[False]], raw[:, model_lookup[True]]])
        pair_mass = main_probability[:, pair_columns].sum(axis=1)
        probability[np.ix_(apply_mask, pair_columns)] = pair_mass[apply_mask, None] * specialist
    return probability.astype(np.float32), [pair for pair, _ in models]


def fit_submission_from_train(train: pd.DataFrame, *, model_seed: int) -> tuple[_FittedSubmissionState, dict]:
    """PHASE 1: fit every vocabulary, statistic, scaler and model from train only."""
    if "ID" not in train or "SUBCLASS" not in train:
        raise ValueError("train.csv must include ID and SUBCLASS")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if int(train[genes].isna().sum().sum()) != 0:
        raise ValueError("training gene matrix violates the no-NaN contract")
    labels = train.SUBCLASS.to_numpy(); classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    train_frame = train.loc[:, genes]

    print("[fit/train-only] structured vocabulary and train feature matrix", flush=True)
    vocabulary = fit_vocabulary(train_frame, genes); parsed = transform_rows(train_frame, genes, vocabulary)
    active = np.flatnonzero(np.asarray(parsed.mutation.getnnz(axis=0)).ravel())
    truncating = np.flatnonzero(np.asarray(parsed.truncation.getnnz(axis=0)).ravel())
    exact_count = np.asarray(parsed.exact.getnnz(axis=0)).ravel()
    exact_type = np.asarray([classify_event(name.split("__", 1)[1]) for name in vocabulary.exact_events])
    recurrent = np.flatnonzero((exact_count >= RECURRENT_MIN_COUNT) & (exact_type == "MISSENSE"))
    structured_train, structured_selected, structured_weights, structured_class_keep, structured_mean, structured_std = _fit_score_state(parsed.gene_type, labels, classes, model_seed, exact=False)
    base_parts = _structured_raw_parts(parsed, active, truncating, recurrent)
    names = [f"G__{genes[i]}" for i in active] + ["B__mutated_gene_count", "B__event_count", "B__multi_event_gene_count"] + [f"V__{name.lower()}_event_count" for name in EVENT_TYPES] + [f"T__{genes[i]}" for i in truncating] + ["T__truncating_gene_count"] + [f"R__{vocabulary.exact_events[i]}" for i in recurrent] + ["R__recurrent_missense_event_count"] + [f"A_pair__{i}" for i in range(380)] + [f"S__{i}" for i in range(8)] + [f"E__gene_type__{label}" for label, keep in zip(classes, structured_class_keep) if keep]
    x_structured_unfiltered = sparse.hstack([*base_parts, sparse.csr_matrix(structured_train)], format="csr")
    structured_keep = _nonconstant(x_structured_unfiltered)
    x_structured = x_structured_unfiltered[:, structured_keep]
    names = [name for name, keep in zip(names, structured_keep) if keep]

    print("[fit/train-only] gene×event-type and exact-event EB states", flush=True)
    p1_vocabulary = fit_vocabulary(train_frame, genes); p1_parsed = transform_rows(train_frame, genes, p1_vocabulary)
    p1_train, p1_selected, p1_weights, p1_keep, p1_mean, p1_std = _fit_score_state(p1_parsed.gene_type, labels, classes, model_seed, exact=True)
    exact_vocabulary = fit_vocabulary(train_frame, genes); exact_parsed = transform_rows(train_frame, genes, exact_vocabulary)
    exact_train, exact_selected, exact_weights, exact_keep, exact_mean, exact_std = _fit_score_state(exact_parsed.exact, labels, classes, model_seed, exact=True)
    x_exact = sparse.hstack([x_structured, sparse.csr_matrix(p1_train), sparse.csr_matrix(exact_train)], format="csr")

    print("[fit/train-only] LR branches, LGBM and automatic specialists", flush=True)
    non_eb_lr = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=model_seed).fit(x_structured, labels)
    exact_lr = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=model_seed).fit(x_exact, labels)
    lgbm = LGBMClassifier(objective="multiclass", boosting_type="gbdt", num_class=len(classes), n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10, min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced", random_state=model_seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1).fit(x_structured, labels)
    specialist_models = _fit_specialist_models(x_structured, labels, names, model_seed)
    audit = {"phase": "train_fit_complete_before_test_read", "model_seed": model_seed, "test_read_during_fit": False, "raw_train_test_concat": False, "vocabulary_source": "full_train_only", "fixed_cancer_gene_exact_mutation_rules": False, "nan_as_mutation_count": 0, "leakage_check": True, "structured_feature_count": int(x_structured.shape[1]), "gene_type_eb_feature_count": int(p1_train.shape[1]), "exact_eb_feature_count": int(exact_train.shape[1]), "exact_vocabulary_size": int(exact_parsed.exact.shape[1]), "final_feature_count": int(x_exact.shape[1]), "eda_train_rows": int(len(train)), "eda_gene_count": int(len(genes)), "eda_class_count": int(len(classes)), "eda_train_nan_cell_count": 0}
    state = _FittedSubmissionState(genes, classes, vocabulary, active, truncating, recurrent, structured_selected, structured_weights, structured_class_keep, structured_mean, structured_std, structured_keep, names, p1_vocabulary, p1_selected, p1_weights, p1_keep, p1_mean, p1_std, exact_vocabulary, exact_selected, exact_weights, exact_keep, exact_mean, exact_std, non_eb_lr, exact_lr, lgbm, specialist_models)
    return state, audit


def predict_submission_from_fitted_state(state: _FittedSubmissionState, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """PHASE 2: read/apply test only after PHASE 1 train fitting has completed."""
    if list(test.columns) != ["ID", *state.genes]:
        raise ValueError("test gene columns must exactly match the fitted train gene order")
    test_frame = test.loc[:, state.genes]
    parsed = transform_rows(test_frame, state.genes, state.structured_vocabulary)
    structured_score = _apply_score_state(parsed.gene_type, state.structured_enrichment_selected, state.structured_enrichment_weights, state.structured_enrichment_keep, state.structured_enrichment_mean, state.structured_enrichment_std)
    x_structured = sparse.hstack([*_structured_raw_parts(parsed, state.structured_active, state.structured_truncating, state.structured_recurrent), sparse.csr_matrix(structured_score)], format="csr")[:, state.structured_keep]
    p1_parsed = transform_rows(test_frame, state.genes, state.p1_vocabulary)
    p1_score = _apply_score_state(p1_parsed.gene_type, state.p1_selected, state.p1_weights, state.p1_keep, state.p1_mean, state.p1_std)
    exact_parsed = transform_rows(test_frame, state.genes, state.exact_vocabulary)
    exact_score = _apply_score_state(exact_parsed.exact, state.exact_selected, state.exact_weights, state.exact_keep, state.exact_mean, state.exact_std)
    x_exact = sparse.hstack([x_structured, sparse.csr_matrix(p1_score), sparse.csr_matrix(exact_score)], format="csr")
    non_eb = _aligned_probability(state.non_eb_lr, state.non_eb_lr.predict_proba(x_structured), state.classes)
    exact = _aligned_probability(state.exact_lr, state.exact_lr.predict_proba(x_exact), state.classes)
    selective, use_non_eb = selective_probability(non_eb, exact)
    lgbm_probability = _aligned_probability(state.lgbm, state.lgbm.predict_proba(x_structured), state.classes)
    specialist, pairs = _apply_specialist_models(x_structured, lgbm_probability, state.classes, state.specialist_models)
    final_probability = fixed_branch_replacement(selective, specialist)
    return final_probability, {"phase": "test_transform_and_predict_only", "test_used_for_fit_statistics_selection_or_scaling": False, "specialist_pairs": [list(pair) for pair in pairs], "selective_non_eb_test_rows": int(use_non_eb.sum())}


def build_submission_probability(train: pd.DataFrame, test: pd.DataFrame, *, model_seed: int = MODEL_SEED) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit the accepted full-train model and apply it to test exactly once."""
    if "ID" not in train or "SUBCLASS" not in train or "ID" not in test:
        raise ValueError("train/test schema must include ID and train must include SUBCLASS")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if list(test.columns) != ["ID", *genes]:
        raise ValueError("test gene columns must exactly match train gene order")
    if int(train[genes].isna().sum().sum()) != 0:
        raise ValueError("training gene matrix violates the no-NaN contract")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    train_frame, test_frame = train.loc[:, genes], test.loc[:, genes]

    print("[submission] fit train-only structured mutation features", flush=True)
    x_train, x_test, names, structured_audit = build_design_matrices(
        train_frame, test_frame, labels, genes, seed=model_seed,
    )
    print("[submission] fit H0 multinomial Logistic Regression", flush=True)
    non_eb_probability, non_eb_warnings = _fit_lr_probability(x_train, labels, x_test, classes, model_seed)

    print("[submission] fit train-only gene×event-type Empirical-Bayes evidence", flush=True)
    eb_train, eb_test = empirical_bayes_features(
        train_frame, test_frame, labels, classes, genes, seed=model_seed,
    )
    x_train_eb = sparse.hstack([x_train, sparse.csr_matrix(eb_train)], format="csr")
    x_test_eb = sparse.hstack([x_test, sparse.csr_matrix(eb_test)], format="csr")
    print("[submission] fit train-only exact-event Empirical-Bayes evidence", flush=True)
    exact_train, exact_test, exact_vocabulary_size = exact_eb_features(
        train_frame, test_frame, labels, classes, genes, seed=model_seed,
    )
    x_train_exact = sparse.hstack([x_train_eb, sparse.csr_matrix(exact_train)], format="csr")
    x_test_exact = sparse.hstack([x_test_eb, sparse.csr_matrix(exact_test)], format="csr")
    exact_probability, exact_warnings = _fit_lr_probability(x_train_exact, labels, x_test_exact, classes, model_seed)
    selective_lr_probability, use_non_eb = selective_probability(non_eb_probability, exact_probability)

    print("[submission] fit full-train LGBM and automatic two-pair specialist", flush=True)
    lgbm = LGBMClassifier(
        objective="multiclass", boosting_type="gbdt", num_class=len(classes),
        n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10,
        min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced",
        random_state=model_seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    lgbm.fit(x_train, labels)
    lgbm_probability = _aligned_probability(lgbm, lgbm.predict_proba(x_test), classes)
    specialist_probability, specialist_pairs = _hard_specialist(
        x_train, labels, x_test, lgbm_probability, classes, names, model_seed,
    )
    final_probability = fixed_branch_replacement(selective_lr_probability, specialist_probability)
    audit = {
        "run_id": RUN_ID,
        "model_seed": model_seed,
        "lr_weight": SELECTIVE_LR_WEIGHT,
        "specialist_weight": H0_SPECIALIST_WEIGHT,
        "selective_margin": SELECTIVE_MARGIN,
        "threshold_retuned": False,
        "test_role": "transform_and_predict_only",
        "test_read_for_fit_statistics_selection_or_scaling": False,
        "raw_train_test_concat": False,
        "vocabulary_source": "full_train_only",
        "specialist_pair_source": "full_train_only_automatic_discovery",
        "fixed_cancer_gene_exact_mutation_rules": False,
        "nan_as_mutation_count": int(structured_audit["nan_as_mutation_count"]),
        "leakage_check": bool(not structured_audit["raw_train_test_concat"]),
        "structured_feature_count": int(x_train.shape[1]),
        "gene_type_eb_feature_count": int(eb_train.shape[1]),
        "exact_eb_feature_count": int(exact_train.shape[1]),
        "exact_vocabulary_size": int(exact_vocabulary_size),
        "exact_event_vocabulary_source": "full_train_only",
        "exact_event_support_cutoff": None,
        "final_feature_count": int(x_train_exact.shape[1]),
        "specialist_pairs": [list(pair) for pair in specialist_pairs],
        "selective_non_eb_test_rows": int(use_non_eb.sum()),
        "convergence_warning_count": int(non_eb_warnings + exact_warnings),
        "eda_train_rows": int(len(train)),
        "eda_gene_count": int(len(genes)),
        "eda_class_count": int(len(classes)),
        "eda_train_nan_cell_count": int(train[genes].isna().sum().sum()),
    }
    del lgbm, x_train_eb, x_test_eb, x_train_exact, x_test_exact, exact_train, exact_test
    gc.collect()
    if audit["nan_as_mutation_count"] != 0 or not audit["leakage_check"]:
        raise AssertionError("submission safety contract failed")
    return final_probability, classes, audit


def run(output_name: str = "submission_h0_exact_event_eb_seed42.csv") -> Path:
    started = perf_counter()
    raw = data_directory()
    print("[submission] PHASE 1/2 — read train only and fit all train-only states", flush=True)
    train = pd.read_csv(raw / "train.csv")
    state, audit = fit_submission_from_train(train, model_seed=MODEL_SEED)
    print("[submission] PHASE 1/2 complete — now read test for transform/prediction only", flush=True)
    test = pd.read_csv(raw / "test.csv")
    sample = pd.read_csv(raw / "sample_submission.csv")
    probability, prediction_audit = predict_submission_from_fitted_state(state, test)
    audit.update(prediction_audit)
    submission = make_submission_frame(sample, test, probability, state.classes)
    destination = submission_directory() / output_name
    submission.to_csv(destination, index=False)
    audit.update({"output_file": str(destination), "row_count": int(len(submission)), "runtime_seconds": perf_counter() - started, "environment": environment_metadata()})
    audit_path = destination.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    reloaded = pd.read_csv(destination)
    if not reloaded.equals(submission):
        raise AssertionError("submission round-trip validation failed")
    print(json.dumps({"submission": str(destination), "audit": str(audit_path), "rows": len(submission), "leakage_check": audit["leakage_check"], "nan_as_mutation_count": audit["nan_as_mutation_count"]}, ensure_ascii=False), flush=True)
    return destination


def run_seed_bagged(
    output_name: str = "submission_h0_exact_event_eb_seed42_777_2024_bagged.csv",
    seeds: tuple[int, ...] = (42, 777, 2024),
) -> Path:
    """Fit each predeclared seed on full train and equally average test probabilities."""
    if tuple(seeds) != (42, 777, 2024):
        raise ValueError("the validated seed-bagging contract is exactly (42, 777, 2024)")
    started = perf_counter(); raw = data_directory()
    print("[submission] PHASE 1/2 — read train only; fit all three seed models before test is read", flush=True)
    train = pd.read_csv(raw / "train.csv")
    states, audits, classes = [], [], None
    for seed in seeds:
        print(f"[submission] train-only full-train seed {seed}", flush=True)
        state, audit = fit_submission_from_train(train, model_seed=seed)
        current_classes = state.classes
        if classes is None:
            classes = current_classes
        elif not np.array_equal(classes, current_classes):
            raise AssertionError("class order differs between seed fits")
        states.append(state); audits.append(audit)
    print("[submission] PHASE 1/2 complete — read test only for frozen-state transformation and prediction", flush=True)
    test = pd.read_csv(raw / "test.csv"); sample = pd.read_csv(raw / "sample_submission.csv")
    probability_rows = []
    for state, audit in zip(states, audits):
        probability, prediction_audit = predict_submission_from_fitted_state(state, test)
        audit.update(prediction_audit)
        probability_rows.append(probability)
    averaged = average_seed_probabilities(probability_rows)
    submission = make_submission_frame(sample, test, averaged, classes)
    destination = submission_directory() / output_name; submission.to_csv(destination, index=False)
    audit = {
        "run_id": RUN_ID + "-seed-bagging",
        "seeds": list(seeds),
        "seed_weights": [1.0 / len(seeds)] * len(seeds),
        "weight_tuned": False,
        "test_role": "transform_and_predict_only",
        "raw_train_test_concat": False,
        "leakage_check": bool(all(item["leakage_check"] for item in audits)),
        "nan_as_mutation_count": int(max(item["nan_as_mutation_count"] for item in audits)),
        "per_seed_audits": audits,
        "output_file": str(destination),
        "row_count": int(len(submission)),
        "runtime_seconds": perf_counter() - started,
        "environment": environment_metadata(),
    }
    audit_path = destination.with_suffix(".audit.json"); audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not pd.read_csv(destination).equals(submission) or not audit["leakage_check"] or audit["nan_as_mutation_count"] != 0:
        raise AssertionError("seed-bagged submission safety validation failed")
    print(json.dumps({"submission": str(destination), "audit": str(audit_path), "rows": len(submission), "leakage_check": audit["leakage_check"], "nan_as_mutation_count": audit["nan_as_mutation_count"]}, ensure_ascii=False), flush=True)
    return destination


def smoke_test() -> dict:
    """Train-only portability and parser-contract check; does not read test.csv."""
    train_path = data_directory() / "train.csv"
    train = pd.read_csv(train_path, nrows=64)
    if "ID" not in train or "SUBCLASS" not in train:
        raise ValueError("train.csv must include ID and SUBCLASS")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if not genes:
        raise ValueError("no gene columns found in train.csv")
    if normalise_cell(np.nan) or normalise_cell("") or normalise_cell("WT"):
        raise AssertionError("NaN, blank, and WT must produce zero mutation events")
    if normalise_cell("R132H R132H") != ("R132H",):
        raise AssertionError("duplicate events must be deduplicated within a cell")
    parsed = transform_rows(train.loc[:, genes], genes, fit_vocabulary(train.loc[:, genes], genes))
    audit = {
        "smoke": True,
        "test_read": False,
        "train_rows_checked": int(len(train)),
        "gene_count": int(len(genes)),
        "class_count": int(train.SUBCLASS.nunique()),
        "train_nan_cell_count": int(train[genes].isna().sum().sum()),
        "parsed_mutation_nnz": int(parsed.mutation.nnz),
        "nan_as_mutation_count": 0,
        "leakage_check": True,
        "raw_train_test_concat": False,
    }
    if audit["train_nan_cell_count"] != 0:
        raise AssertionError("the accepted training contract requires no gene-cell NaN")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit



# =============================================================================
# 8. Command-line entry point: equal 42/777/2024 probability bagging
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the self-contained Exact-event EB 3-seed Dacon submission.")
    parser.add_argument("--root", type=Path, help="project root; use this when the script is copied outside the repository")
    parser.add_argument("--data-dir", type=Path, help="directory containing train.csv/test.csv/sample_submission.csv; defaults to /data when available")
    parser.add_argument("--output-dir", type=Path, help="directory for the CSV and audit JSON; defaults to the project submission folder")
    parser.add_argument("--output-name", default="submission_h0_exact_event_eb_seed42_777_2024_bagged.csv")
    parser.add_argument("--smoke", action="store_true", help="run the train-only parser/path smoke test without training or reading test.csv")
    args = parser.parse_args()
    global ROOT_OVERRIDE, DATA_DIR_OVERRIDE, OUTPUT_DIR_OVERRIDE
    ROOT_OVERRIDE = args.root
    DATA_DIR_OVERRIDE = args.data_dir
    OUTPUT_DIR_OVERRIDE = args.output_dir
    if args.smoke:
        smoke_test()
        return
    run_seed_bagged(output_name=args.output_name, seeds=(42, 777, 2024))


if __name__ == "__main__":
    main()
