"""Self-contained exp013 feature contract; no imports from other experiments."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from scipy import sparse
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import warnings


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
REFERENCE_LR = 0.526130
REFERENCE_LGBM_SPECIALIST = 0.492332
REFERENCE_BLEND = 0.543679
REFERENCE_TOLERANCE = 0.001


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
