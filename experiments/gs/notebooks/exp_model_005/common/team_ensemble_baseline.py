"""Train-only primitives for the fixed team 3-way OOF baseline.

This module intentionally never accepts ``test`` data.  Every token space is
created from the caller-provided fit rows and validation rows are projected
onto that fixed space.
"""
from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier


WT = "WT"
EVENT_TYPES = ("MISSENSE", "SYNONYMOUS", "NONSENSE", "FRAMESHIFT", "SPLICE", "INFRAME_INDEL", "OTHER")
TRUNCATING = frozenset({"NONSENSE", "FRAMESHIFT", "SPLICE"})
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
SPLICE_RE = re.compile(r"SPLICE|IVS|[+-]\d+")
INDEL_RE = re.compile(r"DEL|INS|DUP")


@dataclass(frozen=True)
class EventCache:
    """Minimal event store used by the train-only vocabulary contract."""

    events: pd.DataFrame
    row_count: int
    genes: tuple[str, ...] = ()
    mutation_matrix: sparse.csr_matrix | None = None
    truncation_matrix: sparse.csr_matrix | None = None
    burden: np.ndarray | None = None
    variant: np.ndarray | None = None
    amino: np.ndarray | None = None
    topology: np.ndarray | None = None

    @classmethod
    def from_rows(cls, rows: list[list[tuple[str, str]]]) -> "EventCache":
        records = [
            (row, gene, event_type)
            for row, row_events in enumerate(rows)
            for gene, event_type in row_events
        ]
        return cls(pd.DataFrame(records, columns=["row", "gene", "event_type"]), len(rows))


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


def parse_train_frame(frame: pd.DataFrame, genes: list[str], show_progress: bool = False) -> EventCache:
    """Parse one training frame without any test-dependent vocabulary or statistics."""
    records: list[tuple[int, int, str, str, str]] = []
    mut_rows: list[int] = []
    mut_cols: list[int] = []
    trunc_rows: list[int] = []
    trunc_cols: list[int] = []
    for gene_idx, gene in enumerate(genes):
        for row_idx, value in enumerate(frame[gene].array):
            events = normalise_cell(value)
            if not events:
                continue
            mut_rows.append(row_idx)
            mut_cols.append(gene_idx)
            for event in events:
                event_type = classify_event(event)
                records.append((row_idx, gene_idx, gene, event, event_type))
                if event_type in TRUNCATING:
                    trunc_rows.append(row_idx)
                    trunc_cols.append(gene_idx)

    n_rows, n_genes = len(frame), len(genes)
    mutation = sparse.coo_matrix(
        (np.ones(len(mut_rows), dtype=np.float32), (mut_rows, mut_cols)),
        shape=(n_rows, n_genes),
    ).tocsr()
    mutation.data[:] = 1.0
    truncation = sparse.coo_matrix(
        (np.ones(len(trunc_rows), dtype=np.float32), (trunc_rows, trunc_cols)),
        shape=(n_rows, n_genes),
    ).tocsr()
    truncation.data[:] = 1.0
    events_frame = pd.DataFrame(records, columns=["row", "gene_idx", "gene", "event", "event_type"])
    if not events_frame.empty:
        events_frame = events_frame.drop_duplicates(["row", "gene_idx", "event"]).reset_index(drop=True)

    burden = np.zeros((n_rows, 3), dtype=np.float32)
    burden[:, 0] = np.asarray(mutation.sum(axis=1)).ravel()
    variant = np.zeros((n_rows, len(EVENT_TYPES)), dtype=np.float32)
    amino = np.zeros((n_rows, 20 + 20 + 380 + 6), dtype=np.float32)
    topology = np.zeros((n_rows, 8), dtype=np.float32)
    if not events_frame.empty:
        burden[:, 1] = events_frame.groupby("row").size().reindex(range(n_rows), fill_value=0).to_numpy()
        by_gene = events_frame.groupby(["row", "gene_idx"]).size()
        burden[:, 2] = by_gene.gt(1).groupby(level=0).sum().reindex(range(n_rows), fill_value=0).to_numpy()
        for column, event_type in enumerate(EVENT_TYPES):
            variant[:, column] = events_frame.event_type.eq(event_type).groupby(events_frame.row).sum().reindex(range(n_rows), fill_value=0).to_numpy()
        parsed = events_frame.event.str.extract(SUB_RE)
        events_frame["ref"] = parsed[0]
        events_frame["pos"] = pd.to_numeric(parsed[1], errors="coerce")
        events_frame["alt"] = parsed[2]
        aa_index = {letter: index for index, letter in enumerate(AA)}
        pair_index = {(ref, alt): index for index, (ref, alt) in enumerate((a, b) for a in AA for b in AA if a != b)}
        for row, ref, alt, pos in events_frame.dropna(subset=["ref", "alt", "pos"])[["row", "ref", "alt", "pos"]].itertuples(index=False):
            if ref in aa_index:
                amino[row, aa_index[ref]] += 1
            if alt in aa_index:
                amino[row, 20 + aa_index[alt]] += 1
            if (ref, alt) in pair_index:
                amino[row, 40 + pair_index[(ref, alt)]] += 1
            for bin_index, (low, high) in enumerate(((1, 50), (51, 100), (101, 250), (251, 500), (501, 1000), (1001, np.inf))):
                if low <= pos <= high:
                    amino[row, 420 + bin_index] += 1
                    break
        gene_counts = events_frame.groupby(["row", "gene_idx"]).agg(event_count=("event", "size"), type_count=("event_type", "nunique"))
        for column, mask in enumerate((gene_counts.event_count.eq(1), gene_counts.event_count.eq(2), gene_counts.event_count.ge(3), gene_counts.type_count.ge(2))):
            topology[:, column] = mask.groupby(level=0).sum().reindex(range(n_rows), fill_value=0).to_numpy()
        topology[:, 4] = gene_counts.event_count.groupby(level=0).max().reindex(range(n_rows), fill_value=0).to_numpy()
        type_counts = pd.crosstab(events_frame.row, events_frame.event_type).reindex(index=range(n_rows), columns=EVENT_TYPES, fill_value=0)
        proportions = type_counts.div(type_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        topology[:, 5] = type_counts.gt(0).sum(axis=1).to_numpy()
        topology[:, 6] = -(proportions.where(proportions.gt(0), 1) * np.log(proportions.where(proportions.gt(0), 1))).sum(axis=1).to_numpy()
        topology[:, 7] = proportions.max(axis=1).to_numpy()
    return EventCache(events_frame, n_rows, tuple(genes), mutation, truncation, burden, variant, amino, topology)


def build_train_only_gene_type_vocabulary(cache: EventCache, fit_rows: np.ndarray) -> tuple[str, ...]:
    """Return sorted gene×type tokens observed in ``fit_rows`` only."""
    fit = cache.events[cache.events["row"].isin(np.asarray(fit_rows, dtype=np.int64))]
    if fit.empty:
        return ()
    tokens = fit["gene"].astype(str) + "__" + fit["event_type"].astype(str)
    return tuple(sorted(tokens.drop_duplicates().tolist()))


def project_gene_type_matrix(
    cache: EventCache, rows: np.ndarray, vocabulary: tuple[str, ...]
) -> sparse.csr_matrix:
    """Project rows onto a pre-existing train-only vocabulary.

    A token observed only in validation has no output column and is ignored.
    """
    row_ids = np.asarray(rows, dtype=np.int64)
    local_lookup = {int(row): index for index, row in enumerate(row_ids)}
    token_lookup = {token: index for index, token in enumerate(vocabulary)}
    subset = cache.events[cache.events["row"].isin(row_ids)]
    if subset.empty or not vocabulary:
        return sparse.csr_matrix((len(row_ids), len(vocabulary)), dtype=np.float32)
    tokens = subset["gene"].astype(str) + "__" + subset["event_type"].astype(str)
    cols = tokens.map(token_lookup)
    known = cols.notna().to_numpy()
    if not known.any():
        return sparse.csr_matrix((len(row_ids), len(vocabulary)), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (
            np.ones(int(known.sum()), dtype=np.float32),
            (
                subset.loc[known, "row"].map(local_lookup).to_numpy(dtype=np.int64),
                cols[known].to_numpy(dtype=np.int64),
            ),
        ),
        shape=(len(row_ids), len(vocabulary)),
    ).tocsr()
    matrix.data[:] = 1.0
    return matrix


def _project_exact_event_matrix(cache: EventCache, rows: np.ndarray, vocabulary: tuple[str, ...]) -> sparse.csr_matrix:
    """Project exact ``GENE__EVENT`` values onto fixed, fit-row-only columns."""
    row_ids = np.asarray(rows, dtype=np.int64)
    row_lookup = {int(row): index for index, row in enumerate(row_ids)}
    token_lookup = {token: index for index, token in enumerate(vocabulary)}
    subset = cache.events[cache.events["row"].isin(row_ids)]
    if subset.empty or not vocabulary:
        return sparse.csr_matrix((len(row_ids), len(vocabulary)), dtype=np.float32)
    tokens = subset["gene"].astype(str) + "__" + subset["event"].astype(str)
    columns = tokens.map(token_lookup)
    known = columns.notna().to_numpy()
    if not known.any():
        return sparse.csr_matrix((len(row_ids), len(vocabulary)), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (np.ones(int(known.sum()), dtype=np.float32),
         (subset.loc[known, "row"].map(row_lookup).to_numpy(dtype=np.int64), columns[known].to_numpy(dtype=np.int64))),
        shape=(len(row_ids), len(vocabulary)),
    ).tocsr()
    matrix.data[:] = 1.0
    return matrix


def _nonconstant_columns(matrix: sparse.csr_matrix) -> np.ndarray:
    minimum = np.asarray(matrix.min(axis=0).toarray()).ravel()
    maximum = np.asarray(matrix.max(axis=0).toarray()).ravel()
    return minimum != maximum


FINAL_EXACT_HOTSPOTS = (("BRAF", "V600E"), ("IDH1", "R132H"), ("PIK3CA", "H1047R"), ("PIK3CA", "E545K"))
ENSEMBLE_WEIGHTS = {"multinomial": 0.55, "ovr": 0.30, "lightgbm": 0.15}
ENRICHMENT_MIN_SUPPORT = 10
ENRICHMENT_ALPHA = 1.0
ENRICHMENT_SHRINKAGE = 10.0
ENRICHMENT_WEIGHT_CLIP = 4.0


def _event_vocabulary(cache: EventCache, fit_rows: np.ndarray, *, missense_only: bool = False, min_support: int = 1) -> tuple[str, ...]:
    subset = cache.events[cache.events["row"].isin(np.asarray(fit_rows, dtype=np.int64))]
    if missense_only:
        subset = subset[subset.event_type.eq("MISSENSE")]
    if subset.empty:
        return ()
    names = subset.gene.astype(str) + "__" + subset.event.astype(str)
    counts = names.value_counts()
    return tuple(sorted(counts[counts.ge(min_support)].index.tolist()))


def _discover_confusion_pairs(cache: EventCache, fit_rows: np.ndarray, labels: np.ndarray, seed: int) -> tuple[tuple[str, str, int], ...]:
    """Derive confusion pairs solely from inner OOF predictions on ``fit_rows``."""
    if cache.mutation_matrix is None:
        raise ValueError("parsed mutation matrix is required")
    y_fit = labels[fit_rows]
    prediction = np.empty(len(fit_rows), dtype=object)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for local_train, local_valid in splitter.split(np.zeros(len(y_fit)), y_fit):
        model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=300, class_weight="balanced", random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(cache.mutation_matrix[fit_rows[local_train]], y_fit[local_train])
        prediction[local_valid] = model.predict(cache.mutation_matrix[fit_rows[local_valid]])
    score: list[tuple[float, str, str]] = []
    classes = np.unique(y_fit)
    for left_idx, left in enumerate(classes):
        for right in classes[left_idx + 1:]:
            swapped = int(((y_fit == left) & (prediction == right)).sum() + ((y_fit == right) & (prediction == left)).sum())
            support = max(int((y_fit == left).sum() + (y_fit == right).sum()), 1)
            score.append((swapped / support, str(left), str(right)))
    score.sort(key=lambda value: (-value[0], value[1], value[2]))
    return tuple((left, right, 5) for _, left, right in score[:8])


def _structured_matrix(cache: EventCache, fit_rows: np.ndarray, out_rows: np.ndarray, labels: np.ndarray, seed: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str]]:
    """Build the fixed team structured blocks using outer-fit selection only."""
    if cache.mutation_matrix is None or cache.truncation_matrix is None:
        raise ValueError("parsed matrices are required")
    all_rows = np.arange(cache.row_count, dtype=np.int64)
    active_genes = np.flatnonzero(np.asarray(cache.mutation_matrix[fit_rows].getnnz(axis=0)).ravel())
    parts: list[sparse.csr_matrix] = [cache.mutation_matrix[:, active_genes]]
    names = [f"G__{cache.genes[index]}" for index in active_genes]
    parts.extend([sparse.csr_matrix(np.log1p(cache.burden)), sparse.csr_matrix(np.log1p(cache.variant))])
    names.extend(["B__mutated_gene_count", "B__event_count", "B__multi_event_gene_count"])
    names.extend([f"V__{event_type.lower()}_event_count" for event_type in EVENT_TYPES])
    active_trunc = np.flatnonzero(np.asarray(cache.truncation_matrix[fit_rows].getnnz(axis=0)).ravel())
    parts.append(cache.truncation_matrix[:, active_trunc])
    names.extend(f"T__{cache.genes[index]}" for index in active_trunc)
    parts.append(sparse.csr_matrix(np.asarray(cache.truncation_matrix.sum(axis=1))))
    names.append("T__truncating_gene_count")
    recurrent = _event_vocabulary(cache, fit_rows, missense_only=True, min_support=5)
    recurrent_matrix = _project_exact_event_matrix(cache, all_rows, recurrent)
    parts.extend([recurrent_matrix, sparse.csr_matrix(np.asarray(recurrent_matrix.sum(axis=1)))])
    names.extend(f"R__{token}" for token in recurrent)
    names.append("R__recurrent_missense_event_count")
    parts.append(sparse.csr_matrix(np.log1p(cache.amino[:, 40:420])))
    names.extend(f"A_pair__{index}" for index in range(380))
    parts.append(sparse.csr_matrix(cache.topology))
    names.extend(f"S__{index}" for index in range(cache.topology.shape[1]))
    hotspots = tuple(f"{gene}__{event}" for gene, event in FINAL_EXACT_HOTSPOTS)
    parts.append(_project_exact_event_matrix(cache, all_rows, hotspots))
    names.extend(f"D__exact_{token}" for token in hotspots)
    for left, right, top_k in _discover_confusion_pairs(cache, fit_rows, labels, seed):
        y_fit = labels[fit_rows]
        left_mask, right_mask = y_fit == left, y_fit == right
        if not left_mask.any() or not right_mask.any():
            continue
        left_counts = np.asarray(cache.mutation_matrix[fit_rows][left_mask].getnnz(axis=0)).ravel()
        right_counts = np.asarray(cache.mutation_matrix[fit_rows][right_mask].getnnz(axis=0)).ravel()
        support = left_counts + right_counts
        contrast = left_counts / left_mask.sum() - right_counts / right_mask.sum()
        selected = sorted(np.flatnonzero(support >= 10), key=lambda index: (-abs(contrast[index]), -support[index], cache.genes[index]))[:top_k]
        if not selected:
            continue
        parts.append(sparse.csr_matrix(cache.mutation_matrix[:, selected].sum(axis=1)))
        signs = sparse.csr_matrix(np.sign(contrast[selected]).astype(np.float32)).T
        parts.append(cache.mutation_matrix[:, selected].dot(signs))
        names.extend([f"C__{left}_vs_{right}_count", f"C__{left}_vs_{right}_contrast"])
    matrix = sparse.hstack(parts, format="csr")
    keep = _nonconstant_columns(matrix[fit_rows])
    return matrix[fit_rows][:, keep], matrix[out_rows][:, keep], [name for name, enabled in zip(names, keep) if enabled]


def _fit_enrichment_weights(token_matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(token_matrix.getnnz(axis=0)).ravel()
    selected = np.flatnonzero((support >= ENRICHMENT_MIN_SUPPORT) & (support < len(labels)))
    if not len(selected):
        return selected, np.zeros((len(classes), 0), dtype=np.float32)
    matrix = token_matrix[:, selected]
    selected_support = support[selected].astype(np.float64)
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, class_name in enumerate(classes):
        positive_mask = labels == class_name
        positive = np.asarray(matrix[positive_mask].getnnz(axis=0)).ravel().astype(np.float64)
        negative = selected_support - positive
        positive_size = max(int(positive_mask.sum()), 1)
        negative_size = max(len(labels) - positive_size, 1)
        weights[class_index] = (
            np.log((positive + ENRICHMENT_ALPHA) / (positive_size - positive + ENRICHMENT_ALPHA))
            - np.log((negative + ENRICHMENT_ALPHA) / (negative_size - negative + ENRICHMENT_ALPHA))
        )
    weights *= selected_support[None, :] / (selected_support[None, :] + ENRICHMENT_SHRINKAGE)
    return selected, np.clip(weights, -ENRICHMENT_WEIGHT_CLIP, ENRICHMENT_WEIGHT_CLIP).astype(np.float32)


def _apply_enrichment(token_matrix: sparse.csr_matrix, selected: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if not len(selected):
        return np.zeros((token_matrix.shape[0], weights.shape[0]), dtype=np.float32)
    matrix = token_matrix[:, selected]
    scores = np.asarray(matrix @ weights.T, dtype=np.float32)
    denominator = np.sqrt(np.maximum(np.asarray(matrix.getnnz(axis=1)).ravel(), 1)).astype(np.float32)
    return scores / denominator[:, None]


def _cross_fitted_team_enrichment(cache: EventCache, fit_rows: np.ndarray, out_rows: np.ndarray, labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create train score only from inner-fit rows, then apply outer-fit weights to out rows."""
    train_score = np.zeros((len(fit_rows), len(classes)), dtype=np.float32)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for local_fit, local_holdout in inner.split(np.zeros(len(fit_rows)), labels[fit_rows]):
        source_rows, holdout_rows = fit_rows[local_fit], fit_rows[local_holdout]
        vocabulary = build_train_only_gene_type_vocabulary(cache, source_rows)
        selected, weights = _fit_enrichment_weights(project_gene_type_matrix(cache, source_rows, vocabulary), labels[source_rows], classes)
        train_score[local_holdout] = _apply_enrichment(project_gene_type_matrix(cache, holdout_rows, vocabulary), selected, weights)
    vocabulary = build_train_only_gene_type_vocabulary(cache, fit_rows)
    selected, weights = _fit_enrichment_weights(project_gene_type_matrix(cache, fit_rows, vocabulary), labels[fit_rows], classes)
    out_score = _apply_enrichment(project_gene_type_matrix(cache, out_rows, vocabulary), selected, weights)
    mean = train_score.mean(axis=0)
    std = np.maximum(train_score.std(axis=0), 1e-6)
    return ((train_score - mean) / std).astype(np.float32), ((out_score - mean) / std).astype(np.float32)


class BaselineOOF(NamedTuple):
    probabilities: np.ndarray
    classes: np.ndarray
    fold_metrics: pd.DataFrame
    summary: dict[str, object]


def _make_models(seed: int):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("LightGBM is required for the fixed team baseline") from error
    parameters = dict(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed)
    return {
        "multinomial": LogisticRegression(**parameters),
        "ovr": OneVsRestClassifier(LogisticRegression(**parameters), n_jobs=1),
        "lightgbm": LGBMClassifier(objective="multiclass", n_estimators=100, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1),
    }


def run_team_baseline_oof(train: pd.DataFrame, genes: list[str], labels: np.ndarray, seed: int = 42) -> BaselineOOF:
    """Reproduce fixed team OOF using train rows only; no test argument exists."""
    start = time.time()
    cache = parse_train_frame(train[genes], genes, show_progress=False)
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    oof = np.zeros((len(train), len(classes)), dtype=np.float32)
    rows: list[dict[str, object]] = []
    warning_total = 0
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (fit_rows, valid_rows) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        x_fit, x_valid, feature_names = _structured_matrix(cache, fit_rows, valid_rows, labels, seed + fold)
        enrichment_fit, enrichment_valid = _cross_fitted_team_enrichment(cache, fit_rows, valid_rows, labels, classes, seed * 100 + fold)
        x_fit = sparse.hstack([x_fit, sparse.csr_matrix(enrichment_fit)], format="csr")
        x_valid = sparse.hstack([x_valid, sparse.csr_matrix(enrichment_valid)], format="csr")
        blend = np.zeros((len(valid_rows), len(classes)), dtype=np.float64)
        for name, model in _make_models(seed).items():
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_fit, labels[fit_rows])
            warning_total += sum(issubclass(item.category, ConvergenceWarning) for item in captured)
            raw_probability = model.predict_proba(x_valid)
            aligned = np.zeros_like(blend)
            aligned[:, np.searchsorted(classes, np.asarray(model.classes_, dtype=object))] = raw_probability
            blend += ENSEMBLE_WEIGHTS[name] * aligned
        oof[valid_rows] = blend.astype(np.float32)
        rows.append({"fold": fold, "macro_f1": float(f1_score(labels[valid_rows], classes[blend.argmax(1)], average="macro")), "feature_count": int(x_fit.shape[1]), "base_feature_count": len(feature_names)})
    score = float(f1_score(labels, classes[oof.argmax(1)], average="macro"))
    return BaselineOOF(oof, classes, pd.DataFrame(rows), {
        "oof_macro_f1": score,
        "reference_macro_f1": 0.54202,
        "baseline_reproduction_match": bool(abs(score - 0.54202) <= 0.003),
        "convergence_warning_count": int(warning_total),
        "test_read": False,
        "runtime_seconds": float(time.time() - start),
    })
