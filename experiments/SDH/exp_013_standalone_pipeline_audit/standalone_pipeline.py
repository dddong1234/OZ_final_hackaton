"""Standalone, train-only implementation of the SDH exp012 champion pipeline.

This module intentionally imports no code from another experiment.  Every
learned preprocessing decision is fitted from the supplied training rows and
the resulting state is only applied to validation/test rows.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


WT = "WT"
EVENT_TYPES = (
    "MISSENSE",
    "SYNONYMOUS",
    "NONSENSE",
    "FRAMESHIFT",
    "SPLICE",
    "INFRAME_INDEL",
    "OTHER",
)
TRUNCATING = frozenset({"NONSENSE", "FRAMESHIFT", "SPLICE"})
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
SPLICE_RE = re.compile(r"SPLICE|IVS|[+-]\d+")
INDEL_RE = re.compile(r"DEL|INS|DUP")

RECURRENT_MIN_COUNT = 5
ENRICHMENT_MIN_SUPPORT = 10
ENRICHMENT_SHRINKAGE = 10.0
ENRICHMENT_ALPHA = 1.0
ENRICHMENT_WEIGHT_CLIP = 4.0


def normalise_cell(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == WT:
        return ()
    tokens = re.sub(r"[;,|]+", " ", text).split()
    return tuple(
        dict.fromkeys(token.removeprefix("P.") for token in tokens if token)
    )


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


@dataclass(frozen=True)
class Vocabulary:
    exact_events: tuple[str, ...]
    gene_types: tuple[str, ...]


@dataclass
class ParsedFrame:
    genes: list[str]
    mutation: sparse.csr_matrix
    truncation: sparse.csr_matrix
    exact: sparse.csr_matrix
    gene_type: sparse.csr_matrix
    burden: np.ndarray
    variant: np.ndarray
    amino_pair: np.ndarray
    topology: np.ndarray

    @property
    def n_rows(self) -> int:
        return self.mutation.shape[0]


def _event_records(frame: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    records: list[tuple[int, int, str, str]] = []
    for gene_index, gene in enumerate(genes):
        for row_index, value in enumerate(frame[gene].array):
            for event in normalise_cell(value):
                records.append(
                    (row_index, gene_index, event, classify_event(event))
                )
    events = pd.DataFrame(
        records,
        columns=["row", "gene_index", "event", "event_type"],
    )
    if events.empty:
        return events
    events = events.drop_duplicates(
        ["row", "gene_index", "event"]
    ).reset_index(drop=True)
    events["gene"] = events["gene_index"].map(dict(enumerate(genes)))
    events["exact_name"] = events["gene"] + "__" + events["event"]
    events["gene_type_name"] = events["gene"] + "__" + events["event_type"]
    return events


def fit_vocabulary(frame: pd.DataFrame, genes: list[str]) -> Vocabulary:
    """Fit both column spaces from training rows only."""

    events = _event_records(frame, genes)
    if events.empty:
        return Vocabulary((), ())
    return Vocabulary(
        tuple(sorted(events["exact_name"].unique())),
        tuple(sorted(events["gene_type_name"].unique())),
    )


def _binary_matrix(
    events: pd.DataFrame,
    name_column: str,
    vocabulary: tuple[str, ...],
    n_rows: int,
) -> sparse.csr_matrix:
    if events.empty or not vocabulary:
        return sparse.csr_matrix((n_rows, len(vocabulary)), dtype=np.float32)
    lookup = {name: index for index, name in enumerate(vocabulary)}
    columns = events[name_column].map(lookup)
    known = columns.notna().to_numpy()
    matrix = sparse.coo_matrix(
        (
            np.ones(int(known.sum()), dtype=np.float32),
            (
                events["row"].to_numpy(dtype=np.int32)[known],
                columns.to_numpy()[known].astype(np.int32),
            ),
        ),
        shape=(n_rows, len(vocabulary)),
    ).tocsr()
    matrix.data[:] = 1.0
    return matrix


def transform_rows(
    frame: pd.DataFrame,
    genes: list[str],
    vocabulary: Vocabulary,
) -> ParsedFrame:
    """Apply a fixed training vocabulary; all other blocks are row-local."""

    n_rows = len(frame)
    n_genes = len(genes)
    events = _event_records(frame, genes)

    if events.empty:
        mutation = sparse.csr_matrix((n_rows, n_genes), dtype=np.float32)
        truncation = sparse.csr_matrix((n_rows, n_genes), dtype=np.float32)
    else:
        mutated = events[["row", "gene_index"]].drop_duplicates()
        mutation = sparse.coo_matrix(
            (
                np.ones(len(mutated), dtype=np.float32),
                (mutated["row"], mutated["gene_index"]),
            ),
            shape=(n_rows, n_genes),
        ).tocsr()
        truncating = events.loc[
            events["event_type"].isin(TRUNCATING), ["row", "gene_index"]
        ].drop_duplicates()
        truncation = sparse.coo_matrix(
            (
                np.ones(len(truncating), dtype=np.float32),
                (truncating["row"], truncating["gene_index"]),
            ),
            shape=(n_rows, n_genes),
        ).tocsr()
    mutation.data[:] = 1.0
    truncation.data[:] = 1.0

    exact = _binary_matrix(
        events, "exact_name", vocabulary.exact_events, n_rows
    )
    gene_type = _binary_matrix(
        events, "gene_type_name", vocabulary.gene_types, n_rows
    )

    burden = np.zeros((n_rows, 3), dtype=np.float32)
    burden[:, 0] = np.asarray(mutation.sum(axis=1)).ravel()
    variant = np.zeros((n_rows, len(EVENT_TYPES)), dtype=np.float32)
    amino_pair = np.zeros((n_rows, 380), dtype=np.float32)
    topology = np.zeros((n_rows, 8), dtype=np.float32)

    if not events.empty:
        burden[:, 1] = (
            events.groupby("row").size().reindex(range(n_rows), fill_value=0)
        )
        by_gene = events.groupby(["row", "gene_index"]).size()
        burden[:, 2] = (
            by_gene.gt(1)
            .groupby(level=0)
            .sum()
            .reindex(range(n_rows), fill_value=0)
        )
        for column, event_type in enumerate(EVENT_TYPES):
            variant[:, column] = (
                events["event_type"]
                .eq(event_type)
                .groupby(events["row"])
                .sum()
                .reindex(range(n_rows), fill_value=0)
            )

        parsed = events["event"].str.extract(SUB_RE)
        substitutions = pd.DataFrame(
            {
                "row": events["row"],
                "ref": parsed[0],
                "alt": parsed[2],
            }
        ).dropna()
        pair_lookup = {
            (ref, alt): index
            for index, (ref, alt) in enumerate(
                (a, b) for a in AA for b in AA if a != b
            )
        }
        for row, ref, alt in substitutions.itertuples(index=False):
            index = pair_lookup.get((ref, alt))
            if index is not None:
                amino_pair[int(row), index] += 1

        gene_counts = events.groupby(["row", "gene_index"]).agg(
            event_count=("event", "size"),
            type_count=("event_type", "nunique"),
        )
        masks = (
            gene_counts.event_count.eq(1),
            gene_counts.event_count.eq(2),
            gene_counts.event_count.ge(3),
            gene_counts.type_count.ge(2),
        )
        for column, mask in enumerate(masks):
            topology[:, column] = (
                mask.groupby(level=0).sum().reindex(range(n_rows), fill_value=0)
            )
        topology[:, 4] = (
            gene_counts.event_count.groupby(level=0)
            .max()
            .reindex(range(n_rows), fill_value=0)
        )
        type_counts = pd.crosstab(events["row"], events["event_type"]).reindex(
            index=range(n_rows), columns=EVENT_TYPES, fill_value=0
        )
        proportions = type_counts.div(
            type_counts.sum(axis=1).replace(0, np.nan), axis=0
        ).fillna(0)
        topology[:, 5] = type_counts.gt(0).sum(axis=1)
        safe = proportions.where(proportions.gt(0), 1)
        topology[:, 6] = -(safe * np.log(safe)).sum(axis=1)
        topology[:, 7] = proportions.max(axis=1)

    return ParsedFrame(
        genes,
        mutation,
        truncation,
        exact,
        gene_type,
        burden,
        variant,
        amino_pair,
        topology,
    )


def fit_transform_pair(
    train_frame: pd.DataFrame,
    apply_frame: pd.DataFrame,
    genes: list[str],
) -> tuple[ParsedFrame, ParsedFrame, Vocabulary]:
    vocabulary = fit_vocabulary(train_frame, genes)
    return (
        transform_rows(train_frame, genes, vocabulary),
        transform_rows(apply_frame, genes, vocabulary),
        vocabulary,
    )


def _nonconstant(matrix: sparse.csr_matrix) -> np.ndarray:
    minimum = np.asarray(matrix.min(axis=0).toarray()).ravel()
    maximum = np.asarray(matrix.max(axis=0).toarray()).ravel()
    return minimum != maximum


def build_b04_matrices(
    train: ParsedFrame,
    apply: ParsedFrame,
    vocabulary: Vocabulary,
    labels: np.ndarray,
    *,
    use_fixed_contrast: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str]]:
    """Fit B04 column selection on train and apply it to another split."""

    genes = train.genes
    active = np.flatnonzero(np.asarray(train.mutation.getnnz(axis=0)).ravel())
    truncating = np.flatnonzero(
        np.asarray(train.truncation.getnnz(axis=0)).ravel()
    )
    exact_counts = np.asarray(train.exact.getnnz(axis=0)).ravel()
    exact_type = np.asarray(
        [classify_event(name.split("__", 1)[1]) for name in vocabulary.exact_events]
    )
    recurrent = np.flatnonzero(
        (exact_counts >= RECURRENT_MIN_COUNT) & (exact_type == "MISSENSE")
    )

    train_parts: list[sparse.csr_matrix] = [train.mutation[:, active]]
    apply_parts: list[sparse.csr_matrix] = [apply.mutation[:, active]]
    names = [f"G__{genes[index]}" for index in active]

    train_parts.extend(
        [sparse.csr_matrix(np.log1p(train.burden)), sparse.csr_matrix(np.log1p(train.variant))]
    )
    apply_parts.extend(
        [sparse.csr_matrix(np.log1p(apply.burden)), sparse.csr_matrix(np.log1p(apply.variant))]
    )
    names.extend(
        ["B__mutated_gene_count", "B__event_count", "B__multi_event_gene_count"]
        + [f"V__{name.lower()}_event_count" for name in EVENT_TYPES]
    )

    train_parts.append(train.truncation[:, truncating])
    apply_parts.append(apply.truncation[:, truncating])
    names.extend(f"T__{genes[index]}" for index in truncating)
    train_parts.append(sparse.csr_matrix(train.truncation.sum(axis=1)))
    apply_parts.append(sparse.csr_matrix(apply.truncation.sum(axis=1)))
    names.append("T__truncating_gene_count")

    train_parts.append(train.exact[:, recurrent])
    apply_parts.append(apply.exact[:, recurrent])
    names.extend(f"R__{vocabulary.exact_events[index]}" for index in recurrent)
    train_parts.append(sparse.csr_matrix(train.exact[:, recurrent].sum(axis=1)))
    apply_parts.append(sparse.csr_matrix(apply.exact[:, recurrent].sum(axis=1)))
    names.append("R__recurrent_missense_event_count")

    train_parts.append(sparse.csr_matrix(np.log1p(train.amino_pair)))
    apply_parts.append(sparse.csr_matrix(np.log1p(apply.amino_pair)))
    names.extend(f"A_pair__{index}" for index in range(380))
    train_parts.append(sparse.csr_matrix(train.topology))
    apply_parts.append(sparse.csr_matrix(apply.topology))
    names.extend(f"S__{index}" for index in range(8))

    if use_fixed_contrast:
        raise ValueError(
            "Fixed cancer-pair contrast was removed from the shared baseline. "
            "Use train-discovered, outer-fold-local alternatives instead."
        )

    train_matrix = sparse.hstack(train_parts, format="csr")
    apply_matrix = sparse.hstack(apply_parts, format="csr")
    keep = _nonconstant(train_matrix)
    return (
        train_matrix[:, keep],
        apply_matrix[:, keep],
        [name for name, included in zip(names, keep) if included],
    )


def _fit_enrichment_weights(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(matrix.getnnz(axis=0)).ravel()
    selected = np.flatnonzero(
        (support >= ENRICHMENT_MIN_SUPPORT) & (support < matrix.shape[0])
    )
    if not len(selected):
        return selected, np.zeros((len(classes), 0), dtype=np.float32)
    matrix = matrix[:, selected]
    support = support[selected].astype(np.float64)
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, class_name in enumerate(classes):
        positive_mask = labels == class_name
        positive_size = int(positive_mask.sum())
        negative_size = len(labels) - positive_size
        positive = np.asarray(matrix[positive_mask].getnnz(axis=0)).ravel()
        negative = support - positive
        positive_log_odds = np.log(
            (positive + ENRICHMENT_ALPHA)
            / (positive_size - positive + ENRICHMENT_ALPHA)
        )
        negative_log_odds = np.log(
            (negative + ENRICHMENT_ALPHA)
            / (negative_size - negative + ENRICHMENT_ALPHA)
        )
        weights[class_index] = positive_log_odds - negative_log_odds
    weights *= support[None, :] / (support[None, :] + ENRICHMENT_SHRINKAGE)
    return selected, np.clip(
        weights, -ENRICHMENT_WEIGHT_CLIP, ENRICHMENT_WEIGHT_CLIP
    ).astype(np.float32)


def _apply_enrichment(
    matrix: sparse.csr_matrix,
    selected: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if not len(selected):
        return np.zeros((matrix.shape[0], weights.shape[0]), dtype=np.float32)
    rows = matrix[:, selected]
    scores = np.asarray(rows @ weights.T, dtype=np.float32)
    denominator = np.sqrt(
        np.maximum(np.asarray(rows.getnnz(axis=1)).ravel(), 1)
    ).astype(np.float32)
    return scores / denominator[:, None]


def cross_fitted_enrichment(
    train: ParsedFrame,
    apply: ParsedFrame,
    labels: np.ndarray,
    *,
    seed: int,
    inner_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes = sorted(np.unique(labels).tolist())
    train_scores = np.zeros((train.n_rows, len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )
    for fit_index, holdout_index in splitter.split(
        np.zeros(train.n_rows), labels
    ):
        selected, weights = _fit_enrichment_weights(
            train.gene_type[fit_index], labels[fit_index], classes
        )
        train_scores[holdout_index] = _apply_enrichment(
            train.gene_type[holdout_index], selected, weights
        )
    selected, weights = _fit_enrichment_weights(
        train.gene_type, labels, classes
    )
    apply_scores = _apply_enrichment(apply.gene_type, selected, weights)

    keep = train_scores.min(axis=0) != train_scores.max(axis=0)
    train_scores = train_scores[:, keep]
    apply_scores = apply_scores[:, keep]
    names = [
        f"E__gene_type__{name}"
        for name, included in zip(classes, keep)
        if included
    ]
    mean = train_scores.mean(axis=0)
    deviation = train_scores.std(axis=0)
    deviation[deviation < 1e-6] = 1.0
    return (
        ((train_scores - mean) / deviation).astype(np.float32),
        ((apply_scores - mean) / deviation).astype(np.float32),
        names,
    )


def build_design_matrices(
    train_frame: pd.DataFrame,
    apply_frame: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    genes: list[str],
    *,
    seed: int = 42,
    use_fixed_contrast: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str], dict]:
    label_array = np.asarray(labels)
    train, apply, vocabulary = fit_transform_pair(train_frame, apply_frame, genes)
    x_train, x_apply, names = build_b04_matrices(
        train,
        apply,
        vocabulary,
        label_array,
        use_fixed_contrast=use_fixed_contrast,
    )
    base_feature_count = len(names)
    train_scores, apply_scores, enrichment_names = cross_fitted_enrichment(
        train, apply, label_array, seed=seed
    )
    x_train = sparse.hstack(
        [x_train, sparse.csr_matrix(train_scores)], format="csr"
    )
    x_apply = sparse.hstack(
        [x_apply, sparse.csr_matrix(apply_scores)], format="csr"
    )
    names.extend(enrichment_names)
    audit = {
        "raw_train_test_concat": False,
        "vocabulary_source": "fit_frame_only",
        "exact_vocabulary_size": len(vocabulary.exact_events),
        "gene_type_vocabulary_size": len(vocabulary.gene_types),
        "base_feature_count": base_feature_count,
        "enrichment_feature_count": len(enrichment_names),
        "total_feature_count": len(names),
        "fixed_contrast_enabled": False,
        "fixed_exact_event_enabled": False,
    }
    return x_train, x_apply, names, audit


def make_model(seed: int = 42) -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        C=0.07,
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )


def evaluate_seed(
    train: pd.DataFrame,
    genes: list[str],
    *,
    seed: int = 42,
    use_fixed_contrast: bool = False,
) -> dict:
    labels = train["SUBCLASS"].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prediction = np.empty(len(train), dtype=object)
    fold_scores: list[float] = []
    feature_counts: list[int] = []
    warning_count = 0
    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(train, labels), start=1
    ):
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        fit_labels = labels.iloc[fit_index].reset_index(drop=True)
        x_fit, x_valid, names, _ = build_design_matrices(
            fit_frame,
            valid_frame,
            fit_labels,
            genes,
            seed=seed * 100 + fold,
            use_fixed_contrast=use_fixed_contrast,
        )
        model = make_model(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_fit, fit_labels)
        warning_count += sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        fold_prediction = model.predict(x_valid)
        prediction[valid_index] = fold_prediction
        fold_scores.append(
            f1_score(
                labels.iloc[valid_index],
                fold_prediction,
                average="macro",
                zero_division=0,
            )
        )
        feature_counts.append(len(names))
    return {
        "seed": seed,
        "oof_f1_macro": f1_score(
            labels, prediction, average="macro", zero_division=0
        ),
        "oof_accuracy": accuracy_score(labels, prediction),
        "fold_f1_mean": float(np.mean(fold_scores)),
        "fold_f1_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "feature_count_mean": float(np.mean(feature_counts)),
        "convergence_warning_count": warning_count,
        "prediction": prediction,
    }
