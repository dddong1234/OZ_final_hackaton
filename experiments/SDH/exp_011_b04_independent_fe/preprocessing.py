"""B04-fixed, leakage-safe feature blocks for SDH exp011.

The GS champion implementation is loaded directly from its merged source file.
Only the extra blocks below are owned by exp011.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold


B04_ID = "H-AS-LR-exact-confusion-pairs-Apair-log1p"
ENRICHMENT_MIN_SUPPORT = 10
ENRICHMENT_ALPHA = 1.0
ENRICHMENT_SHRINKAGE = 20.0
ENRICHMENT_WEIGHT_CLIP = 4.0


def _load_b04_module() -> ModuleType:
    module_name = "_sdh_exp011_gs_b04_source"
    if module_name in sys.modules:
        return sys.modules[module_name]
    experiments_dir = Path(__file__).resolve().parents[2]
    source = (
        experiments_dir
        / "gs"
        / "notebooks"
        / "eda_pre_002"
        / "common"
        / "exp-gs-002-memory-safe.py"
    )
    if not source.exists():
        raise FileNotFoundError(f"B04 원본 모듈을 찾지 못했습니다: {source}")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"B04 모듈 spec 생성 실패: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


B04 = _load_b04_module()
B04_CANDIDATE = B04.CANDIDATES[B04_ID]


@dataclass(frozen=True)
class FeatureCase:
    name: str
    blocks: tuple[str, ...]
    description: str


@dataclass
class FeatureContext:
    cache: object
    gene_type_matrix: sparse.csr_matrix
    gene_type_names: list[str]


def make_context(
    frame: pd.DataFrame,
    genes: list[str],
    *,
    show_progress: bool = True,
) -> FeatureContext:
    """Parse deterministic row-local representations once for every case."""

    cache = B04.RowCache.build(frame, genes, show_progress=show_progress)
    events = cache.events
    if events.empty:
        matrix = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
        return FeatureContext(cache, matrix, [])

    observed = events[["row", "gene_idx", "event_type"]].drop_duplicates().copy()
    observed["gene_type"] = (
        observed["gene_idx"].map(dict(enumerate(genes))).astype(str)
        + "__"
        + observed["event_type"].astype(str)
    )
    names = sorted(observed["gene_type"].unique())
    lookup = {name: index for index, name in enumerate(names)}
    matrix = sparse.coo_matrix(
        (
            np.ones(len(observed), dtype=np.float32),
            (
                observed["row"].to_numpy(dtype=np.int32),
                observed["gene_type"].map(lookup).to_numpy(dtype=np.int32),
            ),
        ),
        shape=(len(frame), len(names)),
    ).tocsr()
    matrix.data[:] = 1.0
    return FeatureContext(cache, matrix, names)


def make_independent_cases() -> dict[str, FeatureCase]:
    """Every non-baseline case adds exactly one block to B04."""

    cases = (
        FeatureCase("case_00_b04", (), "GS B04 champion reproduction"),
        FeatureCase(
            "case_01_b04_plus_burden_bins",
            ("burden_bins",),
            "fixed one-hot bins for the three burden counts",
        ),
        FeatureCase(
            "case_02_b04_plus_row_profile",
            ("row_profile",),
            "row-local event-type proportions and burden ratios",
        ),
        FeatureCase(
            "case_03_b04_plus_gene_enrichment",
            ("gene_enrichment",),
            "26 cross-fitted class scores from mutated-gene presence",
        ),
        FeatureCase(
            "case_04_b04_plus_gene_type_enrichment",
            ("gene_type_enrichment",),
            "26 cross-fitted class scores from gene by event type",
        ),
        FeatureCase(
            "case_05_b04_plus_exact_event_enrichment",
            ("exact_event_enrichment",),
            "26 cross-fitted class scores from exact gene-event tokens",
        ),
    )
    return {case.name: case for case in cases}


def combine_cases(name: str, cases: Sequence[FeatureCase]) -> FeatureCase:
    """Combine only independently screened blocks while preserving order."""

    blocks = tuple(dict.fromkeys(block for case in cases for block in case.blocks))
    descriptions = " + ".join(case.name for case in cases)
    return FeatureCase(name, blocks, descriptions)


def _b04_builder(context: FeatureContext, *, burden_bins: bool):
    candidate = B04_CANDIDATE
    return B04.FoldMatrixBuilder(
        context.cache,
        candidate.backbone,
        candidate.exact_events,
        candidate.gene_pairs,
        candidate.gene_groups,
        candidate.hotspot_top_k,
        candidate.contrast_pairs,
        candidate.amino_mode,
        candidate.log1p_counts,
        burden_bins,
    )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros_like(numerator, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=output,
        where=denominator != 0,
    )
    return output


def _row_profile(context: FeatureContext) -> tuple[np.ndarray, list[str]]:
    cache = context.cache
    gene_count = cache.burden[:, 0].astype(np.float32)
    event_count = cache.burden[:, 1].astype(np.float32)
    multi_gene_count = cache.burden[:, 2].astype(np.float32)
    truncating_gene_count = np.asarray(
        cache.truncation_matrix.sum(axis=1)
    ).ravel().astype(np.float32)

    columns = [
        _safe_divide(event_count, gene_count),
        _safe_divide(multi_gene_count, gene_count),
        _safe_divide(truncating_gene_count, gene_count),
    ]
    names = [
        "P__events_per_mutated_gene",
        "P__multi_event_gene_share",
        "P__truncating_gene_share",
    ]
    for column, event_type in enumerate(B04.EVENT_TYPES):
        columns.append(_safe_divide(cache.variant[:, column], event_count))
        names.append(f"P__{event_type.lower()}_event_share")
    return np.column_stack(columns).astype(np.float32), names


def _fit_enrichment_weights(
    token_matrix: sparse.csr_matrix,
    fit_index: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    fit_matrix = token_matrix[fit_index]
    support = np.asarray(fit_matrix.getnnz(axis=0)).ravel()
    selected = np.flatnonzero(
        (support >= ENRICHMENT_MIN_SUPPORT) & (support < len(fit_index))
    )
    if not len(selected):
        return selected, np.zeros((len(classes), 0), dtype=np.float32)

    fit_matrix = fit_matrix[:, selected]
    support = support[selected].astype(np.float64)
    fit_labels = labels[fit_index]
    weights = np.zeros((len(classes), len(selected)), dtype=np.float64)
    for class_index, class_name in enumerate(classes):
        positive_mask = fit_labels == class_name
        positive_size = int(positive_mask.sum())
        negative_size = len(fit_index) - positive_size
        positive = np.asarray(
            fit_matrix[positive_mask].getnnz(axis=0)
        ).ravel().astype(np.float64)
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
    weights = np.clip(
        weights,
        -ENRICHMENT_WEIGHT_CLIP,
        ENRICHMENT_WEIGHT_CLIP,
    )
    return selected, weights.astype(np.float32)


def _apply_enrichment(
    token_matrix: sparse.csr_matrix,
    row_index: np.ndarray,
    selected: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if not len(selected):
        return np.zeros((len(row_index), weights.shape[0]), dtype=np.float32)
    rows = token_matrix[row_index][:, selected]
    scores = np.asarray(rows @ weights.T, dtype=np.float32)
    denominator = np.sqrt(
        np.maximum(np.asarray(rows.getnnz(axis=1)).ravel(), 1)
    ).astype(np.float32)
    return scores / denominator[:, None]


def _cross_fitted_enrichment(
    token_matrix: sparse.csr_matrix,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    labels: pd.Series,
    *,
    inner_seed: int,
    prefix: str,
    inner_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create OOF scores for outer-train and train-fitted scores for valid."""

    label_array = labels.to_numpy()
    classes = sorted(np.unique(label_array[train_index]).tolist())
    train_scores = np.zeros((len(train_index), len(classes)), dtype=np.float32)
    outer_labels = label_array[train_index]
    splitter = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=inner_seed,
    )
    for inner_fit, inner_holdout in splitter.split(
        np.zeros(len(train_index)), outer_labels
    ):
        fit_global = train_index[inner_fit]
        holdout_global = train_index[inner_holdout]
        selected, weights = _fit_enrichment_weights(
            token_matrix, fit_global, label_array, classes
        )
        train_scores[inner_holdout] = _apply_enrichment(
            token_matrix, holdout_global, selected, weights
        )

    selected, weights = _fit_enrichment_weights(
        token_matrix, train_index, label_array, classes
    )
    valid_scores = _apply_enrichment(
        token_matrix, valid_index, selected, weights
    )
    names = [f"E__{prefix}__{class_name}" for class_name in classes]
    return train_scores, valid_scores, names


def _standardize_and_filter(
    train_values: np.ndarray,
    valid_values: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    minimum = train_values.min(axis=0)
    maximum = train_values.max(axis=0)
    keep = minimum != maximum
    train_values = train_values[:, keep]
    valid_values = valid_values[:, keep]
    kept_names = [name for name, selected in zip(names, keep) if selected]
    if not train_values.shape[1]:
        return train_values, valid_values, kept_names
    mean = train_values.mean(axis=0)
    standard_deviation = train_values.std(axis=0)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    train_values = (train_values - mean) / standard_deviation
    valid_values = (valid_values - mean) / standard_deviation
    return (
        train_values.astype(np.float32),
        valid_values.astype(np.float32),
        kept_names,
    )


def build_case_matrices(
    context: FeatureContext,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    labels: pd.Series,
    case: FeatureCase,
    *,
    inner_seed: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str], dict[str, int]]:
    """Build exact B04 plus the requested independent extra blocks."""

    builder = _b04_builder(
        context,
        burden_bins="burden_bins" in case.blocks,
    )
    train_matrix, valid_matrix, names = builder.build(
        train_index, valid_index, labels
    )
    base_feature_count = len(names)

    for block in case.blocks:
        if block == "burden_bins":
            continue
        if block == "row_profile":
            all_values, extra_names = _row_profile(context)
            train_values = all_values[train_index]
            valid_values = all_values[valid_index]
        elif block == "gene_enrichment":
            train_values, valid_values, extra_names = _cross_fitted_enrichment(
                context.cache.mutation_matrix,
                train_index,
                valid_index,
                labels,
                inner_seed=inner_seed,
                prefix="gene",
            )
        elif block == "gene_type_enrichment":
            train_values, valid_values, extra_names = _cross_fitted_enrichment(
                context.gene_type_matrix,
                train_index,
                valid_index,
                labels,
                inner_seed=inner_seed,
                prefix="gene_type",
            )
        elif block == "exact_event_enrichment":
            train_values, valid_values, extra_names = _cross_fitted_enrichment(
                context.cache.event_matrix,
                train_index,
                valid_index,
                labels,
                inner_seed=inner_seed,
                prefix="exact_event",
            )
        else:
            raise ValueError(f"알 수 없는 exp011 피처 블록: {block}")

        train_values, valid_values, extra_names = _standardize_and_filter(
            train_values, valid_values, extra_names
        )
        if train_values.shape[1]:
            train_matrix = sparse.hstack(
                [train_matrix, sparse.csr_matrix(train_values)], format="csr"
            )
            valid_matrix = sparse.hstack(
                [valid_matrix, sparse.csr_matrix(valid_values)], format="csr"
            )
            names.extend(extra_names)

    metadata = {
        "base_feature_count": base_feature_count,
        "extra_feature_count": len(names) - base_feature_count,
        "total_feature_count": len(names),
    }
    return train_matrix, valid_matrix, names, metadata
