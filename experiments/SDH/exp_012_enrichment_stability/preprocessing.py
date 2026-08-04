"""Configurable, fold-safe gene-by-event-type enrichment for SDH exp012."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold


B04_ID = "H-AS-LR-exact-confusion-pairs-Apair-log1p"
ALPHA = 1.0
WEIGHT_CLIP = 4.0
DECLINING_CLASSES = ("LIHC", "DLBC", "HNSC", "LUSC")


def _load_b04_module() -> ModuleType:
    module_name = "_sdh_exp012_gs_b04_source"
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
class EnrichmentCase:
    name: str
    include_enrichment: bool = True
    min_support: int = 10
    shrinkage: float = 20.0
    excluded_scores: tuple[str, ...] = ()
    description: str = ""


@dataclass
class FeatureContext:
    cache: object
    gene_type_matrix: sparse.csr_matrix
    gene_type_names: list[str]


def make_cases() -> dict[str, EnrichmentCase]:
    cases = (
        EnrichmentCase(
            "case_00_b04",
            include_enrichment=False,
            description="GS B04 reproduction",
        ),
        EnrichmentCase(
            "case_01_winner_support10_shrink20",
            description="exp011 winner: all 26 scores",
        ),
        EnrichmentCase(
            "case_02_support5",
            min_support=5,
            description="retain rarer gene-type tokens",
        ),
        EnrichmentCase(
            "case_03_support20",
            min_support=20,
            description="retain only more stable gene-type tokens",
        ),
        EnrichmentCase(
            "case_04_shrink10",
            shrinkage=10.0,
            description="weaker support shrinkage",
        ),
        EnrichmentCase(
            "case_05_shrink50",
            shrinkage=50.0,
            description="stronger support shrinkage",
        ),
        EnrichmentCase(
            "case_06_exclude_LIHC_score",
            excluded_scores=("LIHC",),
            description="remove LIHC enrichment coordinate",
        ),
        EnrichmentCase(
            "case_07_exclude_DLBC_score",
            excluded_scores=("DLBC",),
            description="remove DLBC enrichment coordinate",
        ),
        EnrichmentCase(
            "case_08_exclude_HNSC_score",
            excluded_scores=("HNSC",),
            description="remove HNSC enrichment coordinate",
        ),
        EnrichmentCase(
            "case_09_exclude_LUSC_score",
            excluded_scores=("LUSC",),
            description="remove LUSC enrichment coordinate",
        ),
        EnrichmentCase(
            "case_10_exclude_declining4_scores",
            excluded_scores=DECLINING_CLASSES,
            description="remove LIHC/DLBC/HNSC/LUSC coordinates",
        ),
    )
    return {case.name: case for case in cases}


def make_context(
    frame: pd.DataFrame,
    genes: list[str],
    *,
    show_progress: bool = True,
) -> FeatureContext:
    """Build only deterministic row-local representations."""

    cache = B04.RowCache.build(frame, genes, show_progress=show_progress)
    events = cache.events
    if events.empty:
        matrix = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
        return FeatureContext(cache, matrix, [])

    observed = events[["row", "gene_idx", "event_type"]].drop_duplicates().copy()
    gene_lookup = dict(enumerate(genes))
    observed["gene_type"] = (
        observed["gene_idx"].map(gene_lookup).astype(str)
        + "__"
        + observed["event_type"].astype(str)
    )
    names = sorted(observed["gene_type"].unique())
    name_lookup = {name: index for index, name in enumerate(names)}
    matrix = sparse.coo_matrix(
        (
            np.ones(len(observed), dtype=np.float32),
            (
                observed["row"].to_numpy(dtype=np.int32),
                observed["gene_type"].map(name_lookup).to_numpy(dtype=np.int32),
            ),
        ),
        shape=(len(frame), len(names)),
    ).tocsr()
    matrix.data[:] = 1.0
    return FeatureContext(cache, matrix, names)


def _b04_builder(context: FeatureContext):
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
        candidate.b_count_binning,
    )


def _fit_weights(
    token_matrix: sparse.csr_matrix,
    fit_index: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    case: EnrichmentCase,
) -> tuple[np.ndarray, np.ndarray]:
    fit_matrix = token_matrix[fit_index]
    support = np.asarray(fit_matrix.getnnz(axis=0)).ravel()
    selected = np.flatnonzero(
        (support >= case.min_support) & (support < len(fit_index))
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
            (positive + ALPHA) / (positive_size - positive + ALPHA)
        )
        negative_log_odds = np.log(
            (negative + ALPHA) / (negative_size - negative + ALPHA)
        )
        weights[class_index] = positive_log_odds - negative_log_odds

    weights *= support[None, :] / (support[None, :] + case.shrinkage)
    weights = np.clip(weights, -WEIGHT_CLIP, WEIGHT_CLIP)
    return selected, weights.astype(np.float32)


def _apply_scores(
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


def _cross_fitted_scores(
    context: FeatureContext,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    labels: pd.Series,
    case: EnrichmentCase,
    *,
    inner_seed: int,
    inner_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
        selected, weights = _fit_weights(
            context.gene_type_matrix,
            fit_global,
            label_array,
            classes,
            case,
        )
        train_scores[inner_holdout] = _apply_scores(
            context.gene_type_matrix,
            holdout_global,
            selected,
            weights,
        )

    selected, weights = _fit_weights(
        context.gene_type_matrix,
        train_index,
        label_array,
        classes,
        case,
    )
    valid_scores = _apply_scores(
        context.gene_type_matrix,
        valid_index,
        selected,
        weights,
    )
    keep = np.asarray(
        [class_name not in case.excluded_scores for class_name in classes]
    )
    train_scores = train_scores[:, keep]
    valid_scores = valid_scores[:, keep]
    names = [
        f"E__gene_type__{class_name}"
        for class_name, selected_class in zip(classes, keep)
        if selected_class
    ]
    return train_scores, valid_scores, names


def _standardize_and_filter(
    train_scores: np.ndarray,
    valid_scores: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keep = train_scores.min(axis=0) != train_scores.max(axis=0)
    train_scores = train_scores[:, keep]
    valid_scores = valid_scores[:, keep]
    names = [name for name, included in zip(names, keep) if included]
    if not train_scores.shape[1]:
        return train_scores, valid_scores, names
    mean = train_scores.mean(axis=0)
    standard_deviation = train_scores.std(axis=0)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return (
        ((train_scores - mean) / standard_deviation).astype(np.float32),
        ((valid_scores - mean) / standard_deviation).astype(np.float32),
        names,
    )


def build_case_matrices(
    context: FeatureContext,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    labels: pd.Series,
    case: EnrichmentCase,
    *,
    inner_seed: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str], dict[str, int]]:
    builder = _b04_builder(context)
    train_matrix, valid_matrix, names = builder.build(
        train_index, valid_index, labels
    )
    base_feature_count = len(names)

    if case.include_enrichment:
        train_scores, valid_scores, score_names = _cross_fitted_scores(
            context,
            train_index,
            valid_index,
            labels,
            case,
            inner_seed=inner_seed,
        )
        train_scores, valid_scores, score_names = _standardize_and_filter(
            train_scores, valid_scores, score_names
        )
        if train_scores.shape[1]:
            train_matrix = sparse.hstack(
                [train_matrix, sparse.csr_matrix(train_scores)], format="csr"
            )
            valid_matrix = sparse.hstack(
                [valid_matrix, sparse.csr_matrix(valid_scores)], format="csr"
            )
            names.extend(score_names)

    metadata = {
        "base_feature_count": base_feature_count,
        "extra_feature_count": len(names) - base_feature_count,
        "total_feature_count": len(names),
    }
    return train_matrix, valid_matrix, names, metadata
