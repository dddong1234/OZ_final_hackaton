"""Leakage-safe hierarchical class-enrichment experiments for SDH exp018.

The B04 backbone is built by the audited exp013 standalone implementation.
This module adds only fold-train-fitted supervised score blocks:

* gene x amino-acid substitution pair (for example IDH1__R>H)
* gene x mutation type x position bin
* hierarchical residuals that shrink a fine token toward gene x mutation type

Raw train and validation/test frames are never concatenated.  Every vocabulary,
support threshold, class log-odds, shrinkage decision, and standardisation
statistic is fitted on the current outer-fold training split only.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier


def _load_audited_baseline():
    module_name = "_sdh_exp018_audited_exp013"
    if module_name in sys.modules:
        return sys.modules[module_name]
    source = (
        Path(__file__).resolve().parents[1]
        / "exp_013_standalone_pipeline_audit"
        / "standalone_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"exp013 standalone을 불러오지 못했습니다: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_audited_baseline()

CLASSES_EXPECTED = 26
OUTER_SPLITS = 5
INNER_SPLITS = 5
LR_C = 0.07
LR_MAX_ITER = 2000
ALPHA = 1.0
WEIGHT_CLIP = 4.0
DEFAULT_SHRINKAGE = 10.0
THREE_WAY_WEIGHTS = {"multinomial": 0.55, "ovr": 0.30, "lgbm": 0.15}
SAFE_BASELINE_CASE = "e01_gene_type_baseline"


@dataclass(frozen=True)
class EnrichmentCase:
    name: str
    gene_type: bool = True
    amino_mode: str = "none"       # none / independent / residual
    position_mode: str = "none"    # none / independent / residual
    position_scheme: str = "p50"   # p50 / coarse6
    min_support: int = 10
    backoff: float = 10.0
    description: str = ""


@dataclass(frozen=True)
class TokenPair:
    train: sparse.csr_matrix
    apply: sparse.csr_matrix
    names: tuple[str, ...]
    parent_columns: np.ndarray


@dataclass
class PreparedFold:
    fold: int
    fit_index: np.ndarray
    valid_index: np.ndarray
    fit_labels: np.ndarray
    x_fit_base: sparse.csr_matrix
    x_valid_base: sparse.csr_matrix
    extras: dict[str, tuple[np.ndarray, np.ndarray, list[str]]]
    audits: dict[str, dict]


@dataclass
class OOFResult:
    name: str
    model: str
    seed: int
    probability: np.ndarray
    prediction: np.ndarray
    classes: np.ndarray
    fold_scores: list[float]
    feature_counts: list[int]
    convergence_warning_count: int

    @property
    def macro_f1(self) -> float:
        return float(self._macro_f1)

    @property
    def accuracy(self) -> float:
        return float(self._accuracy)

    def attach_metrics(self, labels: np.ndarray) -> "OOFResult":
        self._macro_f1 = f1_score(
            labels, self.prediction, average="macro", zero_division=0
        )
        self._accuracy = accuracy_score(labels, self.prediction)
        return self


def case_catalog() -> dict[str, EnrichmentCase]:
    """Core ablations plus a small, pre-declared stability grid."""

    cases = (
        EnrichmentCase(
            "e00_b04_no_enrichment",
            gene_type=False,
            description="safe B04 backbone only",
        ),
        EnrichmentCase(
            SAFE_BASELINE_CASE,
            description="exp013 gene×mutation-type enrichment baseline",
        ),
        EnrichmentCase(
            "e02_amino_only",
            gene_type=False,
            amino_mode="independent",
            description="gene×A-pair enrichment replaces gene-type",
        ),
        EnrichmentCase(
            "e03_gene_type_plus_amino_independent",
            amino_mode="independent",
            description="two independent 26-class score blocks",
        ),
        EnrichmentCase(
            "e04_gene_type_plus_amino_residual",
            amino_mode="residual",
            description="A-pair deviation from parent gene-type",
        ),
        EnrichmentCase(
            "e05_position50_only",
            gene_type=False,
            position_mode="independent",
            description="gene×type×50-aa-bin enrichment only",
        ),
        EnrichmentCase(
            "e06_gene_type_plus_position50_independent",
            position_mode="independent",
            description="independent 50-aa position score",
        ),
        EnrichmentCase(
            "e07_gene_type_plus_position50_residual",
            position_mode="residual",
            description="50-aa position deviation from gene-type",
        ),
        EnrichmentCase(
            "e08_gene_type_plus_position6_residual",
            position_mode="residual",
            position_scheme="coarse6",
            description="coarse position deviation from gene-type",
        ),
        EnrichmentCase(
            "e09_all_independent",
            amino_mode="independent",
            position_mode="independent",
            description="gene-type + A-pair + position, all independent",
        ),
        EnrichmentCase(
            "e10_all_hierarchical_residual",
            amino_mode="residual",
            position_mode="residual",
            description="gene-type + two fine residual blocks",
        ),
        EnrichmentCase(
            "e11_amino_residual_position_independent",
            amino_mode="residual",
            position_mode="independent",
            description="mixed parameterisation control A",
        ),
        EnrichmentCase(
            "e12_amino_independent_position_residual",
            amino_mode="independent",
            position_mode="residual",
            description="mixed parameterisation control B",
        ),
        EnrichmentCase(
            "e13_residual_support5",
            amino_mode="residual",
            position_mode="residual",
            min_support=5,
            description="retain rarer fine tokens",
        ),
        EnrichmentCase(
            "e14_residual_support20",
            amino_mode="residual",
            position_mode="residual",
            min_support=20,
            description="retain only stable fine tokens",
        ),
        EnrichmentCase(
            "e15_residual_backoff5",
            amino_mode="residual",
            position_mode="residual",
            backoff=5.0,
            description="weaker parent backoff",
        ),
        EnrichmentCase(
            "e16_residual_backoff20",
            amino_mode="residual",
            position_mode="residual",
            backoff=20.0,
            description="stronger parent backoff",
        ),
    )
    return {case.name: case for case in cases}


def _position_label(position: int, scheme: str) -> str:
    if scheme == "p50":
        return f"P50_{(position - 1) // 50:03d}"
    if scheme != "coarse6":
        raise ValueError(f"지원하지 않는 position scheme: {scheme}")
    boundaries = (50, 100, 250, 500, 1000)
    return f"P6_{sum(position > boundary for boundary in boundaries)}"


def _fine_events(
    frame: pd.DataFrame,
    genes: list[str],
    kind: str,
    position_scheme: str = "p50",
) -> pd.DataFrame:
    events = BASE._event_records(frame, genes)
    columns = ["row", "token", "parent"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    parsed = events["event"].str.extract(BASE.SUB_RE)
    work = events[["row", "gene", "event_type"]].copy()
    work["ref"] = parsed[0]
    work["position"] = pd.to_numeric(parsed[1], errors="coerce")
    work["alt"] = parsed[2]
    work = work.dropna(subset=["ref", "position", "alt"])
    work["position"] = work["position"].astype(int)
    work = work.loc[work["position"] > 0].copy()
    if kind == "amino":
        work = work.loc[
            work["ref"].isin(BASE.AA)
            & work["alt"].isin(BASE.AA)
            & work["ref"].ne(work["alt"])
        ].copy()
        work["token"] = (
            work["gene"] + "__" + work["ref"] + ">" + work["alt"]
        )
        work["parent"] = work["gene"] + "__MISSENSE"
    elif kind == "position":
        work["position_bin"] = work["position"].map(
            lambda value: _position_label(int(value), position_scheme)
        )
        work["token"] = (
            work["gene"]
            + "__"
            + work["event_type"]
            + "__"
            + work["position_bin"]
        )
        work["parent"] = work["gene"] + "__" + work["event_type"]
    else:
        raise ValueError(f"지원하지 않는 token kind: {kind}")
    return work[columns].drop_duplicates().reset_index(drop=True)


def _matrix_from_events(
    events: pd.DataFrame,
    vocabulary: tuple[str, ...],
    n_rows: int,
) -> sparse.csr_matrix:
    if events.empty or not vocabulary:
        return sparse.csr_matrix((n_rows, len(vocabulary)), dtype=np.float32)
    lookup = {name: index for index, name in enumerate(vocabulary)}
    columns = events["token"].map(lookup)
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


def fit_transform_fine_tokens(
    train_frame: pd.DataFrame,
    apply_frame: pd.DataFrame,
    genes: list[str],
    parent_names: tuple[str, ...],
    *,
    kind: str,
    position_scheme: str = "p50",
) -> TokenPair:
    """Fit a fine-token vocabulary on train and only project apply rows."""

    train_events = _fine_events(train_frame, genes, kind, position_scheme)
    apply_events = _fine_events(apply_frame, genes, kind, position_scheme)
    names = tuple(sorted(train_events["token"].unique()))
    parent_by_token = (
        train_events.drop_duplicates("token").set_index("token")["parent"].to_dict()
    )
    parent_lookup = {name: index for index, name in enumerate(parent_names)}
    missing = sorted(
        token for token in names if parent_by_token[token] not in parent_lookup
    )
    if missing:
        raise AssertionError(f"fine token parent가 train vocabulary에 없습니다: {missing[:3]}")
    parent_columns = np.asarray(
        [parent_lookup[parent_by_token[token]] for token in names], dtype=np.int32
    )
    return TokenPair(
        _matrix_from_events(train_events, names, len(train_frame)),
        _matrix_from_events(apply_events, names, len(apply_frame)),
        names,
        parent_columns,
    )


def _raw_log_odds(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    classes: list[str],
    columns: np.ndarray,
) -> np.ndarray:
    if not len(columns):
        return np.zeros((len(classes), 0), dtype=np.float32)
    selected_matrix = matrix[:, columns]
    support = np.asarray(selected_matrix.getnnz(axis=0)).ravel().astype(np.float64)
    weights = np.zeros((len(classes), len(columns)), dtype=np.float64)
    for class_index, class_name in enumerate(classes):
        positive_mask = labels == class_name
        positive_size = int(positive_mask.sum())
        negative_size = len(labels) - positive_size
        positive = np.asarray(
            selected_matrix[positive_mask].getnnz(axis=0)
        ).ravel().astype(np.float64)
        negative = support - positive
        weights[class_index] = (
            np.log((positive + ALPHA) / (positive_size - positive + ALPHA))
            - np.log((negative + ALPHA) / (negative_size - negative + ALPHA))
        )
    return weights.astype(np.float32)


def _fit_independent(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    classes: list[str],
    min_support: int,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(matrix.getnnz(axis=0)).ravel()
    selected = np.flatnonzero(
        (support >= min_support) & (support < matrix.shape[0])
    )
    weights = _raw_log_odds(matrix, labels, classes, selected)
    if len(selected):
        shrink = support[selected] / (support[selected] + DEFAULT_SHRINKAGE)
        weights *= shrink[None, :]
    return selected, np.clip(weights, -WEIGHT_CLIP, WEIGHT_CLIP)


def _fit_residual(
    fine: sparse.csr_matrix,
    parent: sparse.csr_matrix,
    parent_columns: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    min_support: int,
    backoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(fine.getnnz(axis=0)).ravel()
    selected = np.flatnonzero(
        (support >= min_support) & (support < fine.shape[0])
    )
    if not len(selected):
        return selected, np.zeros((len(classes), 0), dtype=np.float32)
    fine_weights = _raw_log_odds(fine, labels, classes, selected)
    parent_weights = _raw_log_odds(
        parent, labels, classes, parent_columns[selected]
    )
    reliability = support[selected] / (support[selected] + backoff)
    residual = (fine_weights - parent_weights) * reliability[None, :]
    return selected, np.clip(residual, -WEIGHT_CLIP, WEIGHT_CLIP)


def _apply_scores(
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


def _standardise_crossfit(
    train_scores: np.ndarray,
    apply_scores: np.ndarray,
    classes: list[str],
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keep = train_scores.min(axis=0) != train_scores.max(axis=0)
    train_scores = train_scores[:, keep]
    apply_scores = apply_scores[:, keep]
    names = [
        f"{prefix}__{class_name}"
        for class_name, included in zip(classes, keep)
        if included
    ]
    if not train_scores.shape[1]:
        return train_scores, apply_scores, names
    mean = train_scores.mean(axis=0)
    deviation = train_scores.std(axis=0)
    deviation[deviation < 1e-6] = 1.0
    return (
        ((train_scores - mean) / deviation).astype(np.float32),
        ((apply_scores - mean) / deviation).astype(np.float32),
        names,
    )


def cross_fitted_fine_scores(
    token_pair: TokenPair,
    parent_train: sparse.csr_matrix,
    parent_apply: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    seed: int,
    mode: str,
    min_support: int,
    backoff: float,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes = sorted(np.unique(labels).tolist())
    train_scores = np.zeros((len(labels), len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=seed
    )
    for fit_index, holdout_index in splitter.split(np.zeros(len(labels)), labels):
        if mode == "independent":
            selected, weights = _fit_independent(
                token_pair.train[fit_index], labels[fit_index], classes, min_support
            )
        elif mode == "residual":
            selected, weights = _fit_residual(
                token_pair.train[fit_index],
                parent_train[fit_index],
                token_pair.parent_columns,
                labels[fit_index],
                classes,
                min_support,
                backoff,
            )
        else:
            raise ValueError(f"지원하지 않는 mode: {mode}")
        train_scores[holdout_index] = _apply_scores(
            token_pair.train[holdout_index], selected, weights
        )

    if mode == "independent":
        selected, weights = _fit_independent(
            token_pair.train, labels, classes, min_support
        )
    else:
        selected, weights = _fit_residual(
            token_pair.train,
            parent_train,
            token_pair.parent_columns,
            labels,
            classes,
            min_support,
            backoff,
        )
    apply_scores = _apply_scores(token_pair.apply, selected, weights)
    return _standardise_crossfit(train_scores, apply_scores, classes, prefix)


def _requested_block_keys(cases: dict[str, EnrichmentCase]) -> set[tuple]:
    keys: set[tuple] = set()
    for case in cases.values():
        if case.amino_mode != "none":
            keys.add(("amino", case.amino_mode, "na", case.min_support, case.backoff))
        if case.position_mode != "none":
            keys.add((
                "position",
                case.position_mode,
                case.position_scheme,
                case.min_support,
                case.backoff,
            ))
    return keys


def prepare_seed(
    train: pd.DataFrame,
    genes: list[str],
    cases: dict[str, EnrichmentCase],
    *,
    seed: int,
    verbose: bool = True,
) -> list[PreparedFold]:
    """Prepare all outer-fold matrices once and reuse them across FE cases."""

    labels = train["SUBCLASS"].reset_index(drop=True)
    label_array = labels.to_numpy()
    splitter = StratifiedKFold(
        n_splits=OUTER_SPLITS, shuffle=True, random_state=seed
    )
    requested = _requested_block_keys(cases)
    prepared: list[PreparedFold] = []

    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(train, labels), start=1
    ):
        if verbose:
            print(f"[prepare] seed={seed} fold={fold}/{OUTER_SPLITS}", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        fit_labels = label_array[fit_index]

        parsed_fit, parsed_valid, vocabulary = BASE.fit_transform_pair(
            fit_frame, valid_frame, genes
        )
        x_fit_base, x_valid_base, base_names = BASE.build_b04_matrices(
            parsed_fit, parsed_valid, vocabulary, fit_labels
        )
        gt_fit, gt_valid, gt_names = BASE.cross_fitted_enrichment(
            parsed_fit,
            parsed_valid,
            fit_labels,
            seed=seed * 100 + fold,
        )

        token_pairs: dict[tuple[str, str], TokenPair] = {}
        for kind, _, scheme, _, _ in requested:
            token_key = (kind, scheme)
            if token_key not in token_pairs:
                token_pairs[token_key] = fit_transform_fine_tokens(
                    fit_frame,
                    valid_frame,
                    genes,
                    vocabulary.gene_types,
                    kind=kind,
                    position_scheme="p50" if scheme == "na" else scheme,
                )

        block_cache: dict[tuple, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        for key in sorted(requested):
            kind, mode, scheme, min_support, backoff = key
            pair = token_pairs[(kind, scheme)]
            prefix = (
                f"E__gene_apair__{mode}__s{min_support}__k{backoff:g}"
                if kind == "amino"
                else f"E__gene_type_pos_{scheme}__{mode}__s{min_support}__k{backoff:g}"
            )
            block_cache[key] = cross_fitted_fine_scores(
                pair,
                parsed_fit.gene_type,
                parsed_valid.gene_type,
                fit_labels,
                seed=seed * 100 + fold,
                mode=mode,
                min_support=min_support,
                backoff=backoff,
                prefix=prefix,
            )

        extras: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        audits: dict[str, dict] = {}
        for case_name, case in cases.items():
            train_blocks: list[np.ndarray] = []
            valid_blocks: list[np.ndarray] = []
            extra_names: list[str] = []
            if case.gene_type:
                train_blocks.append(gt_fit)
                valid_blocks.append(gt_valid)
                extra_names.extend(gt_names)
            if case.amino_mode != "none":
                key = ("amino", case.amino_mode, "na", case.min_support, case.backoff)
                left, right, names = block_cache[key]
                train_blocks.append(left); valid_blocks.append(right); extra_names.extend(names)
            if case.position_mode != "none":
                key = (
                    "position", case.position_mode, case.position_scheme,
                    case.min_support, case.backoff,
                )
                left, right, names = block_cache[key]
                train_blocks.append(left); valid_blocks.append(right); extra_names.extend(names)
            if train_blocks:
                train_extra = np.hstack(train_blocks).astype(np.float32)
                valid_extra = np.hstack(valid_blocks).astype(np.float32)
            else:
                train_extra = np.zeros((len(fit_index), 0), dtype=np.float32)
                valid_extra = np.zeros((len(valid_index), 0), dtype=np.float32)
            extras[case_name] = (train_extra, valid_extra, extra_names)
            audits[case_name] = {
                "raw_train_apply_concat": False,
                "vocabulary_source": "outer_fold_fit_only",
                "base_feature_count": len(base_names),
                "extra_feature_count": len(extra_names),
                "total_feature_count": len(base_names) + len(extra_names),
                "fixed_cancer_names": False,
                "fixed_exact_hotspots": False,
            }

        prepared.append(PreparedFold(
            fold,
            fit_index,
            valid_index,
            fit_labels,
            x_fit_base,
            x_valid_base,
            extras,
            audits,
        ))
    return prepared


def _make_model(kind: str, seed: int):
    common = dict(
        solver="lbfgs",
        C=LR_C,
        max_iter=LR_MAX_ITER,
        class_weight="balanced",
        random_state=seed,
    )
    if kind == "multinomial":
        return LogisticRegression(**common)
    if kind == "ovr":
        return OneVsRestClassifier(LogisticRegression(**common), n_jobs=1)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    raise ValueError(f"지원하지 않는 model kind: {kind}")


def _aligned_probability(model, matrix, classes: np.ndarray) -> np.ndarray:
    probability = model.predict_proba(matrix)
    aligned = np.zeros((matrix.shape[0], len(classes)), dtype=np.float64)
    lookup = {str(name): index for index, name in enumerate(classes)}
    for column, name in enumerate(model.classes_):
        aligned[:, lookup[str(name)]] = probability[:, column]
    return aligned


def evaluate_case(
    prepared: list[PreparedFold],
    labels: pd.Series | np.ndarray,
    case_name: str,
    *,
    seed: int,
    model_kind: str = "multinomial",
) -> OOFResult:
    label_array = np.asarray(labels)
    classes = np.asarray(sorted(np.unique(label_array).tolist()))
    probability = np.zeros((len(label_array), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []
    feature_counts: list[int] = []
    warning_count = 0
    for item in prepared:
        fit_extra, valid_extra, names = item.extras[case_name]
        x_fit = sparse.hstack(
            [item.x_fit_base, sparse.csr_matrix(fit_extra)], format="csr"
        )
        x_valid = sparse.hstack(
            [item.x_valid_base, sparse.csr_matrix(valid_extra)], format="csr"
        )
        model = _make_model(model_kind, seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_fit, item.fit_labels)
        warning_count += sum(
            issubclass(warning.category, ConvergenceWarning) for warning in caught
        )
        fold_probability = _aligned_probability(model, x_valid, classes)
        probability[item.valid_index] = fold_probability
        fold_prediction = classes[fold_probability.argmax(axis=1)]
        fold_scores.append(f1_score(
            label_array[item.valid_index],
            fold_prediction,
            average="macro",
            zero_division=0,
        ))
        feature_counts.append(x_fit.shape[1])
    prediction = classes[probability.argmax(axis=1)]
    return OOFResult(
        case_name,
        model_kind,
        seed,
        probability,
        prediction,
        classes,
        fold_scores,
        feature_counts,
        warning_count,
    ).attach_metrics(label_array)


def result_row(result: OOFResult, baseline: float | None = None) -> dict:
    return {
        "seed": result.seed,
        "case": result.name,
        "model": result.model,
        "oof_macro_f1": result.macro_f1,
        "oof_accuracy": result.accuracy,
        "delta_vs_baseline": (
            np.nan if baseline is None else result.macro_f1 - baseline
        ),
        "fold_mean": float(np.mean(result.fold_scores)),
        "fold_std": float(np.std(result.fold_scores)),
        "fold_min": float(np.min(result.fold_scores)),
        "feature_count_mean": float(np.mean(result.feature_counts)),
        "convergence_warnings": result.convergence_warning_count,
    }


def evaluate_three_way(
    prepared: list[PreparedFold],
    labels: pd.Series | np.ndarray,
    case_name: str,
    *,
    seed: int,
) -> dict[str, OOFResult]:
    """Evaluate the fixed historical 0.55/0.30/0.15 ensemble safely."""

    component = {
        name: evaluate_case(
            prepared, labels, case_name, seed=seed, model_kind=name
        )
        for name in THREE_WAY_WEIGHTS
    }
    label_array = np.asarray(labels)
    classes = component["multinomial"].classes
    mixed = sum(
        THREE_WAY_WEIGHTS[name] * component[name].probability
        for name in THREE_WAY_WEIGHTS
    )
    prediction = classes[mixed.argmax(axis=1)]
    reference = component["multinomial"]
    blend = OOFResult(
        case_name,
        "three_way_fixed",
        seed,
        mixed,
        prediction,
        classes,
        [],
        reference.feature_counts,
        sum(result.convergence_warning_count for result in component.values()),
    ).attach_metrics(label_array)
    component["three_way_fixed"] = blend
    return component


def prediction_disagreement(left: OOFResult, right: OOFResult) -> float:
    return float(np.mean(left.prediction != right.prediction))


def rescue_rate(
    anchor: OOFResult,
    candidate: OOFResult,
    labels: pd.Series | np.ndarray,
) -> float:
    truth = np.asarray(labels)
    anchor_wrong = anchor.prediction != truth
    if not anchor_wrong.any():
        return 0.0
    return float(np.mean(candidate.prediction[anchor_wrong] == truth[anchor_wrong]))
