"""XGBoost and CatBoost diversity study on the safe B10 representation.

The implementation deliberately keeps feature fitting separate from model
fitting.  Every vocabulary, recurrent-event selection, automatic contrast
pair, contrast gene, enrichment weight, standardisation statistic, and hybrid
feature selector is fitted on the current outer-fold training rows only.
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


def _load_safe_base():
    name = "_sdh_exp019_safe_exp013"
    if name in sys.modules:
        return sys.modules[name]
    source = (
        Path(__file__).resolve().parents[1]
        / "exp_013_standalone_pipeline_audit"
        / "standalone_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"safe baseline을 불러오지 못했습니다: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_safe_base()

OUTER_SPLITS = 5
AUTO_PAIR_INNER_SPLITS = 3
AUTO_PAIR_COUNT = 8
AUTO_GENES_PER_PAIR = 5
CONTRAST_MIN_SUPPORT = 10
LR_C = 0.07
LR_MAX_ITER = 2000
FIXED_BLEND_WEIGHTS = (0.10, 0.15, 0.20, 0.25, 0.30)
CONTRACT_LGBM_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
INCREMENTAL_CANDIDATE_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20)
FIXED_SCREEN_LGBM_WEIGHT = 0.20
FIXED_SCREEN_CANDIDATE_WEIGHT = 0.10


@dataclass(frozen=True)
class ModelCase:
    name: str
    family: str
    view: str
    n_estimators: int
    depth: int
    learning_rate: float
    min_child_weight: float = 5.0
    subsample: float = 0.8
    colsample: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 5.0
    booster: str = "gbtree"
    random_strength: float = 1.0
    bagging_temperature: float = 1.0
    description: str = ""


@dataclass
class PreparedFold:
    fold: int
    fit_index: np.ndarray
    valid_index: np.ndarray
    y_fit: np.ndarray
    views_fit: dict[str, sparse.csr_matrix]
    views_valid: dict[str, sparse.csr_matrix]
    view_names: dict[str, list[str]]
    audit: dict


@dataclass
class OOFResult:
    name: str
    family: str
    seed: int
    probability: np.ndarray
    prediction_index: np.ndarray
    classes: np.ndarray
    fold_scores: list[float]
    feature_counts: list[int]
    warning_count: int = 0

    def metrics(self, encoded_labels: np.ndarray) -> dict[str, float]:
        return {
            "oof_macro_f1": float(f1_score(
                encoded_labels, self.prediction_index, average="macro", zero_division=0
            )),
            "oof_accuracy": float(accuracy_score(encoded_labels, self.prediction_index)),
            "fold_mean": float(np.mean(self.fold_scores)),
            "fold_std": float(np.std(self.fold_scores)),
            "fold_min": float(np.min(self.fold_scores)),
            "feature_count_mean": float(np.mean(self.feature_counts)),
        }


def case_catalog() -> dict[str, ModelCase]:
    """Pre-declared broad screen: 12 XGBoost and 12 CatBoost cases."""

    cases = (
        ModelCase("x01_full_d3_400", "xgb", "full", 400, 3, 0.05, description="shallow full sparse"),
        ModelCase("x02_full_d5_400", "xgb", "full", 400, 5, 0.05, description="medium full sparse"),
        ModelCase("x03_full_d7_400", "xgb", "full", 400, 7, 0.05, description="deep interaction full sparse"),
        ModelCase("x04_full_d4_700", "xgb", "full", 700, 4, 0.03, description="slow conservative full sparse"),
        ModelCase("x05_compact_d3_400", "xgb", "compact", 400, 3, 0.05, description="aggregate compact shallow"),
        ModelCase("x06_compact_d5_700", "xgb", "compact", 700, 5, 0.03, description="aggregate compact deeper"),
        ModelCase("x07_hybrid512_d3", "xgb", "hybrid512", 500, 3, 0.05, description="compact plus 512 supported sparse"),
        ModelCase("x08_hybrid512_d5", "xgb", "hybrid512", 500, 5, 0.05, description="hybrid512 interaction"),
        ModelCase("x09_hybrid1024_d4", "xgb", "hybrid1024", 500, 4, 0.05, description="hybrid1024 balanced"),
        ModelCase("x10_full_d5_strongreg", "xgb", "full", 500, 5, 0.04, reg_alpha=0.5, reg_lambda=10.0, description="strong L1/L2"),
        ModelCase("x11_full_d5_random", "xgb", "full", 500, 5, 0.04, subsample=0.65, colsample=0.55, description="high stochastic diversity"),
        ModelCase("x12_hybrid1024_dart", "xgb", "hybrid1024", 400, 4, 0.05, booster="dart", description="DART diversity"),
        ModelCase("c01_full_d4_500", "cat", "full", 500, 4, 0.05, description="shallow symmetric full"),
        ModelCase("c02_full_d6_500", "cat", "full", 500, 6, 0.05, description="medium symmetric full"),
        ModelCase("c03_full_d8_500", "cat", "full", 500, 8, 0.05, description="deep symmetric full"),
        ModelCase("c04_full_d6_800", "cat", "full", 800, 6, 0.03, description="slow conservative full"),
        ModelCase("c05_compact_d4_500", "cat", "compact", 500, 4, 0.05, description="compact shallow"),
        ModelCase("c06_compact_d6_700", "cat", "compact", 700, 6, 0.04, description="compact medium"),
        ModelCase("c07_compact_d8_500", "cat", "compact", 500, 8, 0.05, description="compact deep interactions"),
        ModelCase("c08_hybrid512_d4", "cat", "hybrid512", 600, 4, 0.04, description="hybrid512 shallow"),
        ModelCase("c09_hybrid512_d6", "cat", "hybrid512", 600, 6, 0.04, description="hybrid512 medium"),
        ModelCase("c10_hybrid1024_d6", "cat", "hybrid1024", 600, 6, 0.04, description="hybrid1024 medium"),
        ModelCase("c11_full_d6_strongreg", "cat", "full", 600, 6, 0.04, reg_lambda=15.0, random_strength=0.5, description="strong L2 stable"),
        ModelCase("c12_hybrid1024_random", "cat", "hybrid1024", 600, 6, 0.04, random_strength=2.0, bagging_temperature=2.0, description="high stochastic diversity"),
    )
    return {case.name: case for case in cases}


def _discover_auto_pairs(
    mutation: sparse.csr_matrix,
    labels: np.ndarray,
    seed: int,
) -> tuple[tuple[str, str, int], ...]:
    """Use only outer-fit rows to discover mutually confused class pairs."""

    classes = np.asarray(sorted(np.unique(labels).tolist()))
    prediction = np.empty(len(labels), dtype=object)
    splitter = StratifiedKFold(
        n_splits=AUTO_PAIR_INNER_SPLITS, shuffle=True, random_state=seed
    )
    for inner_fit, inner_valid in splitter.split(np.zeros(len(labels)), labels):
        model = LogisticRegression(
            solver="lbfgs", C=LR_C, max_iter=300,
            class_weight="balanced", random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(mutation[inner_fit], labels[inner_fit])
        prediction[inner_valid] = model.predict(mutation[inner_valid])
    rows = []
    for left_index, left in enumerate(classes):
        for right in classes[left_index + 1:]:
            swapped = int(
                ((labels == left) & (prediction == right)).sum()
                + ((labels == right) & (prediction == left)).sum()
            )
            support = max(int((labels == left).sum() + (labels == right).sum()), 1)
            rows.append((swapped / support, str(left), str(right)))
    rows.sort(key=lambda value: (-value[0], value[1], value[2]))
    return tuple(
        (left, right, AUTO_GENES_PER_PAIR)
        for _, left, right in rows[:AUTO_PAIR_COUNT]
    )


def _auto_contrast_matrices(
    fit,
    apply,
    labels: np.ndarray,
    pairs: tuple[tuple[str, str, int], ...],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str], list[dict]]:
    fit_columns: list[sparse.csr_matrix] = []
    apply_columns: list[sparse.csr_matrix] = []
    names: list[str] = []
    audit: list[dict] = []
    gene_names = np.asarray(fit.genes)
    for left, right, top_k in pairs:
        left_mask = labels == left
        right_mask = labels == right
        if not left_mask.any() or not right_mask.any():
            continue
        left_counts = np.asarray(fit.mutation[left_mask].getnnz(axis=0)).ravel()
        right_counts = np.asarray(fit.mutation[right_mask].getnnz(axis=0)).ravel()
        support = left_counts + right_counts
        contrast = (
            left_counts / int(left_mask.sum())
            - right_counts / int(right_mask.sum())
        )
        eligible = np.flatnonzero(support >= CONTRAST_MIN_SUPPORT)
        selected = sorted(
            eligible,
            key=lambda index: (
                -abs(contrast[index]), -support[index], str(gene_names[index])
            ),
        )[:top_k]
        if not selected:
            continue
        signs = np.sign(contrast[selected]).astype(np.float32)
        fit_selected = fit.mutation[:, selected]
        apply_selected = apply.mutation[:, selected]
        fit_columns.extend([
            sparse.csr_matrix(np.asarray(fit_selected.sum(axis=1))),
            fit_selected @ sparse.csr_matrix(signs).T,
        ])
        apply_columns.extend([
            sparse.csr_matrix(np.asarray(apply_selected.sum(axis=1))),
            apply_selected @ sparse.csr_matrix(signs).T,
        ])
        names.extend([
            f"C__{left}_vs_{right}_count",
            f"C__{left}_vs_{right}_contrast",
        ])
        audit.append({
            "left": left,
            "right": right,
            "genes": [str(gene_names[index]) for index in selected],
        })
    if not fit_columns:
        return (
            sparse.csr_matrix((fit.n_rows, 0), dtype=np.float32),
            sparse.csr_matrix((apply.n_rows, 0), dtype=np.float32),
            [],
            [],
        )
    return (
        sparse.hstack(fit_columns, format="csr"),
        sparse.hstack(apply_columns, format="csr"),
        names,
        audit,
    )


def _feature_views(
    x_fit: sparse.csr_matrix,
    x_valid: sparse.csr_matrix,
    names: list[str],
) -> tuple[dict[str, sparse.csr_matrix], dict[str, sparse.csr_matrix], dict[str, list[str]]]:
    names_array = np.asarray(names, dtype=object)
    compact_prefixes = (
        "B__", "V__", "A_pair__", "S__", "C__", "E__",
    )
    explicit_compact = {
        "T__truncating_gene_count", "R__recurrent_missense_event_count"
    }
    compact = np.asarray([
        any(name.startswith(prefix) for prefix in compact_prefixes)
        or name in explicit_compact
        for name in names
    ])
    sparse_candidate = np.asarray([
        name.startswith(("G__", "T__", "R__"))
        and name not in explicit_compact
        for name in names
    ])
    support = np.asarray(x_fit.getnnz(axis=0)).ravel()
    candidate_columns = np.flatnonzero(sparse_candidate)
    ranked = sorted(
        candidate_columns,
        key=lambda index: (-support[index], str(names_array[index])),
    )

    selectors = {
        "full": np.arange(len(names), dtype=np.int32),
        "compact": np.flatnonzero(compact),
    }
    for size in (512, 1024):
        selected = np.unique(np.concatenate([
            np.flatnonzero(compact), np.asarray(ranked[:size], dtype=np.int32)
        ]))
        selectors[f"hybrid{size}"] = selected

    fit_views = {name: x_fit[:, columns] for name, columns in selectors.items()}
    valid_views = {name: x_valid[:, columns] for name, columns in selectors.items()}
    view_names = {
        name: names_array[columns].astype(str).tolist()
        for name, columns in selectors.items()
    }
    return fit_views, valid_views, view_names


def prepare_split(
    train: pd.DataFrame,
    genes: list[str],
    fit_index: np.ndarray,
    valid_index: np.ndarray,
    *,
    seed: int,
    fold: int,
) -> PreparedFold:
    """Fit every learned representation on ``fit_index`` and apply to validation."""

    labels = train["SUBCLASS"].reset_index(drop=True).to_numpy()
    fit_index = np.asarray(fit_index, dtype=np.int64)
    valid_index = np.asarray(valid_index, dtype=np.int64)
    fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
    valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
    y_fit = labels[fit_index]
    fit, valid, vocabulary = BASE.fit_transform_pair(fit_frame, valid_frame, genes)
    x_fit, x_valid, names = BASE.build_b04_matrices(
        fit, valid, vocabulary, y_fit, use_fixed_contrast=False
    )
    pairs = _discover_auto_pairs(fit.mutation, y_fit, seed)
    c_fit, c_valid, c_names, contrast_audit = _auto_contrast_matrices(
        fit, valid, y_fit, pairs
    )
    e_fit, e_valid, e_names = BASE.cross_fitted_enrichment(
        fit, valid, y_fit, seed=seed
    )
    x_fit = sparse.hstack(
        [x_fit, c_fit, sparse.csr_matrix(e_fit)], format="csr"
    )
    x_valid = sparse.hstack(
        [x_valid, c_valid, sparse.csr_matrix(e_valid)], format="csr"
    )
    names = names + c_names + e_names
    keep = BASE._nonconstant(x_fit)
    x_fit, x_valid = x_fit[:, keep], x_valid[:, keep]
    names = [name for name, included in zip(names, keep) if included]
    views_fit, views_valid, view_names = _feature_views(x_fit, x_valid, names)
    audit = {
        "raw_train_valid_concat": False,
        "vocabulary_source": "fit_index_only",
        "support_source": "fit_index_only",
        "contrast_source": "fit_index_labels_inner_oof",
        "enrichment_source": "fit_index_inner_crossfit",
        "fixed_exact_event_enabled": False,
        "fixed_cancer_pair_enabled": False,
        "auto_pair_count": len(pairs),
        "auto_contrast": contrast_audit,
        "full_feature_count": x_fit.shape[1],
        "view_feature_counts": {
            name: matrix.shape[1] for name, matrix in views_fit.items()
        },
        "minimum_full_value": float(x_fit.data.min()) if x_fit.nnz else 0.0,
    }
    return PreparedFold(
        fold, fit_index, valid_index, y_fit,
        views_fit, views_valid, view_names, audit,
    )


def prepare_seed(
    train: pd.DataFrame,
    genes: list[str],
    *,
    seed: int,
    verbose: bool = True,
) -> list[PreparedFold]:
    """Prepare safe B10 auto-contrast matrices and four model-specific views."""

    labels = train["SUBCLASS"].reset_index(drop=True).to_numpy()
    splitter = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
    prepared: list[PreparedFold] = []
    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(train)), labels), start=1
    ):
        if verbose:
            print(f"[prepare] seed={seed} fold={fold}/{OUTER_SPLITS}", flush=True)
        item = prepare_split(
            train, genes, fit_index, valid_index,
            seed=seed * 100 + fold, fold=fold,
        )
        item.audit.update({
            "vocabulary_source": "outer_fold_fit_only",
            "support_source": "outer_fold_fit_only",
            "contrast_source": "outer_fold_fit_labels_inner_oof",
            "enrichment_source": "outer_fold_fit_inner_crossfit",
        })
        prepared.append(item)
    return prepared


def _balanced_sample_weight(encoded_labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(encoded_labels, minlength=class_count).astype(np.float64)
    class_weight = len(encoded_labels) / (class_count * np.maximum(counts, 1.0))
    return class_weight[encoded_labels]


def _make_model(case: ModelCase, seed: int, class_count: int):
    if case.family == "xgb":
        from xgboost import XGBClassifier

        parameters = dict(
            objective="multi:softprob",
            num_class=class_count,
            tree_method="hist",
            eval_metric="mlogloss",
            booster=case.booster,
            n_estimators=case.n_estimators,
            max_depth=case.depth,
            learning_rate=case.learning_rate,
            min_child_weight=case.min_child_weight,
            subsample=case.subsample,
            colsample_bytree=case.colsample,
            reg_alpha=case.reg_alpha,
            reg_lambda=case.reg_lambda,
            max_bin=256,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        if case.booster == "dart":
            parameters.update(rate_drop=0.10, skip_drop=0.50, normalize_type="tree")
        return XGBClassifier(**parameters)
    if case.family == "cat":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=case.n_estimators,
            depth=case.depth,
            learning_rate=case.learning_rate,
            l2_leaf_reg=case.reg_lambda,
            random_strength=case.random_strength,
            bootstrap_type="Bayesian",
            bagging_temperature=case.bagging_temperature,
            random_seed=seed,
            thread_count=-1,
            allow_writing_files=False,
            verbose=False,
        )
    raise ValueError(f"지원하지 않는 family: {case.family}")


def _make_lgbm(seed: int):
    """The existing B10 LightGBM partner; this is not an exp19 tuning case."""

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


def _aligned_probability(model, matrix, classes: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    aligned = np.zeros((matrix.shape[0], len(classes)), dtype=np.float64)
    for column, name in enumerate(model.classes_):
        target = int(name) if np.issubdtype(np.asarray(model.classes_).dtype, np.number) else int(np.searchsorted(classes, name))
        aligned[:, target] = raw[:, column]
    np.testing.assert_allclose(aligned.sum(axis=1), 1.0, atol=1e-6)
    return aligned


def _fit_predict_lr(item: PreparedFold, classes: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    model = LogisticRegression(
        solver="lbfgs", C=LR_C, max_iter=LR_MAX_ITER,
        class_weight="balanced", random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(item.views_fit["full"], item.y_fit)
    warning_count = sum(
        issubclass(warning.category, ConvergenceWarning) for warning in caught
    )
    return _aligned_probability(model, item.views_valid["full"], classes), warning_count


def _fit_predict_lgbm(item: PreparedFold, classes: np.ndarray, seed: int) -> np.ndarray:
    encoded = np.searchsorted(classes, item.y_fit)
    model = _make_lgbm(seed)
    model.fit(item.views_fit["full"], encoded)
    return _aligned_probability(model, item.views_valid["full"], classes)


def _fit_predict_case(
    item: PreparedFold,
    classes: np.ndarray,
    case: ModelCase,
    seed: int,
) -> np.ndarray:
    encoded = np.searchsorted(classes, item.y_fit)
    model = _make_model(case, seed, len(classes))
    model.fit(
        item.views_fit[case.view],
        encoded,
        sample_weight=_balanced_sample_weight(encoded, len(classes)),
    )
    return _aligned_probability(model, item.views_valid[case.view], classes)


def evaluate_anchor(
    prepared: list[PreparedFold],
    labels: pd.Series | np.ndarray,
    *,
    seed: int,
) -> OOFResult:
    classes = np.asarray(sorted(np.unique(np.asarray(labels)).tolist()))
    encoded = np.searchsorted(classes, np.asarray(labels))
    probability = np.zeros((len(encoded), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []
    feature_counts: list[int] = []
    warning_count = 0
    for item in prepared:
        model = LogisticRegression(
            solver="lbfgs", C=LR_C, max_iter=LR_MAX_ITER,
            class_weight="balanced", random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(item.views_fit["full"], item.y_fit)
        warning_count += sum(
            issubclass(warning.category, ConvergenceWarning) for warning in caught
        )
        raw = model.predict_proba(item.views_valid["full"])
        aligned = np.zeros((len(item.valid_index), len(classes)), dtype=np.float64)
        for column, name in enumerate(model.classes_):
            aligned[:, np.searchsorted(classes, name)] = raw[:, column]
        probability[item.valid_index] = aligned
        fold_prediction = aligned.argmax(axis=1)
        fold_scores.append(float(f1_score(
            encoded[item.valid_index], fold_prediction,
            average="macro", zero_division=0,
        )))
        feature_counts.append(item.views_fit["full"].shape[1])
    return OOFResult(
        "safe_lr_anchor", "lr", seed, probability, probability.argmax(axis=1),
        classes, fold_scores, feature_counts, warning_count,
    )


def evaluate_lgbm_anchor(
    prepared: list[PreparedFold],
    labels: pd.Series | np.ndarray,
    *,
    seed: int,
) -> OOFResult:
    """Generate the existing B10 LightGBM partner on identical outer folds."""

    classes = np.asarray(sorted(np.unique(np.asarray(labels)).tolist()))
    encoded = np.searchsorted(classes, np.asarray(labels))
    probability = np.zeros((len(encoded), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []
    feature_counts: list[int] = []
    for item in prepared:
        raw = _fit_predict_lgbm(item, classes, seed * 100 + item.fold)
        probability[item.valid_index] = raw
        fold_scores.append(float(f1_score(
            encoded[item.valid_index], raw.argmax(axis=1),
            average="macro", zero_division=0,
        )))
        feature_counts.append(item.views_fit["full"].shape[1])
    return OOFResult(
        "safe_lgbm_anchor", "lgbm", seed,
        probability, probability.argmax(axis=1), classes,
        fold_scores, feature_counts, 0,
    )


def evaluate_case(
    prepared: list[PreparedFold],
    labels: pd.Series | np.ndarray,
    case: ModelCase,
    *,
    seed: int,
) -> OOFResult:
    classes = np.asarray(sorted(np.unique(np.asarray(labels)).tolist()))
    encoded = np.searchsorted(classes, np.asarray(labels))
    probability = np.zeros((len(encoded), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []
    feature_counts: list[int] = []
    for item in prepared:
        y_fit_encoded = np.searchsorted(classes, item.y_fit)
        model = _make_model(case, seed * 100 + item.fold, len(classes))
        model.fit(
            item.views_fit[case.view],
            y_fit_encoded,
            sample_weight=_balanced_sample_weight(y_fit_encoded, len(classes)),
        )
        raw = np.asarray(model.predict_proba(item.views_valid[case.view]), dtype=np.float64)
        if raw.shape != (len(item.valid_index), len(classes)):
            raise AssertionError(
                f"probability shape mismatch: {raw.shape} != {(len(item.valid_index), len(classes))}"
            )
        probability[item.valid_index] = raw
        fold_prediction = raw.argmax(axis=1)
        fold_scores.append(float(f1_score(
            encoded[item.valid_index], fold_prediction,
            average="macro", zero_division=0,
        )))
        feature_counts.append(item.views_fit[case.view].shape[1])
    return OOFResult(
        case.name, case.family, seed, probability, probability.argmax(axis=1),
        classes, fold_scores, feature_counts, 0,
    )


def diversity_metrics(
    anchor: OOFResult,
    candidate: OOFResult,
    labels: pd.Series | np.ndarray,
) -> dict[str, float]:
    classes = anchor.classes
    if not np.array_equal(classes, candidate.classes):
        raise ValueError("class order mismatch")
    encoded = np.searchsorted(classes, np.asarray(labels))
    anchor_correct = anchor.prediction_index == encoded
    candidate_correct = candidate.prediction_index == encoded
    anchor_wrong = ~anchor_correct
    oracle = anchor.prediction_index.copy()
    oracle[candidate_correct] = encoded[candidate_correct]
    return {
        "disagreement": float(np.mean(
            anchor.prediction_index != candidate.prediction_index
        )),
        "rescue_rate": float(candidate_correct[anchor_wrong].mean()) if anchor_wrong.any() else 0.0,
        "reverse_loss_rate": float((~candidate_correct)[anchor_correct].mean()) if anchor_correct.any() else 0.0,
        "oracle_macro_f1": float(f1_score(
            encoded, oracle, average="macro", zero_division=0
        )),
        "probability_correlation": float(np.corrcoef(
            anchor.probability.ravel(), candidate.probability.ravel()
        )[0, 1]),
    }


def blend_result(
    anchor: OOFResult,
    candidate: OOFResult,
    labels: pd.Series | np.ndarray,
    candidate_weight: float,
) -> dict[str, float]:
    classes = anchor.classes
    encoded = np.searchsorted(classes, np.asarray(labels))
    probability = (
        (1.0 - candidate_weight) * anchor.probability
        + candidate_weight * candidate.probability
    )
    prediction = probability.argmax(axis=1)
    return {
        "candidate_weight": candidate_weight,
        "blend_macro_f1": float(f1_score(
            encoded, prediction, average="macro", zero_division=0
        )),
        "blend_accuracy": float(accuracy_score(encoded, prediction)),
    }


def summary_row(
    result: OOFResult,
    anchor: OOFResult,
    labels: pd.Series | np.ndarray,
    case: ModelCase,
) -> dict:
    classes = result.classes
    encoded = np.searchsorted(classes, np.asarray(labels))
    metrics = result.metrics(encoded)
    anchor_metrics = anchor.metrics(encoded)
    diversity = diversity_metrics(anchor, result, labels)
    blends = {
        weight: blend_result(anchor, result, labels, weight)
        for weight in FIXED_BLEND_WEIGHTS
    }
    row = {
        "seed": result.seed,
        "case": result.name,
        "family": result.family,
        "view": case.view,
        **metrics,
        "delta_single_vs_lr": metrics["oof_macro_f1"] - anchor_metrics["oof_macro_f1"],
        **diversity,
        "description": case.description,
    }
    for weight, values in blends.items():
        token = str(int(weight * 100))
        row[f"blend{token}_f1"] = values["blend_macro_f1"]
        row[f"blend{token}_delta"] = values["blend_macro_f1"] - anchor_metrics["oof_macro_f1"]
    return row


def _score_probability(
    probability: np.ndarray,
    classes: np.ndarray,
    labels: pd.Series | np.ndarray,
) -> dict[str, float]:
    encoded = np.searchsorted(classes, np.asarray(labels))
    prediction = probability.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(
            encoded, prediction, average="macro", zero_division=0
        )),
        "accuracy": float(accuracy_score(encoded, prediction)),
    }


def fixed_incremental_metrics(
    lr: OOFResult,
    lgbm: OOFResult,
    candidate: OOFResult,
    labels: pd.Series | np.ndarray,
    *,
    lgbm_weight: float = FIXED_SCREEN_LGBM_WEIGHT,
    candidate_weight: float = FIXED_SCREEN_CANDIDATE_WEIGHT,
) -> dict[str, float]:
    """Cheap, predeclared diagnostic that reuses already trained outer OOF models.

    It is deliberately not the final selection estimate.  Strict promotion is
    made only by :func:`strict_foldlocal_incremental`.
    """

    if not (
        np.array_equal(lr.classes, lgbm.classes)
        and np.array_equal(lr.classes, candidate.classes)
    ):
        raise ValueError("class order mismatch")
    classes = lr.classes
    encoded = np.searchsorted(classes, np.asarray(labels))
    baseline = (1.0 - lgbm_weight) * lr.probability + lgbm_weight * lgbm.probability
    augmented = (1.0 - candidate_weight) * baseline + candidate_weight * candidate.probability
    baseline_prediction = baseline.argmax(axis=1)
    augmented_prediction = augmented.argmax(axis=1)
    baseline_correct = baseline_prediction == encoded
    augmented_correct = augmented_prediction == encoded
    recovered = (~baseline_correct) & augmented_correct
    damaged = baseline_correct & (~augmented_correct)
    baseline_score = _score_probability(baseline, classes, labels)
    augmented_score = _score_probability(augmented, classes, labels)
    lgbm_correct = lgbm.prediction_index == encoded
    candidate_correct = candidate.prediction_index == encoded
    lgbm_wrong = ~lgbm_correct
    return {
        "fixed_baseline_f1": baseline_score["macro_f1"],
        "fixed_augmented_f1": augmented_score["macro_f1"],
        "fixed_incremental_delta": augmented_score["macro_f1"] - baseline_score["macro_f1"],
        "base_recovered_count": int(recovered.sum()),
        "base_damaged_count": int(damaged.sum()),
        "base_net_correct_count": int(recovered.sum() - damaged.sum()),
        "base_recovery_rate": float(recovered.sum() / max((~baseline_correct).sum(), 1)),
        "base_damage_rate": float(damaged.sum() / max(baseline_correct.sum(), 1)),
        "candidate_lgbm_disagreement": float(np.mean(
            candidate.prediction_index != lgbm.prediction_index
        )),
        "candidate_lgbm_correlation": float(np.corrcoef(
            candidate.probability.ravel(), lgbm.probability.ravel()
        )[0, 1]),
        "candidate_recovers_lgbm_wrong": float(
            candidate_correct[lgbm_wrong].mean() if lgbm_wrong.any() else 0.0
        ),
        "candidate_breaks_lgbm_correct": float(
            (~candidate_correct)[lgbm_correct].mean() if lgbm_correct.any() else 0.0
        ),
    }


def fixed_incremental_table(
    lr: OOFResult,
    lgbm: OOFResult,
    candidates: dict[str, OOFResult],
    labels: pd.Series | np.ndarray,
    catalog: dict[str, ModelCase],
) -> pd.DataFrame:
    rows = []
    for name, result in candidates.items():
        rows.append({
            "case": name,
            "family": result.family,
            "view": catalog[name].view,
            "single_f1": result.metrics(np.searchsorted(result.classes, np.asarray(labels)))["oof_macro_f1"],
            **fixed_incremental_metrics(lr, lgbm, result, labels),
        })
    return pd.DataFrame(rows).sort_values(
        ["fixed_incremental_delta", "base_net_correct_count", "single_f1"],
        ascending=False,
    ).reset_index(drop=True)


def _best_lgbm_weight(
    lr_probability: np.ndarray,
    lgbm_probability: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    best = {"lgbm_weight": 0.0, "macro_f1": -1.0}
    for weight in CONTRACT_LGBM_WEIGHTS:
        probability = (1.0 - weight) * lr_probability + weight * lgbm_probability
        value = float(f1_score(
            truth, probability.argmax(axis=1), average="macro", zero_division=0
        ))
        if value > best["macro_f1"] + 1e-12:
            best = {"lgbm_weight": weight, "macro_f1": value}
    return best


def _best_incremental_weights(
    lr_probability: np.ndarray,
    lgbm_probability: np.ndarray,
    candidate_probability: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    """Select from a predeclared grid; ties prefer no/smaller candidate weight."""

    best = {
        "lr_weight": 1.0,
        "lgbm_weight": 0.0,
        "candidate_weight": 0.0,
        "macro_f1": -1.0,
    }
    for candidate_weight in INCREMENTAL_CANDIDATE_WEIGHTS:
        for lgbm_weight in CONTRACT_LGBM_WEIGHTS:
            lr_weight = 1.0 - candidate_weight - lgbm_weight
            if lr_weight < 0.50 - 1e-12:
                continue
            probability = (
                lr_weight * lr_probability
                + lgbm_weight * lgbm_probability
                + candidate_weight * candidate_probability
            )
            value = float(f1_score(
                truth, probability.argmax(axis=1),
                average="macro", zero_division=0,
            ))
            if value > best["macro_f1"] + 1e-12:
                best = {
                    "lr_weight": lr_weight,
                    "lgbm_weight": lgbm_weight,
                    "candidate_weight": candidate_weight,
                    "macro_f1": value,
                }
    return best


def build_foldlocal_anchor_cache(
    train: pd.DataFrame,
    genes: list[str],
    outer_prepared: list[PreparedFold],
    labels: pd.Series | np.ndarray,
    *,
    seed: int,
    inner_splits: int = 3,
    verbose: bool = True,
) -> list[dict]:
    """Build inner-fold-safe matrices and LR/LGBM probabilities once per seed."""

    labels_array = np.asarray(labels)
    classes = np.asarray(sorted(np.unique(labels_array).tolist()))
    cache: list[dict] = []
    for outer in outer_prepared:
        outer_indices = np.asarray(outer.fit_index)
        outer_truth = labels_array[outer_indices]
        truth_encoded = np.searchsorted(classes, outer_truth)
        lr_probability = np.zeros((len(outer_indices), len(classes)), dtype=np.float64)
        lgbm_probability = np.zeros_like(lr_probability)
        inner_items = []
        splitter = StratifiedKFold(
            n_splits=inner_splits, shuffle=True,
            random_state=seed * 100 + outer.fold,
        )
        for inner_fold, (inner_fit, inner_valid) in enumerate(
            splitter.split(np.zeros(len(outer_indices)), outer_truth), start=1
        ):
            if verbose:
                print(
                    f"[fold-local anchor] outer={outer.fold}/5 inner={inner_fold}/{inner_splits}",
                    flush=True,
                )
            item = prepare_split(
                train,
                genes,
                outer_indices[inner_fit],
                outer_indices[inner_valid],
                seed=seed * 10000 + outer.fold * 100 + inner_fold,
                fold=inner_fold,
            )
            lr_raw, _ = _fit_predict_lr(item, classes, seed * 10000 + outer.fold * 100 + inner_fold)
            lgbm_raw = _fit_predict_lgbm(item, classes, seed * 10000 + outer.fold * 100 + inner_fold)
            lr_probability[inner_valid] = lr_raw
            lgbm_probability[inner_valid] = lgbm_raw
            inner_items.append({"holdout_position": inner_valid, "prepared": item})
        cache.append({
            "outer_fold": outer.fold,
            "outer_valid_index": np.asarray(outer.valid_index),
            "inner_truth": truth_encoded,
            "inner_lr": lr_probability,
            "inner_lgbm": lgbm_probability,
            "inner_items": inner_items,
            "baseline_choice": _best_lgbm_weight(
                lr_probability, lgbm_probability, truth_encoded
            ),
        })
    return cache


def strict_foldlocal_incremental(
    cache: list[dict],
    labels: pd.Series | np.ndarray,
    lr_outer: OOFResult,
    lgbm_outer: OOFResult,
    candidate_outer: OOFResult,
    case: ModelCase,
    *,
    seed: int,
    verbose: bool = True,
) -> dict:
    """Unbiased outer estimate of adding one candidate above LR+LGBM.

    Only the candidate's inner fits are new.  Existing outer OOF probabilities
    and the shared fold-local LR/LGBM cache are reused.
    """

    classes = lr_outer.classes
    labels_array = np.asarray(labels)
    encoded = np.searchsorted(classes, labels_array)
    baseline_probability = np.zeros_like(lr_outer.probability)
    augmented_probability = np.zeros_like(lr_outer.probability)
    choices: list[dict] = []
    for outer in cache:
        inner_candidate = np.zeros_like(outer["inner_lr"])
        for item_info in outer["inner_items"]:
            holdout = item_info["holdout_position"]
            item = item_info["prepared"]
            inner_candidate[holdout] = _fit_predict_case(
                item, classes, case,
                seed * 10000 + outer["outer_fold"] * 100 + item.fold,
            )
        augmented_choice = _best_incremental_weights(
            outer["inner_lr"], outer["inner_lgbm"], inner_candidate,
            outer["inner_truth"],
        )
        baseline_choice = outer["baseline_choice"]
        valid = outer["outer_valid_index"]
        lgbm_weight = baseline_choice["lgbm_weight"]
        baseline_probability[valid] = (
            (1.0 - lgbm_weight) * lr_outer.probability[valid]
            + lgbm_weight * lgbm_outer.probability[valid]
        )
        augmented_probability[valid] = (
            augmented_choice["lr_weight"] * lr_outer.probability[valid]
            + augmented_choice["lgbm_weight"] * lgbm_outer.probability[valid]
            + augmented_choice["candidate_weight"] * candidate_outer.probability[valid]
        )
        row = {
            "fold": outer["outer_fold"],
            "baseline_lgbm_weight": lgbm_weight,
            **{key: value for key, value in augmented_choice.items() if key != "macro_f1"},
            "inner_augmented_macro_f1": augmented_choice["macro_f1"],
        }
        choices.append(row)
        if verbose:
            print(
                f"[fold-local {case.name}] outer={outer['outer_fold']}/5 "
                f"base LGBM={lgbm_weight:.2f} -> "
                f"LR/LGBM/candidate={augmented_choice['lr_weight']:.2f}/"
                f"{augmented_choice['lgbm_weight']:.2f}/"
                f"{augmented_choice['candidate_weight']:.2f}",
                flush=True,
            )
    baseline_score = _score_probability(baseline_probability, classes, labels)
    augmented_score = _score_probability(augmented_probability, classes, labels)
    baseline_prediction = baseline_probability.argmax(axis=1)
    augmented_prediction = augmented_probability.argmax(axis=1)
    baseline_correct = baseline_prediction == encoded
    augmented_correct = augmented_prediction == encoded
    recovered = (~baseline_correct) & augmented_correct
    damaged = baseline_correct & (~augmented_correct)
    fold_deltas = []
    for outer in cache:
        valid = outer["outer_valid_index"]
        base_fold = f1_score(
            encoded[valid], baseline_prediction[valid],
            average="macro", zero_division=0,
        )
        augmented_fold = f1_score(
            encoded[valid], augmented_prediction[valid],
            average="macro", zero_division=0,
        )
        fold_deltas.append(float(augmented_fold - base_fold))
    return {
        "seed": seed,
        "case": case.name,
        "family": case.family,
        "view": case.view,
        "baseline_macro_f1": baseline_score["macro_f1"],
        "augmented_macro_f1": augmented_score["macro_f1"],
        "incremental_delta": augmented_score["macro_f1"] - baseline_score["macro_f1"],
        "positive_folds": int(sum(delta > 0 for delta in fold_deltas)),
        "fold_deltas": fold_deltas,
        "recovered_count": int(recovered.sum()),
        "damaged_count": int(damaged.sum()),
        "net_correct_count": int(recovered.sum() - damaged.sum()),
        "candidate_weight_mean": float(np.mean([
            row["candidate_weight"] for row in choices
        ])),
        "candidate_zero_weight_folds": int(sum(
            row["candidate_weight"] == 0 for row in choices
        )),
        "choices": choices,
    }
