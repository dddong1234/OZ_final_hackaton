"""Focal multiclass LGBM, pair specialists, and LR diversity evaluation.

The module reuses only the merged exp13 standalone feature contract.  Every
model receives matrices fitted inside each outer fold; no test information is
used during OOF evaluation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


def _load_exp13():
    exp13_dir = Path(__file__).resolve().parents[1] / "exp_013_standalone_pipeline_audit"
    if str(exp13_dir) not in sys.path:
        sys.path.insert(0, str(exp13_dir))
    import standalone_pipeline

    return standalone_pipeline


EXP13 = _load_exp13()
BLEND_MODEL_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)


@dataclass(frozen=True)
class MainCase:
    name: str
    objective: str
    gamma: float = 1.0
    alpha: float = 0.25
    class_weight: str | None = None
    description: str = ""


@dataclass(frozen=True)
class SpecialistCase:
    name: str
    pair_ranks: tuple[int, ...]
    mode: str
    alpha: float = 0.30
    description: str = ""


@dataclass
class PreparedFold:
    fold: int
    fit_index: np.ndarray
    valid_index: np.ndarray
    x_fit: sparse.csr_matrix
    x_valid: sparse.csr_matrix
    y_fit: np.ndarray
    y_valid: np.ndarray
    feature_count: int
    audit: dict


@dataclass
class DynamicSpecialistFold:
    fold: int
    valid_index: np.ndarray
    pairs: tuple[tuple[str, str], ...]
    probabilities: tuple[np.ndarray, ...]


@dataclass
class OOFResult:
    name: str
    seed: int
    classes: np.ndarray
    probability: np.ndarray
    prediction: np.ndarray
    fold_metrics: pd.DataFrame
    summary: dict


def main_cases() -> dict[str, MainCase]:
    cases = (
        MainCase(
            "main_01_multiclass_balanced",
            "multiclass",
            class_weight="balanced",
            description="standard multiclass + balanced",
        ),
        MainCase(
            "main_02_multiclass_unweighted",
            "multiclass",
            class_weight=None,
            description="standard multiclass without class weighting",
        ),
        MainCase(
            "main_03_focal_g1",
            "focal",
            gamma=1.0,
            alpha=0.25,
            description="screenshot focal baseline",
        ),
        MainCase(
            "main_04_focal_g2",
            "focal",
            gamma=2.0,
            alpha=0.25,
            description="stronger easy-example suppression",
        ),
        MainCase(
            "main_05_focal_g1_balanced",
            "focal",
            gamma=1.0,
            alpha=0.25,
            class_weight="balanced",
            description="focal plus class balancing interaction check",
        ),
    )
    return {case.name: case for case in cases}


def specialist_cases() -> dict[str, SpecialistCase]:
    cases = (
        SpecialistCase(
            "spec_01_rank1_soft_mass_030",
            (0,),
            "soft_mass",
            0.30,
            "outer-fold train에서 자동 발견한 최상위 유사 암종쌍",
        ),
        SpecialistCase(
            "spec_02_rank2_soft_mass_030",
            (1,),
            "soft_mass",
            0.30,
            "outer-fold train에서 자동 발견한 두 번째 유사 암종쌍",
        ),
        SpecialistCase(
            "spec_03_both_soft_mass_015",
            (0, 1),
            "soft_mass",
            0.15,
        ),
        SpecialistCase(
            "spec_04_both_soft_mass_030",
            (0, 1),
            "soft_mass",
            0.30,
        ),
        SpecialistCase(
            "spec_05_both_soft_mass_050",
            (0, 1),
            "soft_mass",
            0.50,
        ),
        SpecialistCase(
            "spec_06_both_soft_predicted_030",
            (0, 1),
            "soft_predicted",
            0.30,
            "correct only rows whose main argmax belongs to the pair",
        ),
        SpecialistCase(
            "spec_07_both_hard_predicted",
            (0, 1),
            "hard_predicted",
            1.00,
            "screenshot-style hard rerouting",
        ),
    )
    return {case.name: case for case in cases}


def prepare_seed(
    train: pd.DataFrame,
    genes: list[str],
    *,
    seed: int,
    use_fixed_contrast: bool = False,
) -> tuple[list[PreparedFold], np.ndarray, np.ndarray]:
    """Build exp13 matrices once per fold and reuse them across model cases."""

    labels = train["SUBCLASS"].to_numpy()
    classes = np.array(sorted(np.unique(labels)))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prepared: list[PreparedFold] = []
    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(train)), labels), start=1
    ):
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        fit_labels = pd.Series(labels[fit_index])
        x_fit, x_valid, names, audit = EXP13.build_design_matrices(
            fit_frame,
            valid_frame,
            fit_labels,
            genes,
            seed=seed * 100 + fold,
            use_fixed_contrast=use_fixed_contrast,
        )
        # Remove externally recognisable, preselected exact-event columns.
        # R__ recurrent missense columns remain: they are selected by support
        # from this outer-fold training split only.
        safe_columns = np.array(
            [not name.startswith(("C__", "D__exact_")) for name in names]
        )
        x_fit = x_fit[:, safe_columns]
        x_valid = x_valid[:, safe_columns]
        names = [name for name, keep in zip(names, safe_columns) if keep]
        assert not any(name.startswith(("C__", "D__exact_")) for name in names)
        audit["fixed_contrast_enabled"] = False
        audit["fixed_exact_event_columns_removed"] = True
        audit["feature_names"] = tuple(names)
        assert audit["raw_train_test_concat"] is False
        assert audit["vocabulary_source"] == "fit_frame_only"
        prepared.append(
            PreparedFold(
                fold,
                fit_index,
                valid_index,
                x_fit,
                x_valid,
                labels[fit_index],
                labels[valid_index],
                len(names),
                audit,
            )
        )
    return prepared, labels, classes


def _softmax(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    raw -= raw.max(axis=1, keepdims=True)
    probability = np.exp(raw)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def make_focal_objective(
    n_classes: int,
    *,
    gamma: float,
    alpha: float,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return multiclass focal gradients with a positive diagonal Hessian.

    LightGBM needs one positive Hessian per class logit.  The gradient is the
    analytical focal gradient.  The Hessian uses the standard stable diagonal
    approximation ``focal_factor * p * (1-p)`` and is clipped away from zero.
    """

    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    def objective(
        y_true: np.ndarray, raw_prediction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(raw_prediction, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(-1, n_classes)
        probability = _softmax(raw)
        target = np.asarray(y_true, dtype=np.int64)
        one_hot = np.eye(n_classes, dtype=np.float64)[target]
        p_true = np.clip(probability[np.arange(len(target)), target], 1e-9, 1 - 1e-9)
        if gamma == 0:
            factor = np.full_like(p_true, alpha)
        else:
            one_minus = 1.0 - p_true
            factor = alpha * (
                one_minus**gamma
                - gamma * p_true * one_minus ** (gamma - 1.0) * np.log(p_true)
            )
        gradient = factor[:, None] * (probability - one_hot)
        hessian = factor[:, None] * probability * (1.0 - probability)
        return gradient, np.maximum(hessian, 1e-6)

    return objective


def _main_parameters(seed: int, case: MainCase, n_classes: int) -> dict:
    objective: str | Callable = "multiclass"
    if case.objective == "focal":
        objective = make_focal_objective(
            n_classes, gamma=case.gamma, alpha=case.alpha
        )
    return {
        "objective": objective,
        "boosting_type": "gbdt",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 25,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_child_samples": 10,
        "min_child_weight": 1e-3,
        "class_weight": case.class_weight,
        "random_state": seed,
        "n_jobs": -1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _aligned_probability(
    model: LGBMClassifier,
    matrix: sparse.csr_matrix,
    classes: np.ndarray,
    *,
    focal: bool,
) -> np.ndarray:
    if focal:
        raw = np.asarray(model.predict(matrix, raw_score=True))
        probability = _softmax(raw)
    else:
        probability = np.asarray(model.predict_proba(matrix))
    if not np.array_equal(model.classes_, classes):
        lookup = {name: index for index, name in enumerate(model.classes_)}
        probability = probability[:, [lookup[name] for name in classes]]
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    return probability


def evaluate_lr_reference(
    prepared: list[PreparedFold], labels: np.ndarray, classes: np.ndarray, *, seed: int
) -> OOFResult:
    probability = np.zeros((len(labels), len(classes)), dtype=np.float64)
    rows = []
    for fold in prepared:
        started = perf_counter()
        model = EXP13.make_model(seed)
        model.fit(fold.x_fit, fold.y_fit)
        raw = model.predict_proba(fold.x_valid)
        lookup = {name: index for index, name in enumerate(model.classes_)}
        fold_probability = raw[:, [lookup[name] for name in classes]]
        probability[fold.valid_index] = fold_probability
        fold_prediction = classes[fold_probability.argmax(axis=1)]
        rows.append(
            {
                "fold": fold.fold,
                "f1_macro": f1_score(fold.y_valid, fold_prediction, average="macro"),
                "accuracy": accuracy_score(fold.y_valid, fold_prediction),
                "feature_count": fold.feature_count,
                "elapsed_seconds": perf_counter() - started,
            }
        )
    return _result("exp13_lr", seed, labels, classes, probability, rows)


def evaluate_main_case(
    prepared: list[PreparedFold],
    labels: np.ndarray,
    classes: np.ndarray,
    case: MainCase,
    *,
    seed: int,
) -> OOFResult:
    probability = np.zeros((len(labels), len(classes)), dtype=np.float64)
    rows = []
    for fold in prepared:
        started = perf_counter()
        model = LGBMClassifier(**_main_parameters(seed, case, len(classes)))
        model.fit(fold.x_fit, fold.y_fit)
        fold_probability = _aligned_probability(
            model, fold.x_valid, classes, focal=case.objective == "focal"
        )
        probability[fold.valid_index] = fold_probability
        fold_prediction = classes[fold_probability.argmax(axis=1)]
        rows.append(
            {
                "fold": fold.fold,
                "f1_macro": f1_score(fold.y_valid, fold_prediction, average="macro"),
                "accuracy": accuracy_score(fold.y_valid, fold_prediction),
                "feature_count": fold.feature_count,
                "elapsed_seconds": perf_counter() - started,
            }
        )
    return _result(case.name, seed, labels, classes, probability, rows)


def _result(
    name: str,
    seed: int,
    labels: np.ndarray,
    classes: np.ndarray,
    probability: np.ndarray,
    fold_rows: list[dict],
) -> OOFResult:
    prediction = classes[probability.argmax(axis=1)]
    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "case": name,
        "seed": seed,
        "oof_f1_macro": f1_score(labels, prediction, average="macro", zero_division=0),
        "oof_accuracy": accuracy_score(labels, prediction),
        "fold_f1_mean": fold_metrics["f1_macro"].mean(),
        "fold_f1_std": fold_metrics["f1_macro"].std(ddof=0),
        "feature_count_mean": fold_metrics["feature_count"].mean(),
        "elapsed_seconds": fold_metrics["elapsed_seconds"].sum(),
    }
    return OOFResult(name, seed, classes, probability, prediction, fold_metrics, summary)


def _discover_similar_class_pairs(
    fold: PreparedFold,
    *,
    top_n: int = 2,
) -> tuple[tuple[str, str], ...]:
    """Find hard class pairs from outer-fold train statistics only.

    Classes are represented by their mean mutation-gene vectors.  The pairs
    with highest cosine similarity are selected.  No class name, validation
    label, external cancer relationship, or test statistic is hard-coded.
    """

    gene_columns = np.array(
        [name.startswith("G__") for name in fold.audit["feature_names"]]
    )
    matrix = fold.x_fit[:, gene_columns]
    classes = sorted(np.unique(fold.y_fit))
    centroids = []
    for name in classes:
        centroid = np.asarray(matrix[fold.y_fit == name].mean(axis=0)).ravel()
        norm = np.linalg.norm(centroid)
        centroids.append(centroid / norm if norm > 0 else centroid)
    candidates = []
    for left_index, left in enumerate(classes):
        for right_index in range(left_index + 1, len(classes)):
            similarity = float(centroids[left_index] @ centroids[right_index])
            candidates.append((-similarity, left, classes[right_index]))
    candidates.sort()
    return tuple((left, right) for _, left, right in candidates[:top_n])


def _specialist_parameters(seed: int) -> dict:
    return {
        "n_estimators": 100,
        "learning_rate": 0.02,
        "num_leaves": 20,
        "min_child_samples": 10,
    }


def fit_specialist_probabilities(
    prepared: list[PreparedFold],
    labels: np.ndarray,
    *,
    seed: int,
) -> list[DynamicSpecialistFold]:
    """Discover and fit binary specialists inside each outer-fold train."""

    output: list[DynamicSpecialistFold] = []
    parameters = _specialist_parameters(seed)
    for fold in prepared:
        pairs = _discover_similar_class_pairs(fold, top_n=2)
        probabilities = []
        for pair in pairs:
            pair_mask = np.isin(fold.y_fit, pair)
            model = LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                reg_alpha=0.0,
                reg_lambda=0.0,
                importance_type="gain",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
                **parameters,
            )
            model.fit(fold.x_fit[pair_mask], fold.y_fit[pair_mask])
            raw = model.predict_proba(fold.x_valid)
            lookup = {name: index for index, name in enumerate(model.classes_)}
            probabilities.append(raw[:, [lookup[name] for name in pair]])
        output.append(
            DynamicSpecialistFold(
                fold.fold, fold.valid_index, pairs, tuple(probabilities)
            )
        )
    return output


def apply_specialist_case(
    main: OOFResult,
    specialist_probability: list[DynamicSpecialistFold],
    labels: np.ndarray,
    case: SpecialistCase,
) -> OOFResult:
    probability = main.probability.copy()
    class_lookup = {name: index for index, name in enumerate(main.classes)}
    original_prediction = main.prediction
    for fold_output in specialist_probability:
        rows = fold_output.valid_index
        for rank in case.pair_ranks:
            if rank >= len(fold_output.pairs):
                continue
            pair = fold_output.pairs[rank]
            left, right = pair
            columns = [class_lookup[left], class_lookup[right]]
            pair_probability = probability[np.ix_(rows, columns)]
            pair_mass = pair_probability.sum(axis=1)
            main_ratio = np.divide(
                pair_probability[:, 0],
                pair_mass,
                out=np.full(len(rows), 0.5),
                where=pair_mass > 0,
            )
            specialist_ratio = fold_output.probabilities[rank][:, 0]
            predicted_in_pair = np.isin(original_prediction[rows], pair)

            if case.mode == "soft_mass":
                weight = case.alpha * pair_mass
                new_ratio = (1.0 - weight) * main_ratio + weight * specialist_ratio
                apply_mask = np.ones(len(rows), dtype=bool)
            elif case.mode == "soft_predicted":
                new_ratio = (1.0 - case.alpha) * main_ratio + case.alpha * specialist_ratio
                apply_mask = predicted_in_pair
            elif case.mode == "hard_predicted":
                new_ratio = specialist_ratio
                apply_mask = predicted_in_pair
            else:
                raise ValueError(case.mode)

            apply_rows = rows[apply_mask]
            probability[apply_rows, columns[0]] = pair_mass[apply_mask] * new_ratio[apply_mask]
            probability[apply_rows, columns[1]] = pair_mass[apply_mask] * (1.0 - new_ratio[apply_mask])

    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    fold_rows = []
    for fold in main.fold_metrics["fold"]:
        fold_rows.append(
            {
                "fold": int(fold),
                "f1_macro": np.nan,
                "accuracy": np.nan,
                "feature_count": main.summary["feature_count_mean"],
                "elapsed_seconds": 0.0,
            }
        )
    result = _result(case.name, main.seed, labels, main.classes, probability, fold_rows)
    result.summary["delta_vs_main"] = (
        result.summary["oof_f1_macro"] - main.summary["oof_f1_macro"]
    )
    result.summary["changed_rows_vs_main"] = int(
        (result.prediction != main.prediction).sum()
    )
    return result


def diversity_metrics(
    champion: OOFResult, candidate: OOFResult, labels: np.ndarray
) -> dict:
    champion_correct = champion.prediction == labels
    candidate_correct = candidate.prediction == labels
    recovered = (~champion_correct) & candidate_correct
    reverse_loss = champion_correct & (~candidate_correct)
    both_wrong = (~champion_correct) & (~candidate_correct)
    oracle_prediction = np.where(
        champion_correct | candidate_correct, labels, candidate.prediction
    )
    return {
        "case": candidate.name,
        "f1_macro": candidate.summary["oof_f1_macro"],
        "delta_vs_lr": candidate.summary["oof_f1_macro"]
        - champion.summary["oof_f1_macro"],
        "disagreement": np.mean(champion.prediction != candidate.prediction),
        "probability_correlation": np.corrcoef(
            champion.probability.ravel(), candidate.probability.ravel()
        )[0, 1],
        "recovery_rate": recovered.sum() / max((~champion_correct).sum(), 1),
        "reverse_loss_rate": reverse_loss.sum() / max(champion_correct.sum(), 1),
        "double_fault_rate": both_wrong.mean(),
        "oracle_f1_macro": f1_score(labels, oracle_prediction, average="macro"),
        "recovered_rows": int(recovered.sum()),
        "reverse_loss_rows": int(reverse_loss.sum()),
    }


def fixed_blends(
    champion: OOFResult, candidate: OOFResult, labels: np.ndarray
) -> pd.DataFrame:
    if not np.array_equal(champion.classes, candidate.classes):
        raise ValueError("class order mismatch")
    rows = []
    for model_weight in BLEND_MODEL_WEIGHTS:
        lr_weight = 1.0 - model_weight
        probability = (
            lr_weight * champion.probability
            + model_weight * candidate.probability
        )
        prediction = champion.classes[probability.argmax(axis=1)]
        rows.append(
            {
                "lr_weight": lr_weight,
                "model_weight": model_weight,
                "f1_macro": f1_score(labels, prediction, average="macro"),
                "accuracy": accuracy_score(labels, prediction),
                "delta_vs_lr": f1_score(labels, prediction, average="macro")
                - champion.summary["oof_f1_macro"],
                "changed_rows_vs_lr": int((prediction != champion.prediction).sum()),
            }
        )
    return pd.DataFrame(rows)


def summaries(results: dict[str, OOFResult]) -> pd.DataFrame:
    return pd.DataFrame([result.summary for result in results.values()]).sort_values(
        "oof_f1_macro", ascending=False
    )


def oof_probability_frame(
    result: OOFResult,
    ids: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    prepared: list[PreparedFold],
) -> pd.DataFrame:
    """Build the team OOF contract with explicit class-column order."""

    fold_assignment = np.full(len(result.prediction), -1, dtype=np.int8)
    for fold in prepared:
        fold_assignment[fold.valid_index] = fold.fold
    assert (fold_assignment > 0).all()
    frame = pd.DataFrame(
        {
            "ID": np.asarray(ids),
            "SUBCLASS": np.asarray(labels),
            "seed": result.seed,
            "fold": fold_assignment,
        }
    )
    for column, class_name in enumerate(result.classes):
        frame[f"prob_{class_name}"] = result.probability[:, column]
    np.testing.assert_allclose(
        frame.filter(like="prob_").sum(axis=1), 1.0, atol=1e-6
    )
    return frame
