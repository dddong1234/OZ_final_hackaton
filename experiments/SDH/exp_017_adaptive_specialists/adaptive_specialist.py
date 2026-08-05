"""Adaptive-capacity dynamic pair specialists for SDH exp17.

All pair discovery and capacity decisions use outer-fold train rows only.
Validation probabilities are used only after the specialist bank is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score


EXP14_DIR = Path(__file__).resolve().parents[1] / "exp_014_focal_lgbm_specialists"
if str(EXP14_DIR) not in sys.path:
    sys.path.insert(0, str(EXP14_DIR))
import lgbm_experiment as exp14


@dataclass(frozen=True)
class Preset:
    name: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int


@dataclass(frozen=True)
class AdaptiveCase:
    name: str
    capacity_policy: str
    routing: str
    threshold: float | None = None
    specialist_alpha: float = 1.0


@dataclass
class SpecialistBankFold:
    fold: int
    valid_index: np.ndarray
    pairs: tuple[tuple[str, str], ...]
    pair_supports: tuple[int, ...]
    probabilities: dict[str, tuple[np.ndarray, ...]]


def presets() -> dict[str, Preset]:
    values = (
        Preset("small", 10, 0.10, 20, 20),
        Preset("medium", 40, 0.05, 20, 15),
        Preset("large", 100, 0.02, 20, 10),
    )
    return {value.name: value for value in values}


def capacity_policies() -> tuple[str, ...]:
    fixed = tuple(
        f"fixed_{left[0]}{right[0]}"
        for left, right in (
            ("small", "small"),
            ("small", "medium"),
            ("small", "large"),
            ("medium", "small"),
            ("medium", "medium"),
            ("medium", "large"),
            ("large", "small"),
            ("large", "medium"),
            ("large", "large"),
        )
    )
    adaptive = ("support_100", "support_200", "support_400")
    return fixed + adaptive


def case_catalog() -> dict[str, AdaptiveCase]:
    routing_specs = (
        ("hard", "hard", None, 1.0),
        ("soft50", "hard", None, 0.50),
        ("soft75", "hard", None, 0.75),
        ("pairm10", "pair_margin", 0.10, 1.0),
        ("pairm20", "pair_margin", 0.20, 1.0),
        ("pairm40", "pair_margin", 0.40, 1.0),
        ("pairm60", "pair_margin", 0.60, 1.0),
        ("pairm80", "pair_margin", 0.80, 1.0),
        ("global02", "global_margin", 0.02, 1.0),
        ("global05", "global_margin", 0.05, 1.0),
        ("global10", "global_margin", 0.10, 1.0),
        ("global20", "global_margin", 0.20, 1.0),
    )
    cases = []
    for policy in capacity_policies():
        for suffix, routing, threshold, alpha in routing_specs:
            cases.append(
                AdaptiveCase(
                    f"{policy}__{suffix}", policy, routing, threshold, alpha
                )
            )
    return {case.name: case for case in cases}


def _make_specialist(seed: int, preset: Preset) -> LGBMClassifier:
    return LGBMClassifier(
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
        n_estimators=preset.n_estimators,
        learning_rate=preset.learning_rate,
        num_leaves=preset.num_leaves,
        min_child_samples=preset.min_child_samples,
    )


def fit_specialist_bank(
    prepared: list[exp14.PreparedFold],
    *,
    seed: int,
) -> list[SpecialistBankFold]:
    """Fit small/medium/large specialists using each outer-fold train only."""

    output = []
    preset_catalog = presets()
    for fold in prepared:
        pairs = exp14._discover_similar_class_pairs(fold, top_n=2)
        supports = tuple(int(np.isin(fold.y_fit, pair).sum()) for pair in pairs)
        print(
            f"[specialist bank] seed={seed} fold={fold.fold} "
            f"pairs={pairs} supports={supports}"
        )
        by_preset = {}
        for preset_name, preset in preset_catalog.items():
            print(f"  fitting preset={preset_name}")
            pair_probabilities = []
            for pair in pairs:
                pair_mask = np.isin(fold.y_fit, pair)
                model = _make_specialist(seed, preset)
                model.fit(fold.x_fit[pair_mask], fold.y_fit[pair_mask])
                raw = np.asarray(model.predict_proba(fold.x_valid), dtype=np.float64)
                lookup = {name: index for index, name in enumerate(model.classes_)}
                probability = raw[:, [lookup[name] for name in pair]]
                np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
                pair_probabilities.append(probability)
            by_preset[preset_name] = tuple(pair_probabilities)
        output.append(
            SpecialistBankFold(
                fold.fold, fold.valid_index, pairs, supports, by_preset
            )
        )
    return output


def select_preset(policy: str, rank: int, support: int) -> str:
    if policy.startswith("fixed_"):
        letters = policy.removeprefix("fixed_")
        lookup = {"s": "small", "m": "medium", "l": "large"}
        return lookup[letters[rank]]
    if policy.startswith("support_"):
        threshold = int(policy.split("_")[1])
        return "small" if support <= threshold else "large"
    raise ValueError(f"unknown capacity policy: {policy}")


def _routing_mask(
    main_probability: np.ndarray,
    main_prediction: np.ndarray,
    pair: tuple[str, str],
    columns: list[int],
    routing: str,
    threshold: float | None,
) -> np.ndarray:
    predicted_in_pair = np.isin(main_prediction, pair)
    if routing == "hard":
        return predicted_in_pair
    if threshold is None:
        raise ValueError("threshold is required")
    if routing == "pair_margin":
        pair_probability = main_probability[:, columns]
        pair_mass = pair_probability.sum(axis=1)
        normalized_margin = np.divide(
            np.abs(pair_probability[:, 0] - pair_probability[:, 1]),
            pair_mass,
            out=np.ones(len(pair_mass), dtype=np.float64),
            where=pair_mass > 0,
        )
        return predicted_in_pair & (normalized_margin <= threshold)
    if routing == "global_margin":
        top_two = np.sort(main_probability, axis=1)[:, -2:]
        global_margin = top_two[:, 1] - top_two[:, 0]
        return predicted_in_pair & (global_margin <= threshold)
    raise ValueError(f"unknown routing: {routing}")


def apply_case(
    main: exp14.OOFResult,
    bank: list[SpecialistBankFold],
    labels: np.ndarray,
    case: AdaptiveCase,
) -> exp14.OOFResult:
    probability = main.probability.copy()
    class_lookup = {name: index for index, name in enumerate(main.classes)}
    routed_counts = []
    for fold_output in bank:
        rows = fold_output.valid_index
        fold_main_probability = main.probability[rows]
        fold_main_prediction = main.prediction[rows]
        fold_routed = 0
        for rank, pair in enumerate(fold_output.pairs):
            columns = [class_lookup[pair[0]], class_lookup[pair[1]]]
            current_pair = probability[np.ix_(rows, columns)]
            pair_mass = current_pair.sum(axis=1)
            main_ratio = np.divide(
                current_pair[:, 0],
                pair_mass,
                out=np.full(len(rows), 0.5),
                where=pair_mass > 0,
            )
            preset_name = select_preset(
                case.capacity_policy, rank, fold_output.pair_supports[rank]
            )
            specialist_ratio = fold_output.probabilities[preset_name][rank][:, 0]
            mask = _routing_mask(
                fold_main_probability,
                fold_main_prediction,
                pair,
                columns,
                case.routing,
                case.threshold,
            )
            new_ratio = (
                (1.0 - case.specialist_alpha) * main_ratio
                + case.specialist_alpha * specialist_ratio
            )
            apply_rows = rows[mask]
            probability[apply_rows, columns[0]] = pair_mass[mask] * new_ratio[mask]
            probability[apply_rows, columns[1]] = pair_mass[mask] * (1.0 - new_ratio[mask])
            fold_routed += int(mask.sum())
        routed_counts.append(fold_routed)

    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    fold_rows = []
    prepared_lookup = {fold.fold: fold for fold in bank}
    for fold_number, routed_count in enumerate(routed_counts, start=1):
        rows = prepared_lookup[fold_number].valid_index
        prediction = main.classes[np.argmax(probability[rows], axis=1)]
        fold_rows.append(
            {
                "fold": fold_number,
                "f1_macro": f1_score(labels[rows], prediction, average="macro", zero_division=0),
                "accuracy": accuracy_score(labels[rows], prediction),
                "feature_count": main.summary["feature_count_mean"],
                "elapsed_seconds": 0.0,
                "routed_rows": routed_count,
            }
        )
    return exp14._result(
        case.name, main.seed, labels, main.classes, probability, fold_rows
    )


def evaluate_catalog(
    main: exp14.OOFResult,
    bank: list[SpecialistBankFold],
    labels: np.ndarray,
    lr: exp14.OOFResult,
    *,
    model_weight: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, exp14.OOFResult]]:
    rows = []
    results = {}
    for case in case_catalog().values():
        result = apply_case(main, bank, labels, case)
        blend_probability = (
            (1.0 - model_weight) * lr.probability
            + model_weight * result.probability
        )
        blend_prediction = result.classes[np.argmax(blend_probability, axis=1)]
        rows.append(
            {
                "case": case.name,
                "capacity_policy": case.capacity_policy,
                "routing": case.routing,
                "threshold": case.threshold,
                "specialist_alpha": case.specialist_alpha,
                "specialist_f1": result.summary["oof_f1_macro"],
                "blend_f1": f1_score(
                    labels, blend_prediction, average="macro", zero_division=0
                ),
                "routed_rows": int(result.fold_metrics["routed_rows"].sum()),
            }
        )
        results[case.name] = result
    leaderboard = pd.DataFrame(rows).sort_values(
        ["blend_f1", "specialist_f1"], ascending=False
    )
    return leaderboard.reset_index(drop=True), results
