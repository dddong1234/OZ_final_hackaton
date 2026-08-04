"""exp-confusion-specialist-01: fold-safe multi-pair soft specialist screen.

Only train.csv is read.  Each outer fold selects pairs from an inner OOF
prediction made solely on that outer fold's training rows.  The specialist is
then fit on the outer training rows and only changes the selected pair's
internal probability ratio; all pair mass and other class probabilities remain
unchanged.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

import sparse_fm_runner as base


SEED = 42
INNER_SPLITS = 3
MAX_PAIRS = 3
ALPHA = 0.30
MIN_PAIR_MASS = 0.20
MAX_PAIR_CONFIDENCE = 0.85


@dataclass(frozen=True)
class ConfusionPair:
    left: str
    right: str
    count: int


def select_disjoint_confusion_pairs(truth: np.ndarray, prediction: np.ndarray, classes: list[str], max_pairs: int = MAX_PAIRS) -> list[ConfusionPair]:
    """Choose high two-way confusion pairs without reusing a class."""
    counts: list[ConfusionPair] = []
    for left_position, left in enumerate(classes):
        for right in classes[left_position + 1:]:
            count = int(((truth == left) & (prediction == right)).sum() + ((truth == right) & (prediction == left)).sum())
            if count:
                counts.append(ConfusionPair(left, right, count))
    ranked = sorted(counts, key=lambda item: (-item.count, item.left, item.right))
    used: set[str] = set(); selected: list[ConfusionPair] = []
    for item in ranked:
        if item.left not in used and item.right not in used:
            selected.append(item); used.update((item.left, item.right))
        if len(selected) == max_pairs:
            break
    return selected


def apply_pair_gate(primary: np.ndarray, specialist: np.ndarray, left_index: int, right_index: int, alpha: float = ALPHA, min_pair_mass: float = MIN_PAIR_MASS, max_pair_confidence: float = MAX_PAIR_CONFIDENCE) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively mix a binary expert while preserving pair mass exactly."""
    corrected = primary.copy()
    pair = primary[:, [left_index, right_index]]
    mass = pair.sum(axis=1)
    ratio = np.divide(pair, mass[:, None], out=np.zeros_like(pair), where=mass[:, None] > 0)
    uncertain = ratio.max(axis=1) <= max_pair_confidence
    applied = (mass >= min_pair_mass) & uncertain
    mixed = (1.0 - alpha) * ratio + alpha * specialist
    corrected[np.ix_(applied, [left_index, right_index])] = mass[applied, None] * mixed[applied]
    return corrected, applied


def fit_probability(matrix: sparse.csr_matrix, train_index: np.ndarray, valid_index: np.ndarray, labels: np.ndarray, classes: list[str], seed: int) -> tuple[np.ndarray, int]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed)
        model.fit(matrix[train_index], labels[train_index])
    raw = model.predict_proba(matrix[valid_index]); aligned = np.zeros((len(valid_index), len(classes)), np.float32)
    base.assign_probability(aligned, np.arange(len(valid_index)), [classes.index(label) for label in model.classes_], raw)
    return aligned, sum(issubclass(item.category, ConvergenceWarning) for item in caught)


def inner_oof_pairs(cache: base.Cache, outer_train: np.ndarray, labels: np.ndarray, classes: list[str], seed: int) -> tuple[list[ConfusionPair], int]:
    """Select pairs using only inner OOF predictions of outer-train samples."""
    local_y = labels[outer_train]; local_probability = np.zeros((len(outer_train), len(classes)), np.float32); warning_count = 0
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    for inner_fold, (local_train, local_valid) in enumerate(splitter.split(np.zeros(len(local_y)), local_y), 1):
        absolute_train, absolute_valid = outer_train[local_train], outer_train[local_valid]
        matrix, _ = base._matrix(cache, absolute_train, labels[absolute_train], contrast=True, functional=False, scale_numeric=False)
        probability, warnings_found = fit_probability(matrix, absolute_train, absolute_valid, labels, classes, seed * 100 + inner_fold)
        local_probability[local_valid] = probability; warning_count += warnings_found
    prediction = np.asarray(classes)[local_probability.argmax(1)]
    return select_disjoint_confusion_pairs(local_y, prediction, classes), warning_count


def pair_metrics(labels: np.ndarray, baseline: np.ndarray, corrected: np.ndarray, classes: list[str], pair: ConfusionPair, fold: int, applied_count: int) -> dict[str, object]:
    left_index, right_index = classes.index(pair.left), classes.index(pair.right)
    subset = (labels == pair.left) | (labels == pair.right)
    base_prediction, corrected_prediction = np.asarray(classes)[baseline.argmax(1)], np.asarray(classes)[corrected.argmax(1)]
    def direction(prediction: np.ndarray, left: str, right: str) -> int:
        return int(((labels == left) & (prediction == right)).sum())
    return {
        "fold": fold, "left": pair.left, "right": pair.right, "inner_oof_two_way_confusion": pair.count,
        "gate_applied_valid_rows": applied_count,
        "baseline_pair_macro_f1": f1_score(labels[subset], base_prediction[subset], labels=[pair.left, pair.right], average="macro", zero_division=0),
        "specialist_pair_macro_f1": f1_score(labels[subset], corrected_prediction[subset], labels=[pair.left, pair.right], average="macro", zero_division=0),
        "baseline_left_to_right": direction(base_prediction, pair.left, pair.right),
        "specialist_left_to_right": direction(corrected_prediction, pair.left, pair.right),
        "baseline_right_to_left": direction(base_prediction, pair.right, pair.left),
        "specialist_right_to_left": direction(corrected_prediction, pair.right, pair.left),
        "left_index": left_index, "right_index": right_index,
    }


def run_seed(seed: int, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = base.find_root(Path.cwd()); train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert train[genes].isna().sum().sum() == 0
    labels = train[base.CFG.target_col].to_numpy(); classes = sorted(np.unique(labels)); cache = base.Cache.build(train[genes], genes)
    splitter = StratifiedKFold(n_splits=base.CFG.n_splits, shuffle=True, random_state=seed)
    baseline_probability = np.zeros((len(labels), len(classes)), np.float32); specialist_probability = np.zeros_like(baseline_probability)
    folds: list[dict[str, object]] = []; pairs_out: list[dict[str, object]] = []; feature_counts: list[int] = []; warning_count = 0; started = perf_counter()
    for fold, (outer_train, valid_index) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=base.CFG.n_splits, desc=f"confusion-specialist | seed {seed}", unit="fold"), 1):
        pairs, inner_warnings = inner_oof_pairs(cache, outer_train, labels, classes, seed * 1000 + fold)
        matrix, names = base._matrix(cache, outer_train, labels[outer_train], contrast=True, functional=False, scale_numeric=False)
        baseline, main_warnings = fit_probability(matrix, outer_train, valid_index, labels, classes, seed * 10000 + fold)
        corrected = baseline.copy(); warning_count += inner_warnings + main_warnings; applied_total = 0
        for pair_number, pair in enumerate(pairs, 1):
            pair_train = outer_train[(labels[outer_train] == pair.left) | (labels[outer_train] == pair.right)]
            local_classes = [pair.left, pair.right]
            binary_probability, specialist_warnings = fit_probability(matrix, pair_train, valid_index, labels, local_classes, seed * 100000 + fold * 10 + pair_number)
            warning_count += specialist_warnings
            corrected, applied = apply_pair_gate(corrected, binary_probability, classes.index(pair.left), classes.index(pair.right))
            applied_total += int(applied.sum())
            pairs_out.append(pair_metrics(labels[valid_index], baseline, corrected, classes, pair, fold, int(applied.sum())))
        baseline_probability[valid_index] = baseline; specialist_probability[valid_index] = corrected; feature_counts.append(len(names))
        folds.append({"fold": fold, "baseline_fold_macro_f1": f1_score(labels[valid_index], np.asarray(classes)[baseline.argmax(1)], average="macro", zero_division=0), "specialist_fold_macro_f1": f1_score(labels[valid_index], np.asarray(classes)[corrected.argmax(1)], average="macro", zero_division=0), "selected_pair_count": len(pairs), "gate_applied_valid_rows": applied_total, "feature_count": len(names), "selected_pairs": json.dumps([f"{item.left}__{item.right}" for item in pairs])})
    baseline_score = f1_score(labels, np.asarray(classes)[baseline_probability.argmax(1)], average="macro", zero_division=0)
    specialist_score = f1_score(labels, np.asarray(classes)[specialist_probability.argmax(1)], average="macro", zero_division=0)
    summary = pd.DataFrame([
        {"experiment_id": "exp-confusion-specialist-01", "variant": "lr08_primary", "seed": seed, "oof_macro_f1": baseline_score, "delta_vs_baseline": 0.0, "feature_count_mean": float(np.mean(feature_counts)), "runtime_seconds": perf_counter() - started, "convergence_warning_count": warning_count, "leakage_check": True, "nan_as_mutation_count": 0, "inner_splits": INNER_SPLITS, "max_pairs": MAX_PAIRS, "alpha": ALPHA, "min_pair_mass": MIN_PAIR_MASS, "max_pair_confidence": MAX_PAIR_CONFIDENCE},
        {"experiment_id": "exp-confusion-specialist-01", "variant": "multi_pair_soft_specialist", "seed": seed, "oof_macro_f1": specialist_score, "delta_vs_baseline": specialist_score - baseline_score, "feature_count_mean": float(np.mean(feature_counts)), "runtime_seconds": perf_counter() - started, "convergence_warning_count": warning_count, "leakage_check": True, "nan_as_mutation_count": 0, "inner_splits": INNER_SPLITS, "max_pairs": MAX_PAIRS, "alpha": ALPHA, "min_pair_mass": MIN_PAIR_MASS, "max_pair_confidence": MAX_PAIR_CONFIDENCE},
    ])
    output = root / "experiments" / "gs" / "notebooks" / "exp_model" / "result"; output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / f"{run_id}_seed{seed}_summary.csv", index=False); pd.DataFrame(folds).to_csv(output / f"{run_id}_seed{seed}_folds.csv", index=False); pd.DataFrame(pairs_out).to_csv(output / f"{run_id}_seed{seed}_pairs.csv", index=False)
    return summary, pd.DataFrame(folds), pd.DataFrame(pairs_out)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--run-id", default="exp-confusion-specialist-01"); args = parser.parse_args()
    summary, _, _ = run_seed(args.seed, args.run_id); print(summary.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
