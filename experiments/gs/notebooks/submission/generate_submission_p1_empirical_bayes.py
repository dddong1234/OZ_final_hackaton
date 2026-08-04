"""Generate the validated P1 + Empirical-Bayes LR submission.

Contract: all learned objects use train labels/rows only. Test is first read only
after the three full-train models have been fit, then receives fixed transforms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


TRAIN_SEEDS = (42, 777, 2024)
RUN_ID = "submission_p1_empirical_bayes_3seed"


def project_root() -> Path:
    here = Path(__file__).resolve()
    for path in (here, *here.parents):
        if (path / "data/raw/train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv를 찾지 못했습니다.")


def load_p1_modules():
    common = project_root() / "experiments/gs/notebooks/exp_model_002/common"
    sys.path.insert(0, str(common))
    from legacy_p1_reference import load_reference
    from p1_core import apply_log_odds, fit_log_odds, fit_lr, normalize_proba

    return (*load_reference(), apply_log_odds, fit_log_odds, fit_lr, normalize_proba)


def standardize_train(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = scores.mean(axis=0, keepdims=True)
    std = np.maximum(scores.std(axis=0, keepdims=True), 1e-6)
    return (scores - mean) / std, mean, std


def token_sets(tokens: pd.DataFrame, row_count: int) -> list[set[str]]:
    result = [set() for _ in range(row_count)]
    for row, token in tokens.itertuples(index=False):
        result[int(row)].add(token)
    return result


def cross_fitted_empirical_scores(sets: list[set[str]], index: np.ndarray, y: np.ndarray, classes: np.ndarray, seed: int,
                                  fit_log_odds, apply_log_odds) -> np.ndarray:
    result = np.zeros((len(index), len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    for inner_train, inner_valid in splitter.split(index, y[index]):
        train_idx, valid_idx = index[inner_train], index[inner_valid]
        weights = fit_log_odds([sets[i] for i in train_idx], y[train_idx], classes, empirical_bayes=True)
        result[inner_valid] = apply_log_odds([sets[i] for i in valid_idx], weights, classes)
    return result


def train_models(base, enrichment, apply_log_odds, fit_log_odds, fit_lr, train: pd.DataFrame, genes: list[str], seeds: tuple[int, ...]):
    cache = base.Cache.build(train[genes], genes)
    y = train[base.CFG.target_col].to_numpy()
    classes = np.asarray(sorted(np.unique(y)))
    train_index = np.arange(len(train))
    matrix, _ = base._matrix(cache, train_index, y, contrast=True, functional=False, scale_numeric=False)
    sets = token_sets(enrichment.gene_event_type_tokens(cache.events), len(train))
    learned = []

    for seed in tqdm(seeds, desc="full-train P1+EB models", unit="seed"):
        inner = cross_fitted_empirical_scores(sets, train_index, y, classes, seed, fit_log_odds, apply_log_odds)
        weights = fit_log_odds([sets[i] for i in train_index], y, classes, empirical_bayes=True)
        inner_z, mean, std = standardize_train(inner)
        x_train = hstack([matrix, csr_matrix(inner_z)], format="csr")
        model, warnings = fit_lr(x_train, y, seed)
        learned.append({"seed": seed, "model": model, "weights": weights, "mean": mean, "std": std,
                        "warning_count": int(warnings), "feature_count": int(x_train.shape[1])})
    return cache, y, classes, matrix.shape[1], learned


def transform_test(base, enrichment, apply_log_odds, train: pd.DataFrame, test: pd.DataFrame, genes: list[str], y: np.ndarray, classes: np.ndarray,
                   expected_base_features: int, learned: list[dict], normalize_proba):
    # Combining rows is only a row-local parse/cache construction. _matrix receives
    # full train indices and labels, so its selected vocabulary/statistics are train-only.
    combined = pd.concat([train[genes], test[genes]], axis=0, ignore_index=True)
    cache = base.Cache.build(combined, genes)
    train_index = np.arange(len(train))
    test_index = np.arange(len(train), len(combined))
    matrix, _ = base._matrix(cache, train_index, y, contrast=True, functional=False, scale_numeric=False)
    if matrix.shape[1] != expected_base_features:
        raise RuntimeError(f"P1 base feature mismatch: train={expected_base_features}, final={matrix.shape[1]}")
    sets = token_sets(enrichment.gene_event_type_tokens(cache.events), len(combined))
    probabilities = []
    for state in tqdm(learned, desc="fixed test transforms", unit="seed"):
        score = apply_log_odds([sets[i] for i in test_index], state["weights"], classes)
        x_test = hstack([matrix[test_index], csr_matrix((score - state["mean"]) / state["std"])], format="csr")
        if x_test.shape[1] != state["feature_count"]:
            raise RuntimeError("train/test feature dimension mismatch")
        probabilities.append(normalize_proba(state["model"].predict_proba(x_test)))
    return np.mean(probabilities, axis=0), int(getattr(cache, "nan_as_mutation_count", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="train-only contract check; does not read test")
    args = parser.parse_args()

    root = project_root()
    base, enrichment, apply_log_odds, fit_log_odds, fit_lr, normalize_proba = load_p1_modules()
    raw = root / "data/raw"
    train = pd.read_csv(raw / "train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    if int(train[genes].isna().sum().sum()) != 0:
        raise RuntimeError("train NaN contract violated")
    _, y, classes, base_feature_count, learned = train_models(base, enrichment, apply_log_odds, fit_log_odds, fit_lr, train, genes, TRAIN_SEEDS)
    if args.dry_run:
        print({"dry_run": True, "train_rows": len(train), "classes": len(classes), "feature_count": learned[0]["feature_count"],
               "warning_count": sum(item["warning_count"] for item in learned), "test_read_after_fit_only": True})
        return

    # Test is read only here: after all train-only model fitting has completed.
    test = pd.read_csv(raw / "test.csv")
    template = pd.read_csv(raw / "sample_submission.csv")
    if list(test[base.CFG.id_col]) != list(template[base.CFG.id_col]):
        raise RuntimeError("sample_submission ID order does not match test")
    if list(test[genes].columns) != genes:
        raise RuntimeError("test gene columns do not exactly match train")
    test_nan_cells = int(test[genes].isna().sum().sum())
    probability, parsed_nan_mutations = transform_test(base, enrichment, apply_log_odds, train, test, genes, y, classes, base_feature_count, learned, normalize_proba)
    nan_as_mutation_count = 0
    if parsed_nan_mutations != 0 or nan_as_mutation_count != 0:
        raise RuntimeError("NaN was interpreted as a mutation")

    output = args.output or (root / "experiments/gs/notebooks/submission" / f"{RUN_ID}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission = template.copy()
    submission[base.CFG.target_col] = classes[probability.argmax(axis=1)]
    if list(submission.columns) != list(template.columns) or len(submission) != len(test):
        raise RuntimeError("submission template contract violated")
    submission.to_csv(output, index=False)
    audit = {
        "run_id": RUN_ID, "seeds": TRAIN_SEEDS, "model": "P1 + Empirical-Bayes enrichment LR probability average",
        "lr": {"solver": "lbfgs", "C": 0.07, "max_iter": 2000, "class_weight": "balanced"},
        "feature_count": learned[0]["feature_count"], "convergence_warning_count": sum(item["warning_count"] for item in learned),
        "test_nan_cells": test_nan_cells, "nan_as_mutation_count": nan_as_mutation_count,
        "test_used_for_fit": False, "test_read_after_fit_only": True,
        "matrix_fit_index": "full_train_only", "submission": str(output), "rows": len(submission),
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
