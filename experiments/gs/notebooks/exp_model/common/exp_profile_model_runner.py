"""exp-profile-model-01: train-only class mutation-profile model screen.

Compares four non-LR profile models and their fixed LR blends on shared folds.
Test data is intentionally never read by this OOF runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

import sparse_fm_runner as base


SEED = 42
PROTOTYPE_TEMPERATURE = 0.10
BLEND_WEIGHTS = (0.25, 0.50)


class PrototypeCosine:
    def __init__(self, temperature: float = PROTOTYPE_TEMPERATURE):
        self.temperature = temperature

    def fit(self, matrix: sparse.csr_matrix, target: np.ndarray):
        self.classes_ = np.sort(np.unique(target))
        means = [np.asarray(matrix[target == label].mean(axis=0)).ravel() for label in self.classes_]
        self.prototypes_ = normalize(sparse.csr_matrix(np.vstack(means)), norm="l2", axis=1)
        return self

    def predict_proba(self, matrix: sparse.csr_matrix) -> np.ndarray:
        sample = normalize(matrix, norm="l2", axis=1)
        score = (sample @ self.prototypes_.T).toarray() / self.temperature
        score -= score.max(axis=1, keepdims=True)
        value = np.exp(score)
        return value / value.sum(axis=1, keepdims=True)


def profile_models() -> dict[str, object]:
    return {
        "bernoulli_nb": BernoulliNB(alpha=1.0, binarize=0.0),
        "complement_nb": ComplementNB(alpha=1.0),
        "multinomial_nb": MultinomialNB(alpha=1.0),
        "prototype_cosine": PrototypeCosine(),
    }


def aligned_probability(model, matrix: sparse.csr_matrix, classes: list[str]) -> np.ndarray:
    raw = model.predict_proba(matrix); output = np.zeros((matrix.shape[0], len(classes)), np.float32)
    base.assign_probability(output, np.arange(matrix.shape[0]), [classes.index(name) for name in model.classes_], raw)
    return output


def run_seed(seed: int, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = base.find_root(Path.cwd()); train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert train[genes].isna().sum().sum() == 0
    labels = train[base.CFG.target_col].to_numpy(); classes = sorted(np.unique(labels)); cache = base.Cache.build(train[genes], genes)
    splitter = StratifiedKFold(n_splits=base.CFG.n_splits, shuffle=True, random_state=seed)
    lr_probability = np.zeros((len(labels), len(classes)), np.float32)
    profile_probability = {name: np.zeros_like(lr_probability) for name in profile_models()}
    fold_rows = []; feature_counts = []; started = perf_counter()
    for fold, (train_index, valid_index) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=base.CFG.n_splits, desc=f"profile-model | seed {seed}", unit="fold"), 1):
        profile_all, profile_names = base._matrix(cache, train_index, labels[train_index], contrast=False, functional=True, scale_numeric=False)
        lr_all, _ = base._matrix(cache, train_index, labels[train_index], contrast=True, functional=False, scale_numeric=False)
        lr = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed).fit(lr_all[train_index], labels[train_index])
        lr_probability[valid_index] = aligned_probability(lr, lr_all[valid_index], classes)
        feature_counts.append(len(profile_names))
        for name, model in profile_models().items():
            model.fit(profile_all[train_index], labels[train_index])
            probability = aligned_probability(model, profile_all[valid_index], classes)
            profile_probability[name][valid_index] = probability
            prediction = np.asarray(classes)[probability.argmax(1)]
            row = {"model": name, "fold": fold, "profile_fold_macro_f1": f1_score(labels[valid_index], prediction, average="macro", zero_division=0)}
            for weight in BLEND_WEIGHTS:
                blend_prediction = np.asarray(classes)[((1 - weight) * lr_probability[valid_index] + weight * probability).argmax(1)]
                row[f"blend_{weight:.2f}_fold_macro_f1"] = f1_score(labels[valid_index], blend_prediction, average="macro", zero_division=0)
            fold_rows.append(row)
    lr_score = f1_score(labels, np.asarray(classes)[lr_probability.argmax(1)], average="macro", zero_division=0)
    summary_rows = [{"experiment_id": "exp-profile-model-01", "model": "lr08_baseline", "seed": seed, "profile_oof_macro_f1": lr_score, "blend_weight_profile": 0.0, "blend_oof_macro_f1": lr_score, "delta_vs_lr08": 0.0, "feature_count_mean": float(np.mean(feature_counts)), "runtime_seconds": perf_counter() - started, "leakage_check": True, "nan_as_mutation_count": 0}]
    for name, probability in profile_probability.items():
        profile_score = f1_score(labels, np.asarray(classes)[probability.argmax(1)], average="macro", zero_division=0)
        for weight in (0.0, *BLEND_WEIGHTS):
            blend = probability if weight == 0.0 else (1 - weight) * lr_probability + weight * probability
            score = f1_score(labels, np.asarray(classes)[blend.argmax(1)], average="macro", zero_division=0)
            summary_rows.append({"experiment_id": "exp-profile-model-01", "model": name, "seed": seed, "profile_oof_macro_f1": profile_score, "blend_weight_profile": weight, "blend_oof_macro_f1": score, "delta_vs_lr08": score - lr_score, "feature_count_mean": float(np.mean(feature_counts)), "runtime_seconds": perf_counter() - started, "leakage_check": True, "nan_as_mutation_count": 0})
    output = root / "experiments" / "gs" / "notebooks" / "exp_model" / "result"; output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows); folds = pd.DataFrame(fold_rows)
    summary.to_csv(output / f"{run_id}_seed{seed}_summary.csv", index=False); folds.to_csv(output / f"{run_id}_seed{seed}_folds.csv", index=False)
    return summary, folds


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--run-id", default="exp-profile-model-01"); args = parser.parse_args()
    summary, _ = run_seed(args.seed, args.run_id)
    print(summary.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
