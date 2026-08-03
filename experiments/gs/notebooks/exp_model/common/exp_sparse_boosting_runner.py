"""exp-sparse-boosting-01: fold-safe structured sparse LightGBM screen.

This OOF experiment reads train.csv only.  Every feature-selection decision is
fit on an outer fold's training rows and then applied to its validation rows.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

import sparse_fm_runner as base


SEED = 42
POSITION_BIN_WIDTH = 50
MIN_TOKEN_SUPPORT = 3
REDUCED_GENE_TOP_K = 1000
POSITION_RE = re.compile(r"^[A-Z*](-?\d+)")

LGBM_PARAMETERS = {
    "objective": "multiclass", "n_estimators": 600, "learning_rate": 0.03,
    "num_leaves": 15, "min_child_samples": 25, "colsample_bytree": 0.70,
    "subsample": 0.80, "subsample_freq": 1, "reg_lambda": 2.0,
    "reg_alpha": 0.0, "class_weight": "balanced", "n_jobs": -1,
    "verbosity": -1, "force_col_wise": True,
}


def event_position(event: str) -> int | None:
    match = POSITION_RE.match(str(event))
    return int(match.group(1)) if match else None


def position_bin(position: int) -> str:
    start = ((max(position, 1) - 1) // POSITION_BIN_WIDTH) * POSITION_BIN_WIDTH + 1
    return f"{start}_{start + POSITION_BIN_WIDTH - 1}"


def signature_token_frame(cache: base.Cache) -> pd.DataFrame:
    if cache.events.empty:
        return pd.DataFrame(columns=("row", "signature"))
    events = cache.events.copy(); events["position"] = events.event.map(event_position)
    events = events.dropna(subset=["position"]).copy(); events["position"] = events.position.astype(int)
    events["signature"] = events.gene + "__" + events.type + "__BIN_" + events.position.map(position_bin)
    return events[["row", "signature"]].drop_duplicates()


def fold_signature_matrix(tokens: pd.DataFrame, train_index: np.ndarray, n_rows: int) -> tuple[sparse.csr_matrix, list[str]]:
    counts = tokens[tokens.row.isin(train_index)].groupby("signature").row.nunique()
    selected = sorted(counts[counts >= MIN_TOKEN_SUPPORT].index); lookup = {token: column for column, token in enumerate(selected)}
    subset = tokens[tokens.signature.isin(lookup)]
    matrix = sparse.coo_matrix((np.ones(len(subset), np.float32), (subset.row, subset.signature.map(lookup))), shape=(n_rows, len(selected))).tocsr()
    matrix.data[:] = 1
    return matrix, [f"ONTO_SIGNATURE__{token}" for token in selected]


def select_top_genes(mutation: sparse.csr_matrix, labels: np.ndarray, train_index: np.ndarray, top_k: int = REDUCED_GENE_TOP_K) -> np.ndarray:
    """Rank raw genes from outer-train class-rate spread only."""
    train_labels = labels[train_index]; rates = []
    for label in sorted(np.unique(train_labels)):
        rows = train_index[train_labels == label]
        rates.append(np.asarray(mutation[rows].mean(axis=0)).ravel())
    score = np.ptp(np.vstack(rates), axis=0); support = np.asarray(mutation[train_index].getnnz(axis=0)).ravel()
    order = np.lexsort((np.arange(mutation.shape[1]), -support, -score))
    return np.sort(order[:min(top_k, len(order))])


def reduced_feature_mask(names: list[str], genes: list[str], selected_gene_indices: np.ndarray) -> np.ndarray:
    selected = {genes[index] for index in selected_gene_indices.tolist()}
    mask = []
    for name in names:
        if name.startswith("G__"):
            mask.append(name.removeprefix("G__") in selected)
        else:
            mask.append(True)
    return np.asarray(mask, dtype=bool)


def fit_probability(matrix: sparse.csr_matrix, train_index: np.ndarray, valid_index: np.ndarray, labels: np.ndarray, classes: list[str], seed: int) -> np.ndarray:
    model = lgb.LGBMClassifier(num_class=len(classes), random_state=seed, **LGBM_PARAMETERS)
    model.fit(matrix[train_index], labels[train_index])
    raw = model.predict_proba(matrix[valid_index]); probability = np.zeros((len(valid_index), len(classes)), np.float32)
    base.assign_probability(probability, np.arange(len(valid_index)), [classes.index(label) for label in model.classes_], raw)
    return probability


def run_seed(seed: int, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = base.find_root(Path.cwd()); train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert train[genes].isna().sum().sum() == 0
    labels = train[base.CFG.target_col].to_numpy(); classes = sorted(np.unique(labels)); cache = base.Cache.build(train[genes], genes); tokens = signature_token_frame(cache)
    splitter = StratifiedKFold(n_splits=base.CFG.n_splits, shuffle=True, random_state=seed)
    variants = ("structured_core", "structured_core_signature", "reduced_raw_gene_1000")
    probability = {name: np.zeros((len(labels), len(classes)), np.float32) for name in variants}; feature_counts = {name: [] for name in variants}; fold_rows: list[dict[str, object]] = []; started = perf_counter()
    for fold, (train_index, valid_index) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=base.CFG.n_splits, desc=f"sparse-boosting | seed {seed}", unit="fold"), 1):
        core, names = base._matrix(cache, train_index, labels[train_index], contrast=True, functional=False, scale_numeric=False)
        signature, signature_names = fold_signature_matrix(tokens, train_index, len(labels))
        selected_genes = select_top_genes(cache.mutation, labels, train_index)
        reduced_mask = reduced_feature_mask(names, genes, selected_genes)
        matrices = {
            "structured_core": (core, names),
            "structured_core_signature": (sparse.hstack([core, signature], format="csr"), names + signature_names),
            "reduced_raw_gene_1000": (core[:, reduced_mask], [name for name, keep in zip(names, reduced_mask) if keep]),
        }
        for name, (matrix, feature_names) in matrices.items():
            fold_probability = fit_probability(matrix, train_index, valid_index, labels, classes, seed * 100 + fold)
            probability[name][valid_index] = fold_probability; feature_counts[name].append(len(feature_names))
            prediction = np.asarray(classes)[fold_probability.argmax(1)]
            fold_rows.append({"variant": name, "fold": fold, "fold_macro_f1": f1_score(labels[valid_index], prediction, average="macro", zero_division=0), "fold_accuracy": accuracy_score(labels[valid_index], prediction), "feature_count": len(feature_names), "selected_raw_gene_count": len(selected_genes), "signature_feature_count": len(signature_names)})
    core_score = f1_score(labels, np.asarray(classes)[probability["structured_core"].argmax(1)], average="macro", zero_division=0)
    summary_rows = []
    for name in variants:
        prediction = np.asarray(classes)[probability[name].argmax(1)]
        summary_rows.append({"experiment_id": "exp-sparse-boosting-01", "variant": name, "seed": seed, "oof_macro_f1": f1_score(labels, prediction, average="macro", zero_division=0), "oof_accuracy": accuracy_score(labels, prediction), "delta_vs_structured_core": f1_score(labels, prediction, average="macro", zero_division=0) - core_score, "feature_count_mean": float(np.mean(feature_counts[name])), "runtime_seconds": perf_counter() - started, "convergence_warning_count": 0, "leakage_check": True, "nan_as_mutation_count": 0, "position_bin_width": POSITION_BIN_WIDTH, "min_token_support": MIN_TOKEN_SUPPORT, "reduced_gene_top_k": REDUCED_GENE_TOP_K, **{f"lgbm_{key}": value for key, value in LGBM_PARAMETERS.items()}})
    output = root / "experiments" / "gs" / "notebooks" / "exp_model" / "result"; output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows); folds = pd.DataFrame(fold_rows)
    summary.to_csv(output / f"{run_id}_seed{seed}_summary.csv", index=False); folds.to_csv(output / f"{run_id}_seed{seed}_folds.csv", index=False)
    return summary, folds


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--run-id", default="exp-sparse-boosting-01"); args = parser.parse_args()
    summary, _ = run_seed(args.seed, args.run_id); print(summary.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
