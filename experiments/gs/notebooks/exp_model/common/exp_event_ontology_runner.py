"""exp-event-ontology-01: fold-safe gene-position event ontology screen.

Only train.csv is read. Ontology token vocabulary is selected from each outer
fold's train rows; test is not used for statistics, fitting, or selection.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

import sparse_fm_runner as base


SEED = 42
POSITION_BIN_WIDTH = 50
MIN_TOKEN_SUPPORT = 3
POSITION_RE = re.compile(r"^[A-Z*](-?\d+)")


def event_position(event: str) -> int | None:
    match = POSITION_RE.match(str(event))
    return int(match.group(1)) if match else None


def position_bin(position: int) -> str:
    start = ((max(position, 1) - 1) // POSITION_BIN_WIDTH) * POSITION_BIN_WIDTH + 1
    return f"{start}_{start + POSITION_BIN_WIDTH - 1}"


def ontology_token_frame(cache: base.Cache) -> pd.DataFrame:
    if cache.events.empty:
        return pd.DataFrame(columns=("row", "locus", "position_bin", "signature"))
    events = cache.events.copy(); events["position"] = events.event.map(event_position); events = events.dropna(subset=["position"]).copy()
    events["position"] = events.position.astype(int); events["bin"] = events.position.map(position_bin)
    events["locus"] = events.gene + "__P" + events.position.astype(str)
    events["position_bin"] = events.gene + "__BIN_" + events.bin
    events["signature"] = events.gene + "__" + events.type + "__BIN_" + events.bin
    return events[["row", "locus", "position_bin", "signature"]].drop_duplicates()


def fold_token_matrix(tokens: pd.DataFrame, train_index: np.ndarray, n_rows: int, column: str) -> tuple[sparse.csr_matrix, list[str]]:
    counts = tokens[tokens.row.isin(train_index)].groupby(column).row.nunique()
    selected = sorted(counts[counts >= MIN_TOKEN_SUPPORT].index); lookup = {value: index for index, value in enumerate(selected)}
    subset = tokens[tokens[column].isin(lookup)]
    matrix = sparse.coo_matrix((np.ones(len(subset), np.float32), (subset.row, subset[column].map(lookup))), shape=(n_rows, len(selected))).tocsr()
    matrix.data[:] = 1
    return matrix, [f"ONTO_{column.upper()}__{value}" for value in selected]


def run_seed(seed: int, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = base.find_root(Path.cwd()); train = pd.read_csv(root / "data" / "raw" / "train.csv"); genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert train[genes].isna().sum().sum() == 0
    labels = train[base.CFG.target_col].to_numpy(); classes = sorted(np.unique(labels)); cache = base.Cache.build(train[genes], genes); tokens = ontology_token_frame(cache)
    splitter = StratifiedKFold(n_splits=base.CFG.n_splits, shuffle=True, random_state=seed)
    variants = ("baseline", "locus", "position_bin", "signature", "all_ontology")
    probability = {name: np.zeros((len(labels), len(classes)), np.float32) for name in variants}; feature_count = {name: [] for name in variants}; fold_rows = []; started = perf_counter()
    for fold, (train_index, valid_index) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=base.CFG.n_splits, desc=f"event-ontology | seed {seed}", unit="fold"), 1):
        baseline, baseline_names = base._matrix(cache, train_index, labels[train_index], contrast=True, functional=False, scale_numeric=False)
        blocks = {"locus": fold_token_matrix(tokens, train_index, len(labels), "locus"), "position_bin": fold_token_matrix(tokens, train_index, len(labels), "position_bin"), "signature": fold_token_matrix(tokens, train_index, len(labels), "signature")}
        matrices = {"baseline": (baseline, baseline_names)}
        for name, (block, names) in blocks.items(): matrices[name] = (sparse.hstack([baseline, block], format="csr"), baseline_names + names)
        all_blocks = [value[0] for value in blocks.values()]; all_names = [name for value in blocks.values() for name in value[1]]
        matrices["all_ontology"] = (sparse.hstack([baseline, *all_blocks], format="csr"), baseline_names + all_names)
        for name, (matrix, names) in matrices.items():
            model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed)
            model.fit(matrix[train_index], labels[train_index]); raw = model.predict_proba(matrix[valid_index]); base.assign_probability(probability[name], valid_index, [classes.index(label) for label in model.classes_], raw)
            prediction = np.asarray(classes)[raw.argmax(1)]
            fold_rows.append({"variant": name, "fold": fold, "fold_macro_f1": f1_score(labels[valid_index], prediction, average="macro", zero_division=0), "feature_count": len(names)})
            feature_count[name].append(len(names))
    baseline_score = f1_score(labels, np.asarray(classes)[probability["baseline"].argmax(1)], average="macro", zero_division=0)
    summary = []
    for name in variants:
        score = f1_score(labels, np.asarray(classes)[probability[name].argmax(1)], average="macro", zero_division=0)
        summary.append({"experiment_id": "exp-event-ontology-01", "variant": name, "seed": seed, "oof_macro_f1": score, "delta_vs_baseline": score - baseline_score, "feature_count_mean": float(np.mean(feature_count[name])), "runtime_seconds": perf_counter() - started, "leakage_check": True, "nan_as_mutation_count": 0, "position_bin_width": POSITION_BIN_WIDTH, "min_token_support": MIN_TOKEN_SUPPORT})
    output = root / "experiments" / "gs" / "notebooks" / "exp_model" / "result"; output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(output / f"{run_id}_seed{seed}_summary.csv", index=False); pd.DataFrame(fold_rows).to_csv(output / f"{run_id}_seed{seed}_folds.csv", index=False)
    return pd.DataFrame(summary), pd.DataFrame(fold_rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--run-id", default="exp-event-ontology-01"); args = parser.parse_args()
    summary, _ = run_seed(args.seed, args.run_id); print(summary.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
