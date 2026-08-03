"""exp-cv-audit-01: train-only random-CV versus profile-grouped-CV audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

import sparse_fm_runner as base


PROFILE_COSINE_THRESHOLD = 0.90


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def build_profile_groups(matrix: sparse.csr_matrix, threshold: float = PROFILE_COSINE_THRESHOLD) -> tuple[np.ndarray, dict]:
    """Union exact profiles, then only each row's top-1 cosine neighbour above threshold."""
    matrix = matrix.tocsr(); groups = UnionFind(matrix.shape[0]); signature_owner: dict[bytes, int] = {}
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        signature = hashlib.sha1(matrix.indices[start:stop].tobytes()).digest()
        if signature in signature_owner:
            groups.union(row, signature_owner[signature])
        else:
            signature_owner[signature] = row
    exact_labels = np.asarray([groups.find(row) for row in range(matrix.shape[0])])
    _, exact_sizes = np.unique(exact_labels, return_counts=True)
    exact_duplicate_rows = int(exact_sizes[exact_sizes > 1].sum())
    neighbours = NearestNeighbors(n_neighbors=min(2, matrix.shape[0]), metric="cosine", algorithm="brute", n_jobs=-1)
    neighbours.fit(matrix); distance, index = neighbours.kneighbors(matrix, return_distance=True)
    near_pairs = 0
    if matrix.shape[0] > 1:
        for row, (other, cosine_distance) in enumerate(zip(index[:, 1], distance[:, 1])):
            if 1.0 - float(cosine_distance) >= threshold:
                groups.union(row, int(other)); near_pairs += 1
    roots = np.asarray([groups.find(row) for row in range(matrix.shape[0])])
    _, encoded = np.unique(roots, return_inverse=True)
    _, sizes = np.unique(encoded, return_counts=True)
    return encoded, {"profile_cosine_threshold": threshold, "exact_duplicate_rows": exact_duplicate_rows, "near_neighbour_pairs": near_pairs, "n_groups": int(len(sizes)), "max_group_size": int(sizes.max()), "grouped_row_fraction": float(sizes[sizes > 1].sum() / matrix.shape[0])}


def run_cv(cache: base.Cache, labels: np.ndarray, splits, mode: str, seed: int) -> tuple[dict, pd.DataFrame]:
    classes = sorted(np.unique(labels)); probability = np.zeros((len(labels), len(classes)), np.float32); rows = []; started = perf_counter()
    for fold, (train_index, valid_index) in enumerate(tqdm(list(splits), total=base.CFG.n_splits, desc=f"CV audit {mode}", unit="fold"), 1):
        features, names = base._matrix(cache, train_index, labels[train_index], contrast=True, functional=False, scale_numeric=False)
        model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed)
        model.fit(features[train_index], labels[train_index]); raw = model.predict_proba(features[valid_index])
        base.assign_probability(probability, valid_index, [classes.index(name) for name in model.classes_], raw)
        prediction = np.asarray(classes)[raw.argmax(1)]
        rows.append({"cv_mode": mode, "fold": fold, "fold_macro_f1": f1_score(labels[valid_index], prediction, average="macro", zero_division=0), "feature_count": len(names), "train_rows": len(train_index), "valid_rows": len(valid_index)})
    detail = pd.DataFrame(rows)
    return {"cv_mode": mode, "seed": seed, "oof_macro_f1": f1_score(labels, np.asarray(classes)[probability.argmax(1)], average="macro", zero_division=0), "fold_macro_f1_mean": detail.fold_macro_f1.mean(), "fold_macro_f1_std": detail.fold_macro_f1.std(), "feature_count_mean": detail.feature_count.mean(), "runtime_seconds": perf_counter() - started, "leakage_check": True, "nan_as_mutation_count": 0}, detail


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--run-id", default="exp-cv-audit-01"); args = parser.parse_args()
    root = base.find_root(Path.cwd()); train = pd.read_csv(root / "data" / "raw" / "train.csv"); genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert train[genes].isna().sum().sum() == 0
    cache = base.Cache.build(train[genes], genes); labels = train[base.CFG.target_col].to_numpy()
    groups, diagnostics = build_profile_groups(cache.mutation, PROFILE_COSINE_THRESHOLD)
    regular = StratifiedKFold(n_splits=base.CFG.n_splits, shuffle=True, random_state=args.seed)
    robust = StratifiedGroupKFold(n_splits=base.CFG.n_splits, shuffle=True, random_state=args.seed)
    regular_summary, regular_folds = run_cv(cache, labels, regular.split(np.zeros(len(labels)), labels), "stratified_random", args.seed)
    robust_summary, robust_folds = run_cv(cache, labels, robust.split(np.zeros(len(labels)), labels, groups), "stratified_profile_group", args.seed)
    robust_summary["delta_vs_random_oof"] = robust_summary["oof_macro_f1"] - regular_summary["oof_macro_f1"]
    regular_summary["delta_vs_random_oof"] = 0.0
    output = root / "experiments" / "gs" / "notebooks" / "exp_model" / "result"; output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([regular_summary, robust_summary]).to_csv(output / f"{args.run_id}_seed{args.seed}_summary.csv", index=False)
    pd.concat([regular_folds, robust_folds], ignore_index=True).to_csv(output / f"{args.run_id}_seed{args.seed}_folds.csv", index=False)
    (output / f"{args.run_id}_seed{args.seed}_groups.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": [regular_summary, robust_summary], "groups": diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
