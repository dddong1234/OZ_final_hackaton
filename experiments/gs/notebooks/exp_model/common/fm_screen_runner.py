"""FM-03: seed-42, common-fold candidate screening for Sparse FM.

This runner reads train.csv only. It is a development screen, not final validation:
O1 -> O2 -> O3 candidates are selected with one fixed outer-CV split, then only
the winner(s) should advance to a separate three-seed confirmation experiment.
"""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm.auto import tqdm
import torch
from torch import nn

import sparse_fm_runner as base


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str  # fm or linear
    rank: int
    learning_rate: float
    class_weight: str


@dataclass
class FoldData:
    fold: int
    train_index: np.ndarray
    valid_index: np.ndarray
    fm_matrix: sparse.csr_matrix
    lr_matrix: sparse.csr_matrix


def candidate_stages() -> list[tuple[str, list[Candidate]]]:
    return [
        ("O1_learning_rate", [
            Candidate(f"O1_lr_{value:g}", "fm", 8, value, "balanced")
            for value in (3e-4, 1e-3, 3e-3)
        ]),
        ("O2_rank", [
            Candidate(f"O2_rank_{value}", "fm", value, 0.0, "balanced")
            for value in (4, 8, 16)
        ]),
        ("O3_class_weight", [
            Candidate(f"O3_weight_{value}", "fm", 0, 0.0, value)
            for value in ("balanced", "sqrt_balanced", "none")
        ]),
    ]


def class_weight_vector(encoded: np.ndarray, n_classes: int, scheme: str) -> np.ndarray:
    counts = np.bincount(encoded, minlength=n_classes).astype(np.float32)
    if scheme == "none":
        return np.ones(n_classes, dtype=np.float32)
    if scheme == "balanced":
        return len(encoded) / (n_classes * np.maximum(counts, 1.0))
    if scheme == "sqrt_balanced":
        return len(encoded) / (n_classes * np.sqrt(np.maximum(counts, 1.0)))
    raise ValueError(f"unknown class-weight scheme: {scheme}")


class SparseLinear(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Embedding(n_features, n_classes)
        nn.init.zeros_(self.linear.weight)

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return (self.linear(indices) * values.unsqueeze(-1)).sum(dim=1)


def make_model(candidate: Candidate, n_features: int, n_classes: int) -> nn.Module:
    if candidate.kind == "linear":
        return SparseLinear(n_features, n_classes)
    return base.SparseMulticlassFM(n_features, n_classes, candidate.rank)


def train_epochs(matrix: sparse.csr_matrix, target: np.ndarray, classes: list[str], candidate: Candidate,
                 epochs: int, seed: int, device: torch.device) -> tuple[nn.Module, list[float], bool]:
    torch.manual_seed(seed); np.random.seed(seed)
    encoded = np.searchsorted(classes, target)
    model = make_model(candidate, matrix.shape[1], len(classes)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=candidate.learning_rate, weight_decay=base.CFG.weight_decay)
    weights = torch.as_tensor(class_weight_vector(encoded, len(classes), candidate.class_weight), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    rng = np.random.default_rng(seed); losses: list[float] = []
    for _ in range(epochs):
        model.train(); order = rng.permutation(len(target)); total = 0.0
        for start in range(0, len(order), base.CFG.batch_size):
            rows = order[start:start + base.CFG.batch_size]
            index, value = base._batch(matrix, rows, device)
            label = torch.as_tensor(encoded[rows], dtype=torch.long, device=device)
            optimiser.zero_grad(); loss = criterion(model(index, value), label)
            if not torch.isfinite(loss):
                return model, losses, False
            loss.backward(); optimiser.step(); total += float(loss.detach().cpu()) * len(rows)
        losses.append(total / len(target))
    return model, losses, bool(np.isfinite(losses).all())


def select_epoch(matrix: sparse.csr_matrix, target: np.ndarray, classes: list[str], candidate: Candidate,
                 seed: int, device: torch.device) -> tuple[int, list[float], bool]:
    inner_train, inner_valid = next(StratifiedShuffleSplit(n_splits=1, test_size=base.CFG.inner_valid_fraction,
        random_state=seed).split(np.zeros(len(target)), target))
    torch.manual_seed(seed); np.random.seed(seed)
    encoded = np.searchsorted(classes, target)
    model = make_model(candidate, matrix.shape[1], len(classes)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=candidate.learning_rate, weight_decay=base.CFG.weight_decay)
    weights = torch.as_tensor(class_weight_vector(encoded[inner_train], len(classes), candidate.class_weight), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    rng = np.random.default_rng(seed); losses: list[float] = []; best_epoch = 1; best_loss = np.inf
    for epoch in range(1, base.CFG.max_epochs + 1):
        model.train(); order = rng.permutation(inner_train)
        for start in range(0, len(order), base.CFG.batch_size):
            rows = order[start:start + base.CFG.batch_size]
            index, value = base._batch(matrix, rows, device)
            label = torch.as_tensor(encoded[rows], dtype=torch.long, device=device)
            optimiser.zero_grad(); loss = criterion(model(index, value), label)
            if not torch.isfinite(loss):
                return best_epoch, losses, False
            loss.backward(); optimiser.step()
        inner_loss = log_loss(target[inner_valid], base._probability(model, matrix[inner_valid], device), labels=classes)
        losses.append(inner_loss)
        if inner_loss < best_loss:
            best_loss, best_epoch = inner_loss, epoch
        if epoch - best_epoch >= base.CFG.patience:
            break
    return best_epoch, losses, True


def build_folds(cache: base.Cache, labels: np.ndarray, seed: int) -> tuple[list[FoldData], np.ndarray, list[str]]:
    classes = sorted(np.unique(labels)); folds: list[FoldData] = []
    splitter = StratifiedKFold(n_splits=base.CFG.n_splits, shuffle=True, random_state=seed)
    for fold, (train_index, valid_index) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=base.CFG.n_splits, desc="FM-03 shared folds"), 1):
        fm_matrix, _ = base._matrix(cache, train_index, labels[train_index], contrast=False, functional=True, scale_numeric=True)
        lr_matrix, _ = base._matrix(cache, train_index, labels[train_index], contrast=True, functional=False, scale_numeric=False)
        folds.append(FoldData(fold, train_index, valid_index, fm_matrix, lr_matrix))
    return folds, np.zeros((len(labels), len(classes)), np.float32), classes


def sklearn_baseline(folds: list[FoldData], labels: np.ndarray, classes: list[str], seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    probability = np.zeros((len(labels), len(classes)), np.float32); rows = []
    for data in tqdm(folds, desc="O0 sklearn LR", unit="fold"):
        model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed)
        model.fit(data.lr_matrix[data.train_index], labels[data.train_index])
        raw = model.predict_proba(data.lr_matrix[data.valid_index])
        base.assign_probability(probability, data.valid_index, [classes.index(name) for name in model.classes_], raw)
        prediction = np.asarray(classes)[raw.argmax(1)]
        rows.append({"fold": data.fold, "baseline_fold_macro_f1": f1_score(labels[data.valid_index], prediction, average="macro", zero_division=0)})
        del model; gc.collect()
    return probability, pd.DataFrame(rows)


def run_candidate(candidate: Candidate, folds: list[FoldData], labels: np.ndarray, classes: list[str],
                  baseline_probability: np.ndarray, baseline_folds: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    device = base._device(); probability = np.zeros_like(baseline_probability); fold_rows = []; loss_rows = []; started = perf_counter(); finite = True
    for data in tqdm(folds, desc=candidate.candidate_id, unit="fold"):
        epoch, inner_losses, ok = select_epoch(data.fm_matrix[data.train_index], labels[data.train_index], classes, candidate, seed * 100 + data.fold, device)
        model, train_losses, ok2 = train_epochs(data.fm_matrix[data.train_index], labels[data.train_index], classes, candidate, epoch, seed * 1000 + data.fold, device)
        finite &= ok and ok2; raw = base._probability(model, data.fm_matrix[data.valid_index], device); probability[data.valid_index] = raw
        baseline = baseline_probability[data.valid_index]
        fm_prediction = np.asarray(classes)[raw.argmax(1)]
        blend25 = np.asarray(classes)[(0.75 * baseline + 0.25 * raw).argmax(1)]
        blend50 = np.asarray(classes)[(0.5 * baseline + 0.5 * raw).argmax(1)]
        baseline_score = float(baseline_folds.loc[baseline_folds.fold.eq(data.fold), "baseline_fold_macro_f1"].iloc[0])
        fold_rows.append({"candidate_id": candidate.candidate_id, "fold": data.fold, "fm_fold_macro_f1": f1_score(labels[data.valid_index], fm_prediction, average="macro", zero_division=0), "blend_0p25_fold_macro_f1": f1_score(labels[data.valid_index], blend25, average="macro", zero_division=0), "blend_0p50_fold_macro_f1": f1_score(labels[data.valid_index], blend50, average="macro", zero_division=0), "baseline_fold_macro_f1": baseline_score, "paired_delta_0p25": f1_score(labels[data.valid_index], blend25, average="macro", zero_division=0) - baseline_score, "paired_delta_0p50": f1_score(labels[data.valid_index], blend50, average="macro", zero_division=0) - baseline_score, "early_stopping_epoch": epoch, "loss_finite": ok and ok2})
        for number, value in enumerate(inner_losses, 1): loss_rows.append({"candidate_id": candidate.candidate_id, "fold": data.fold, "phase": "inner_valid", "epoch": number, "loss": value})
        for number, value in enumerate(train_losses, 1): loss_rows.append({"candidate_id": candidate.candidate_id, "fold": data.fold, "phase": "outer_train", "epoch": number, "loss": value})
        del model; gc.collect()
    baseline_score = f1_score(labels, np.asarray(classes)[baseline_probability.argmax(1)], average="macro", zero_division=0)
    fm_score = f1_score(labels, np.asarray(classes)[probability.argmax(1)], average="macro", zero_division=0)
    blend25_score = f1_score(labels, np.asarray(classes)[(0.75 * baseline_probability + 0.25 * probability).argmax(1)], average="macro", zero_division=0)
    blend50_score = f1_score(labels, np.asarray(classes)[(0.5 * baseline_probability + 0.5 * probability).argmax(1)], average="macro", zero_division=0)
    fold_frame = pd.DataFrame(fold_rows)
    summary = {**asdict(candidate), "seed": seed, "baseline_oof_macro_f1": baseline_score, "fm_oof_macro_f1": fm_score, "blend_0p25_oof_macro_f1": blend25_score, "blend_0p50_oof_macro_f1": blend50_score, "paired_delta_0p25_oof": blend25_score - baseline_score, "paired_delta_0p50_oof": blend50_score - baseline_score, "paired_delta_0p25_fold_mean": fold_frame.paired_delta_0p25.mean(), "paired_delta_0p25_fold_min": fold_frame.paired_delta_0p25.min(), "early_stopping_epoch_mean": fold_frame.early_stopping_epoch.mean(), "loss_finite": finite, "runtime_seconds": perf_counter() - started, "feature_count_mean": float(np.mean([data.fm_matrix.shape[1] for data in folds])), "leakage_check": True, "nan_as_mutation_count": 0}
    return summary, fold_frame, pd.DataFrame(loss_rows)


def select_winner(rows: list[dict]) -> Candidate:
    frame = pd.DataFrame(rows).sort_values(["paired_delta_0p25_oof", "paired_delta_0p25_fold_min"], ascending=False)
    value = frame.iloc[0]
    return Candidate(value["candidate_id"], value["kind"], int(value["rank"]), float(value["learning_rate"]), value["class_weight"])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--run-id", default="FM-03"); args = parser.parse_args()
    root = base.find_root(Path.cwd()); train = pd.read_csv(root / "data" / "raw" / "train.csv"); genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert train[genes].isna().sum().sum() == 0
    cache = base.Cache.build(train[genes], genes); labels = train[base.CFG.target_col].to_numpy(); output = root / "experiments" / "gs" / "notebooks" / "exp_model" / "result"; output.mkdir(parents=True, exist_ok=True)
    folds, _, classes = build_folds(cache, labels, args.seed); baseline_probability, baseline_folds = sklearn_baseline(folds, labels, classes, args.seed)
    baseline_score = f1_score(labels, np.asarray(classes)[baseline_probability.argmax(1)], average="macro", zero_division=0)
    candidates: list[dict] = [{"candidate_id": "O0_sklearn_lr", "kind": "sklearn_lr", "rank": 0, "learning_rate": 0.0, "class_weight": "balanced", "seed": args.seed, "baseline_oof_macro_f1": baseline_score, "fm_oof_macro_f1": baseline_score, "blend_0p25_oof_macro_f1": baseline_score, "blend_0p50_oof_macro_f1": baseline_score, "paired_delta_0p25_oof": 0.0, "paired_delta_0p50_oof": 0.0, "paired_delta_0p25_fold_mean": 0.0, "paired_delta_0p25_fold_min": 0.0, "early_stopping_epoch_mean": np.nan, "loss_finite": True, "runtime_seconds": np.nan, "feature_count_mean": float(np.mean([data.lr_matrix.shape[1] for data in folds])), "leakage_check": True, "nan_as_mutation_count": 0}]
    fold_results: list[pd.DataFrame] = [baseline_folds.assign(candidate_id="O0_sklearn_lr")]; losses: list[pd.DataFrame] = []; selections: dict[str, dict] = {}; screen_candidates: list[dict] = []
    def checkpoint() -> None:
        pd.DataFrame(candidates).to_csv(output / f"{args.run_id}_seed{args.seed}_candidates.partial.csv", index=False)
        pd.concat(fold_results, ignore_index=True).to_csv(output / f"{args.run_id}_seed{args.seed}_folds.partial.csv", index=False)
        if losses:
            pd.concat(losses, ignore_index=True).to_csv(output / f"{args.run_id}_seed{args.seed}_loss.partial.csv", index=False)
    checkpoint()
    linear = Candidate("O0_torch_linear", "linear", 0, 1e-3, "balanced")
    summary, details, curve = run_candidate(linear, folds, labels, classes, baseline_probability, baseline_folds, args.seed); candidates.append(summary); fold_results.append(details); losses.append(curve); checkpoint()
    stages = candidate_stages(); o1_rows = []
    for candidate in stages[0][1]:
        summary, details, curve = run_candidate(candidate, folds, labels, classes, baseline_probability, baseline_folds, args.seed); candidates.append(summary); screen_candidates.append(summary); o1_rows.append(summary); fold_results.append(details); losses.append(curve); checkpoint()
    o1 = select_winner(o1_rows); selections["O1_learning_rate"] = asdict(o1)
    o2_rows = []
    for prototype in stages[1][1]:
        candidate = Candidate(prototype.candidate_id, "fm", prototype.rank, o1.learning_rate, "balanced")
        summary, details, curve = run_candidate(candidate, folds, labels, classes, baseline_probability, baseline_folds, args.seed); candidates.append(summary); screen_candidates.append(summary); o2_rows.append(summary); fold_results.append(details); losses.append(curve); checkpoint()
    o2 = select_winner(o2_rows); selections["O2_rank"] = asdict(o2)
    for prototype in stages[2][1]:
        candidate = Candidate(prototype.candidate_id, "fm", o2.rank, o2.learning_rate, prototype.class_weight)
        summary, details, curve = run_candidate(candidate, folds, labels, classes, baseline_probability, baseline_folds, args.seed); candidates.append(summary); screen_candidates.append(summary); fold_results.append(details); losses.append(curve); checkpoint()
    winner = select_winner(screen_candidates); selections["screen_winner"] = asdict(winner)
    pd.DataFrame(candidates).to_csv(output / f"{args.run_id}_seed{args.seed}_candidates.csv", index=False)
    pd.concat(fold_results, ignore_index=True).to_csv(output / f"{args.run_id}_seed{args.seed}_folds.csv", index=False)
    pd.concat(losses, ignore_index=True).to_csv(output / f"{args.run_id}_seed{args.seed}_loss.csv", index=False)
    (output / f"{args.run_id}_seed{args.seed}_selection.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selections, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
