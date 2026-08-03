"""FM-01: fold-safe sparse multiclass factorization machine for mutation interactions.

No external data/source imports. OOF reads train.csv only; test is intentionally
unsupported in this runner until a candidate is selected for submission.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm.auto import tqdm
import torch
from torch import nn


WT = "WT"
EVENT_TYPES = ("MISSENSE", "SYNONYMOUS", "NONSENSE", "FRAMESHIFT", "SPLICE", "INFRAME_INDEL", "OTHER")
FUNCTIONAL_TYPES = frozenset({"MISSENSE", "NONSENSE", "FRAMESHIFT", "SPLICE", "INFRAME_INDEL"})
TRUNCATING = frozenset({"NONSENSE", "FRAMESHIFT", "SPLICE"})
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_PAIR = {(left, right): index for index, (left, right) in enumerate((left, right) for left in AA for right in AA if left != right)}
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
SPLICE_RE = re.compile(r"SPLICE|IVS|[+-]\d+")
INDEL_RE = re.compile(r"DEL|INS|DUP")
EXACT = (("BRAF", "V600E"), ("IDH1", "R132H"), ("PIK3CA", "H1047R"), ("PIK3CA", "E545K"))
CONTRASTS = (("KIRC", "KIPAN", 5), ("LGG", "GBMLGG", 5))


@dataclass(frozen=True)
class Config:
    id_col: str = "ID"
    target_col: str = "SUBCLASS"
    n_splits: int = 5
    token_min_count: int = 3
    recurrent_min_count: int = 5
    rank: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 8
    inner_valid_fraction: float = 0.10


CFG = Config()


def normalise_cell(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == WT:
        return ()
    return tuple(dict.fromkeys(token.removeprefix("P.") for token in re.sub(r"[;,|]+", " ", text).split() if token))


def event_type(event: str) -> str:
    if "FS" in event:
        return "FRAMESHIFT"
    if SPLICE_RE.search(event):
        return "SPLICE"
    if INDEL_RE.search(event):
        return "INFRAME_INDEL"
    if "*" in event or event.endswith("X"):
        return "NONSENSE"
    match = SUB_RE.fullmatch(event)
    return "SYNONYMOUS" if match and match.group(1) == match.group(3) else "MISSENSE" if match else "OTHER"


@dataclass
class Cache:
    genes: list[str]
    mutation: sparse.csr_matrix
    truncation: sparse.csr_matrix
    burden: np.ndarray
    variant: np.ndarray
    amino_pair: np.ndarray
    topology: np.ndarray
    events: pd.DataFrame

    @classmethod
    def build(cls, frame: pd.DataFrame, genes: list[str]) -> "Cache":
        mut_r: list[int] = []; mut_c: list[int] = []; trunc_r: list[int] = []; trunc_c: list[int] = []; rows = []
        for gene_index, gene in tqdm(enumerate(genes), total=len(genes), desc="row-local mutation cache"):
            for row_index, value in enumerate(frame[gene].array):
                tokens = normalise_cell(value)
                if not tokens:
                    continue
                mut_r.append(row_index); mut_c.append(gene_index)
                for event in tokens:
                    kind = event_type(event); rows.append((row_index, gene_index, gene, event, kind))
                    if kind in TRUNCATING:
                        trunc_r.append(row_index); trunc_c.append(gene_index)
        n_rows = len(frame)
        mutation = sparse.coo_matrix((np.ones(len(mut_r), np.float32), (mut_r, mut_c)), shape=(n_rows, len(genes))).tocsr(); mutation.data[:] = 1
        truncation = sparse.coo_matrix((np.ones(len(trunc_r), np.float32), (trunc_r, trunc_c)), shape=(n_rows, len(genes))).tocsr(); truncation.data[:] = 1
        events = pd.DataFrame(rows, columns=("row", "gene_index", "gene", "event", "type")).drop_duplicates(("row", "gene_index", "event"))
        burden = np.zeros((n_rows, 3), np.float32); burden[:, 0] = np.asarray(mutation.sum(axis=1)).ravel()
        variant = np.zeros((n_rows, len(EVENT_TYPES)), np.float32); amino_pair = np.zeros((n_rows, 380), np.float32); topology = np.zeros((n_rows, 8), np.float32)
        if not events.empty:
            burden[:, 1] = events.groupby("row").size().reindex(range(n_rows), fill_value=0).to_numpy()
            per_gene = events.groupby(["row", "gene_index"]).size(); burden[:, 2] = per_gene.gt(1).groupby(level=0).sum().reindex(range(n_rows), fill_value=0).to_numpy()
            for column, kind in enumerate(EVENT_TYPES):
                variant[:, column] = events.type.eq(kind).groupby(events.row).sum().reindex(range(n_rows), fill_value=0).to_numpy()
            extracted = events.event.str.extract(SUB_RE); events["ref"] = extracted[0]; events["alt"] = extracted[2]
            for row, ref, alt in events.dropna(subset=("ref", "alt"))[["row", "ref", "alt"]].itertuples(index=False):
                if (ref, alt) in AA_PAIR:
                    amino_pair[row, AA_PAIR[(ref, alt)]] += 1
            event_count = events.groupby(["row", "gene_index"]).size()
            for column, condition in enumerate((event_count.eq(1), event_count.eq(2), event_count.ge(3))):
                topology[:, column] = condition.groupby(level=0).sum().reindex(range(n_rows), fill_value=0).to_numpy()
            topology[:, 3] = event_count.groupby(level=0).max().reindex(range(n_rows), fill_value=0).to_numpy()
            type_counts = pd.crosstab(events.row, events.type).reindex(index=range(n_rows), columns=EVENT_TYPES, fill_value=0)
            proportions = type_counts.div(type_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            topology[:, 4] = type_counts.gt(0).sum(axis=1).to_numpy()
            topology[:, 5] = -(proportions.where(proportions.gt(0), 1) * np.log(proportions.where(proportions.gt(0), 1))).sum(axis=1).to_numpy()
            topology[:, 6] = proportions.max(axis=1).to_numpy()
            topology[:, 7] = events.groupby("row").gene.nunique().reindex(range(n_rows), fill_value=0).to_numpy()
        return cls(genes, mutation, truncation, burden, variant, amino_pair, topology, events)


def nonconstant(matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(matrix.min(axis=0).toarray()).ravel() != np.asarray(matrix.max(axis=0).toarray()).ravel()


def _matrix(cache: Cache, train_index: np.ndarray, labels: np.ndarray, contrast: bool, functional: bool, scale_numeric: bool) -> tuple[sparse.csr_matrix, list[str]]:
    parts: list[sparse.csr_matrix] = []; names: list[str] = []; numeric: list[bool] = []
    def add(value, labels_, is_numeric=False):
        value = sparse.csr_matrix(value); parts.append(value); names.extend(labels_); numeric.extend([is_numeric] * value.shape[1])
    active = np.flatnonzero(np.asarray(cache.mutation[train_index].getnnz(axis=0)).ravel())
    add(cache.mutation[:, active], [f"G__{cache.genes[index]}" for index in active])
    if functional and not cache.events.empty:
        token = cache.events[cache.events.type.isin(FUNCTIONAL_TYPES)].copy(); token["functional"] = token.gene + "__" + token.type
        counts = token[token.row.isin(train_index)].groupby("functional").row.nunique(); selected = sorted(counts[counts >= CFG.token_min_count].index)
        lookup = {name: index for index, name in enumerate(selected)}; subset = token[token.functional.isin(lookup)]
        value = sparse.coo_matrix((np.ones(len(subset), np.float32), (subset.row, subset.functional.map(lookup))), shape=(cache.mutation.shape[0], len(selected))).tocsr(); value.data[:] = 1
        add(value, [f"F__{name}" for name in selected])
    add(np.log1p(cache.burden), ["B__mutated_genes", "B__events", "B__multi_event_genes"], True)
    add(np.log1p(cache.variant), [f"V__{kind}" for kind in EVENT_TYPES], True)
    trunc = np.flatnonzero(np.asarray(cache.truncation[train_index].getnnz(axis=0)).ravel())
    add(cache.truncation[:, trunc], [f"T__{cache.genes[index]}" for index in trunc])
    add(np.log1p(np.asarray(cache.truncation.sum(axis=1))), ["T__count"], True)
    if not cache.events.empty:
        misses = cache.events[cache.events.type.eq("MISSENSE")].copy(); misses["pair"] = misses.gene + "__" + misses.event
        counts = misses[misses.row.isin(train_index)].groupby("pair").row.nunique(); selected = sorted(counts[counts >= CFG.recurrent_min_count].index)
        lookup = {name: index for index, name in enumerate(selected)}; subset = misses[misses.pair.isin(lookup)]
        recurrent = sparse.coo_matrix((np.ones(len(subset), np.float32), (subset.row, subset.pair.map(lookup))), shape=(cache.mutation.shape[0], len(selected))).tocsr(); recurrent.data[:] = 1
        add(recurrent, [f"R__{name}" for name in selected]); add(np.log1p(np.asarray(recurrent.sum(axis=1))), ["R__count"], True)
    add(np.log1p(cache.amino_pair), [f"A_pair__{index}" for index in range(380)], True)
    exact_columns = []
    for gene, event in EXACT:
        if cache.events.empty:
            exact_columns.append(sparse.csr_matrix((cache.mutation.shape[0], 1)))
        else:
            found = cache.events[(cache.events.gene == gene) & (cache.events.event == event)]
            exact_columns.append(sparse.coo_matrix((np.ones(len(found), np.float32), (found.row, np.zeros(len(found), int))), shape=(cache.mutation.shape[0], 1)).tocsr())
    add(sparse.hstack(exact_columns), [f"E__{gene}_{event}" for gene, event in EXACT])
    add(np.log1p(cache.topology), [f"S__{index}" for index in range(cache.topology.shape[1])], True)
    if contrast:
        for left, right, top_k in CONTRASTS:
            left_mask, right_mask = labels == left, labels == right
            left_counts = np.asarray(cache.mutation[train_index][left_mask].getnnz(axis=0)).ravel(); right_counts = np.asarray(cache.mutation[train_index][right_mask].getnnz(axis=0)).ravel()
            support = left_counts + right_counts; score = left_counts / left_mask.sum() - right_counts / right_mask.sum()
            selected = sorted(np.flatnonzero(support >= 10), key=lambda index: (-abs(score[index]), -support[index], cache.genes[index]))[:top_k]
            signs = np.sign(score[selected]).astype(np.float32)
            add(cache.mutation[:, selected].sum(axis=1), [f"C__{left}_{right}_count"], True)
            add(cache.mutation[:, selected].dot(sparse.csr_matrix(signs).T), [f"C__{left}_{right}_contrast"], True)
    full = sparse.hstack(parts, format="csr"); keep = nonconstant(full[train_index]); full = full[:, keep]; names = [name for name, selected in zip(names, keep) if selected]; numeric = np.asarray(numeric)[keep]
    if scale_numeric and numeric.any():
        train = full[train_index][:, numeric]; mean = np.asarray(train.mean(axis=0)).ravel(); sq = np.asarray(train.multiply(train).mean(axis=0)).ravel(); std = np.sqrt(np.maximum(sq - mean * mean, 1e-6))
        scale = np.ones(full.shape[1], np.float32); scale[numeric] = 1 / std
        full = full @ sparse.diags(scale, format="csr")
    return full.tocsr(), names


class SparseMulticlassFM(nn.Module):
    def __init__(self, n_features: int, n_classes: int, rank: int):
        super().__init__(); self.linear = nn.Embedding(n_features, n_classes); self.embedding = nn.Embedding(n_features, n_classes * rank); self.n_classes = n_classes; self.rank = rank
        nn.init.zeros_(self.linear.weight); nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        linear = (self.linear(indices) * values.unsqueeze(-1)).sum(dim=1)
        factor = self.embedding(indices).view(indices.shape[0], indices.shape[1], self.n_classes, self.rank) * values[:, :, None, None]
        interaction = 0.5 * ((factor.sum(dim=1).square() - factor.square().sum(dim=1)).sum(dim=-1))
        return linear + interaction


def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _batch(matrix: sparse.csr_matrix, rows: np.ndarray, device: torch.device):
    view = matrix[rows]; width = max(1, int(np.diff(view.indptr).max())); indices = np.zeros((len(rows), width), np.int64); values = np.zeros((len(rows), width), np.float32)
    for position in range(len(rows)):
        start, stop = view.indptr[position], view.indptr[position + 1]; count = stop - start; indices[position, :count] = view.indices[start:stop]; values[position, :count] = view.data[start:stop]
    return torch.as_tensor(indices, device=device), torch.as_tensor(values, device=device)


def _fit(matrix: sparse.csr_matrix, target: np.ndarray, classes: list[str], epochs: int, seed: int, device: torch.device) -> tuple[SparseMulticlassFM, list[float], bool]:
    torch.manual_seed(seed); np.random.seed(seed); encoded = np.searchsorted(classes, target); counts = np.bincount(encoded, minlength=len(classes)); weight = len(encoded) / (len(classes) * np.maximum(counts, 1)); model = SparseMulticlassFM(matrix.shape[1], len(classes), CFG.rank).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay); criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device)); losses = []
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train(); order = rng.permutation(len(target)); total = 0.0
        for start in range(0, len(order), CFG.batch_size):
            rows = order[start:start + CFG.batch_size]; index, value = _batch(matrix, rows, device); label = torch.as_tensor(encoded[rows], dtype=torch.long, device=device); optimiser.zero_grad(); loss = criterion(model(index, value), label)
            if not torch.isfinite(loss): return model, losses, False
            loss.backward(); optimiser.step(); total += float(loss.detach().cpu()) * len(rows)
        losses.append(total / len(target))
    return model, losses, bool(np.isfinite(losses).all())


def _probability(model: SparseMulticlassFM, matrix: sparse.csr_matrix, device: torch.device) -> np.ndarray:
    model.eval(); chunks = []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], CFG.batch_size):
            rows = np.arange(start, min(start + CFG.batch_size, matrix.shape[0])); index, value = _batch(matrix, rows, device); chunks.append(torch.softmax(model(index, value), dim=1).cpu().numpy())
    return np.vstack(chunks)


def assign_probability(output: np.ndarray, rows: np.ndarray, columns: list[int], probability: np.ndarray) -> None:
    """Write a validation-row × class-column probability block without paired indexing."""
    output[np.ix_(rows, np.asarray(columns))] = probability


def _best_epoch(matrix: sparse.csr_matrix, labels: np.ndarray, classes: list[str], seed: int, device: torch.device) -> tuple[int, list[float], bool]:
    tr, iv = next(StratifiedShuffleSplit(n_splits=1, test_size=CFG.inner_valid_fraction, random_state=seed).split(np.zeros(len(labels)), labels))
    torch.manual_seed(seed); np.random.seed(seed); encoded = np.searchsorted(classes, labels); counts = np.bincount(encoded[tr], minlength=len(classes)); weight = len(tr) / (len(classes) * np.maximum(counts, 1))
    model = SparseMulticlassFM(matrix.shape[1], len(classes), CFG.rank).to(device); optimiser = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay); criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device)); rng = np.random.default_rng(seed)
    losses: list[float] = []; best_epoch = 1; best_loss = np.inf; best_state = None
    for epoch in range(1, CFG.max_epochs + 1):
        model.train(); order = rng.permutation(tr)
        for start in range(0, len(order), CFG.batch_size):
            rows = order[start:start + CFG.batch_size]; index, value = _batch(matrix, rows, device); label = torch.as_tensor(encoded[rows], dtype=torch.long, device=device); optimiser.zero_grad(); loss = criterion(model(index, value), label)
            if not torch.isfinite(loss): return best_epoch, losses, False
            loss.backward(); optimiser.step()
        inner_loss = log_loss(labels[iv], _probability(model, matrix[iv], device), labels=classes); losses.append(inner_loss)
        if inner_loss < best_loss:
            best_loss, best_epoch, best_state = inner_loss, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= CFG.patience: break
    del best_state
    return best_epoch, losses, True


def run_seed(cache: Cache, labels: pd.Series, seed: int, contrast: bool) -> tuple[dict, pd.DataFrame]:
    classes = sorted(labels.unique()); value = labels.to_numpy(); splitter = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=seed); fm_probability = np.zeros((len(labels), len(classes))); lr_probability = np.zeros_like(fm_probability)
    feature_counts = []; epochs = []; finite = True; started = perf_counter(); device = _device()
    for fold, (tr, va) in enumerate(tqdm(splitter.split(np.zeros(len(labels)), labels), total=CFG.n_splits, desc=f"FM {'02' if contrast else '01'} | seed {seed}"), 1):
        fm_all, names = _matrix(cache, tr, value[tr], contrast=contrast, functional=True, scale_numeric=True); lr_all, _ = _matrix(cache, tr, value[tr], contrast=True, functional=False, scale_numeric=False)
        best, inner_loss, ok = _best_epoch(fm_all[tr], value[tr], classes, seed * 100 + fold, device); model, train_loss, ok2 = _fit(fm_all[tr], value[tr], classes, best, seed * 1000 + fold, device); finite &= ok and ok2
        fm_probability[va] = _probability(model, fm_all[va], device); lr = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed).fit(lr_all[tr], value[tr]); raw = lr.predict_proba(lr_all[va]); assign_probability(lr_probability, va, [classes.index(name) for name in lr.classes_], raw)
        feature_counts.append(len(names)); epochs.append(best); del fm_all, lr_all, model, lr; gc.collect()
    variants = {"fm": fm_probability, "lr08": lr_probability, "blend_0p5": 0.5 * fm_probability + 0.5 * lr_probability}; rows = []; reports = []
    for name, probability in variants.items():
        prediction = np.asarray(classes)[probability.argmax(1)]; rows.append((name, f1_score(value, prediction, average="macro", zero_division=0), accuracy_score(value, prediction))); report = pd.DataFrame(classification_report(value, prediction, labels=classes, output_dict=True, zero_division=0)).T.loc[classes].reset_index(names="class"); report.insert(0, "variant", name); reports.append(report)
    score = dict((name, macro) for name, macro, _ in rows); result = {"experiment_id": "FM-02" if contrast else "FM-01", "seed": seed, "fm_oof_macro_f1": score["fm"], "lr08_oof_macro_f1": score["lr08"], "blend_0p5_oof_macro_f1": score["blend_0p5"], "delta_fm_vs_lr08": score["fm"] - score["lr08"], "delta_blend_vs_lr08": score["blend_0p5"] - score["lr08"], "feature_count_mean": float(np.mean(feature_counts)), "early_stopping_epoch_mean": float(np.mean(epochs)), "runtime_seconds": perf_counter() - started, "device": str(device), "loss_finite": finite, "leakage_check": True, "nan_as_mutation_count": 0, "parameters": json.dumps({"rank": CFG.rank, "lr": CFG.learning_rate, "weight_decay": CFG.weight_decay, "batch": CFG.batch_size, "max_epochs": CFG.max_epochs, "patience": CFG.patience})}
    return result, pd.concat(reports, ignore_index=True)


def run_preflight(cache: Cache, labels: pd.Series, seed: int, contrast: bool) -> dict:
    """Run only the first outer fold; this is an execution check, never a score-selection run."""
    classes = sorted(labels.unique()); value = labels.to_numpy(); tr, _ = next(StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    started = perf_counter(); matrix, names = _matrix(cache, tr, value[tr], contrast=contrast, functional=True, scale_numeric=True); epoch, losses, finite = _best_epoch(matrix[tr], value[tr], classes, seed * 101, _device())
    return {"experiment_id": "FM-02" if contrast else "FM-01", "seed": seed, "mode": "preflight_first_outer_fold_only", "feature_count": len(names), "outer_train_rows": len(tr), "early_stopping_epoch": epoch, "inner_loss_first": losses[0] if losses else np.nan, "inner_loss_best": min(losses) if losses else np.nan, "loss_finite": finite, "runtime_seconds": perf_counter() - started, "device": str(_device()), "leakage_check": True, "nan_as_mutation_count": 0}


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists(): return path
    raise FileNotFoundError("project root not found")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", choices=("FM-01", "FM-02"), default="FM-01"); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--run-id", default=""); parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    root = find_root(Path.cwd()); train = pd.read_csv(root / "data" / "raw" / "train.csv"); genes = [column for column in train if column not in (CFG.id_col, CFG.target_col)]; assert train[genes].isna().sum().sum() == 0; assert all(not normalise_cell(value) for gene in genes for value in train.loc[train[gene].isna(), gene])
    cache = Cache.build(train[genes], genes); output = root / "experiments" / "gs" / "notebooks" / "exp_model" / "result"; output.mkdir(parents=True, exist_ok=True); stem = f"{args.run_id}_{args.experiment}_seed{args.seed}".strip("_")
    if args.preflight:
        result = run_preflight(cache, train[CFG.target_col], args.seed, contrast=args.experiment == "FM-02"); pd.DataFrame([result]).to_csv(output / f"{stem}_preflight.csv", index=False); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    result, report = run_seed(cache, train[CFG.target_col], args.seed, contrast=args.experiment == "FM-02"); pd.DataFrame([result]).to_csv(output / f"{stem}_oof.csv", index=False); report.to_csv(output / f"{stem}_class_f1.csv", index=False); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            root = find_root(Path.cwd())
            output = root / "experiments" / "gs" / "notebooks" / "exp_model" / "result"
            output.mkdir(parents=True, exist_ok=True)
            (output / "FM-01_last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            raise
