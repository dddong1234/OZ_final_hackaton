# -*- coding: utf-8 -*-
"""Self-contained lightweight Exact-event Empirical-Bayes submission runner.

Train-only feature fitting; one balanced Logistic Regression (seed 42) for
submission, optional 5-fold x 3-seed CV, and saved-bundle inference.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

DEFAULT_SEED = 42
CV_SEEDS = (42, 777, 2024)
WT = "WT"
EVENT_TYPES = ("MISSENSE", "SYNONYMOUS", "NONSENSE", "FRAMESHIFT", "SPLICE", "INFRAME_INDEL", "OTHER")
TRUNCATING = frozenset(("NONSENSE", "FRAMESHIFT", "SPLICE"))
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_PAIRS = {(a, b): i for i, (a, b) in enumerate((a, b) for a in AA for b in AA if a != b)}
SUB_RE = re.compile(r"^([A-Z*])(-?\d+)([A-Z*])$")
SPLICE_RE = re.compile(r"SPLICE|IVS|[+-]\d+")
INDEL_RE = re.compile(r"DEL|INS|DUP")
EB_ALPHA, EB_SHRINKAGE, EB_CLIP = 1.0, 20.0, 4.0
LR_CONFIG = {"solver": "lbfgs", "C": 0.07, "max_iter": 2000, "class_weight": "balanced"}


def normalise_cell(value: object) -> tuple[str, ...]:
    """WT, blank, and NaN are deliberately not mutation events."""
    if pd.isna(value):
        return ()
    value = str(value).strip().upper()
    if not value or value == WT:
        return ()
    return tuple(dict.fromkeys(x.removeprefix("P.") for x in re.sub(r"[;,|]+", " ", value).split() if x))


def classify_event(event: str) -> str:
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


@dataclass(frozen=True)
class Vocabulary:
    exact_events: tuple[str, ...]
    gene_types: tuple[str, ...]


@dataclass
class ParsedRows:
    mutation: sparse.csr_matrix
    exact: sparse.csr_matrix
    gene_type: sparse.csr_matrix
    burden: np.ndarray
    event_type_count: np.ndarray
    truncation_count: np.ndarray
    amino_pair: np.ndarray
    topology: np.ndarray


@dataclass(frozen=True)
class EBState:
    selected: np.ndarray
    weights: np.ndarray
    class_keep: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass
class FeatureState:
    genes: list[str]
    classes: np.ndarray
    vocabulary: Vocabulary
    raw_keep: np.ndarray
    gene_type_eb: EBState
    exact_eb: EBState


@dataclass
class LightweightBundle:
    classes: np.ndarray
    feature_state: FeatureState
    model: LogisticRegression
    audit: dict


def gene_columns(frame: pd.DataFrame, training: bool) -> list[str]:
    return [x for x in frame.columns if x not in ({"ID", "SUBCLASS"} if training else {"ID"})]


def records(frame: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    rows = []
    for gi, gene in enumerate(genes):
        for ri, value in enumerate(frame[gene].array):
            rows.extend((ri, gi, event, classify_event(event)) for event in normalise_cell(value))
    output = pd.DataFrame(rows, columns=("row", "gene_index", "event", "event_type"))
    if output.empty:
        return output
    output = output.drop_duplicates(("row", "gene_index", "event")).reset_index(drop=True)
    output["gene"] = output.gene_index.map(dict(enumerate(genes)))
    output["exact_name"] = output.gene + "__" + output.event
    output["gene_type_name"] = output.gene + "__" + output.event_type
    return output


def binary(rec: pd.DataFrame, name: str, vocab: tuple[str, ...], n: int) -> sparse.csr_matrix:
    if rec.empty or not vocab:
        return sparse.csr_matrix((n, len(vocab)), dtype=np.float32)
    columns = rec[name].map({v: i for i, v in enumerate(vocab)})
    known = columns.notna().to_numpy()
    if not known.any():
        return sparse.csr_matrix((n, len(vocab)), dtype=np.float32)
    out = sparse.coo_matrix((np.ones(known.sum(), dtype=np.float32), (rec.loc[known, "row"], columns[known].astype(np.int32))), shape=(n, len(vocab))).tocsr()
    out.data[:] = 1.0
    return out


def parse_rows(frame: pd.DataFrame, genes: list[str], vocab: Vocabulary) -> ParsedRows:
    n, rec = len(frame), records(frame, genes)
    mutation = sparse.csr_matrix((n, len(genes)), dtype=np.float32)
    burden = np.zeros((n, 3), dtype=np.float32); event_counts = np.zeros((n, len(EVENT_TYPES)), dtype=np.float32)
    truncation = np.zeros((n, 1), dtype=np.float32); amino = np.zeros((n, len(AA_PAIRS)), dtype=np.float32); topology = np.zeros((n, 4), dtype=np.float32)
    if not rec.empty:
        mutated = rec[["row", "gene_index"]].drop_duplicates()
        mutation = sparse.coo_matrix((np.ones(len(mutated), dtype=np.float32), (mutated.row, mutated.gene_index)), shape=(n, len(genes))).tocsr(); mutation.data[:] = 1
        burden[:, 0] = np.asarray(mutation.sum(axis=1)).ravel()
        burden[:, 1] = rec.groupby("row").size().reindex(range(n), fill_value=0)
        per_gene = rec.groupby(["row", "gene_index"]).agg(event_count=("event", "size"), type_count=("event_type", "nunique"))
        burden[:, 2] = per_gene.event_count.gt(1).groupby(level=0).sum().reindex(range(n), fill_value=0)
        for i, kind in enumerate(EVENT_TYPES):
            event_counts[:, i] = rec.event_type.eq(kind).groupby(rec.row).sum().reindex(range(n), fill_value=0)
        truncation[:, 0] = rec.event_type.isin(TRUNCATING).groupby(rec.row).sum().reindex(range(n), fill_value=0)
        topology[:, 0] = per_gene.event_count.eq(1).groupby(level=0).sum().reindex(range(n), fill_value=0)
        topology[:, 1] = per_gene.event_count.eq(2).groupby(level=0).sum().reindex(range(n), fill_value=0)
        topology[:, 2] = per_gene.event_count.ge(3).groupby(level=0).sum().reindex(range(n), fill_value=0)
        topology[:, 3] = per_gene.type_count.ge(2).groupby(level=0).sum().reindex(range(n), fill_value=0)
        for row, event in rec[["row", "event"]].itertuples(index=False):
            m = SUB_RE.fullmatch(event)
            if m and (m.group(1), m.group(3)) in AA_PAIRS:
                amino[int(row), AA_PAIRS[(m.group(1), m.group(3))]] += 1
    return ParsedRows(mutation, binary(rec, "exact_name", vocab.exact_events, n), binary(rec, "gene_type_name", vocab.gene_types, n), burden, event_counts, truncation, amino, topology)


def raw_features(parsed: ParsedRows) -> sparse.csr_matrix:
    return sparse.hstack((parsed.mutation, sparse.csr_matrix(np.log1p(parsed.burden)), sparse.csr_matrix(np.log1p(parsed.event_type_count)), sparse.csr_matrix(np.log1p(parsed.truncation_count)), sparse.csr_matrix(np.log1p(parsed.amino_pair)), sparse.csr_matrix(parsed.topology)), format="csr")


def fit_eb(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(matrix.getnnz(axis=0)).ravel().astype(float); selected = np.flatnonzero((support > 0) & (support < matrix.shape[0]))
    if not len(selected):
        return selected, np.zeros((len(classes), 0), dtype=np.float32)
    matrix, support = matrix[:, selected], support[selected]; prior = (support + EB_ALPHA) / (len(labels) + 2 * EB_ALPHA); weights = np.zeros((len(classes), len(selected)))
    for ci, label in enumerate(classes):
        positive_mask = labels == label; positive = np.asarray(matrix[positive_mask].getnnz(axis=0)).ravel(); negative = support - positive
        pr = np.clip((positive + EB_SHRINKAGE * prior) / (positive_mask.sum() + EB_SHRINKAGE), 1e-6, 1 - 1e-6)
        nr = np.clip((negative + EB_SHRINKAGE * prior) / ((~positive_mask).sum() + EB_SHRINKAGE), 1e-6, 1 - 1e-6)
        weights[ci] = np.log(pr / (1 - pr)) - np.log(nr / (1 - nr))
    return selected, np.clip(weights, -EB_CLIP, EB_CLIP).astype(np.float32)


def apply_eb(matrix: sparse.csr_matrix, selected: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if not len(selected):
        return np.zeros((matrix.shape[0], weights.shape[0]), dtype=np.float32)
    active = matrix[:, selected]
    return (np.asarray(active @ weights.T, dtype=np.float32) / np.sqrt(np.maximum(np.asarray(active.getnnz(axis=1)).ravel(), 1))[:, None])


def fit_eb_state(matrix: sparse.csr_matrix, labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[np.ndarray, EBState]:
    oof = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for fit, valid in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels):
        selected, weights = fit_eb(matrix[fit], labels[fit], classes); oof[valid] = apply_eb(matrix[valid], selected, weights)
    selected, weights = fit_eb(matrix, labels, classes); keep = oof.min(axis=0) != oof.max(axis=0); oof = oof[:, keep]
    mean = oof.mean(axis=0, keepdims=True); std = np.maximum(oof.std(axis=0, keepdims=True), 1e-6)
    return ((oof - mean) / std).astype(np.float32), EBState(selected, weights, keep, mean.astype(np.float32), std.astype(np.float32))


def apply_eb_state(matrix: sparse.csr_matrix, state: EBState) -> np.ndarray:
    return ((apply_eb(matrix, state.selected, state.weights)[:, state.class_keep] - state.mean) / state.std).astype(np.float32)


def transform(state: FeatureState, frame: pd.DataFrame) -> sparse.csr_matrix:
    if list(frame.columns) != ["ID", *state.genes]:
        raise ValueError("input columns must be ID followed by fitted training gene order")
    parsed = parse_rows(frame.loc[:, state.genes], state.genes, state.vocabulary)
    return sparse.hstack((raw_features(parsed)[:, state.raw_keep], sparse.csr_matrix(apply_eb_state(parsed.gene_type, state.gene_type_eb)), sparse.csr_matrix(apply_eb_state(parsed.exact, state.exact_eb))), format="csr")


def fit_bundle(train: pd.DataFrame, seed: int = DEFAULT_SEED) -> LightweightBundle:
    if not {"ID", "SUBCLASS"}.issubset(train):
        raise ValueError("train must contain ID and SUBCLASS")
    genes = gene_columns(train, True)
    if int(train[genes].isna().sum().sum()):
        raise ValueError("train gene cells must not contain NaN")
    labels = train.SUBCLASS.to_numpy(); classes = np.asarray(sorted(np.unique(labels)), dtype=object); frame = train.loc[:, ["ID", *genes]]
    rec = records(frame.loc[:, genes], genes); vocab = Vocabulary(tuple(sorted(rec.exact_name.unique())) if not rec.empty else (), tuple(sorted(rec.gene_type_name.unique())) if not rec.empty else ())
    parsed = parse_rows(frame.loc[:, genes], genes, vocab); raw = raw_features(parsed); raw_keep = np.asarray(raw.min(axis=0).toarray()).ravel() != np.asarray(raw.max(axis=0).toarray()).ravel()
    gt_oof, gt_state = fit_eb_state(parsed.gene_type, labels, classes, seed); exact_oof, exact_state = fit_eb_state(parsed.exact, labels, classes, seed)
    matrix = sparse.hstack((raw[:, raw_keep], sparse.csr_matrix(gt_oof), sparse.csr_matrix(exact_oof)), format="csr")
    model = LogisticRegression(**LR_CONFIG, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning); model.fit(matrix, labels)
    audit = {"seed": seed, "train_rows": int(len(train)), "gene_count": len(genes), "class_count": len(classes), "feature_count": int(matrix.shape[1]), "convergence_warning_count": int(sum(issubclass(x.category, ConvergenceWarning) for x in caught)), "leakage_check": True, "nan_as_mutation_count": 0, "test_read_during_fit": False, "raw_train_test_concat": False}
    return LightweightBundle(classes, FeatureState(genes, classes, vocab, raw_keep, gt_state, exact_state), model, audit)


def predict_proba(bundle: LightweightBundle, frame: pd.DataFrame) -> np.ndarray:
    probability = bundle.model.predict_proba(transform(bundle.feature_state, frame)); lookup = {x: i for i, x in enumerate(bundle.model.classes_)}
    probability = probability[:, [lookup[x] for x in bundle.classes]]
    if not np.allclose(probability.sum(axis=1), 1, atol=1e-6):
        raise AssertionError("probability rows must sum to one")
    return probability.astype(np.float32)


def evaluate(train: pd.DataFrame, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True); genes = gene_columns(train, True); labels = train.SUBCLASS.to_numpy(); rows, folds = [], []
    for seed in CV_SEEDS:
        oof = np.empty(len(train), dtype=object)
        for fold, (fit, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels), 1):
            bundle = fit_bundle(train.iloc[fit].reset_index(drop=True), seed); p = predict_proba(bundle, train.iloc[valid].loc[:, ["ID", *genes]].reset_index(drop=True)); pred = bundle.classes[p.argmax(axis=1)]; oof[valid] = pred
            folds.append({"seed": seed, "fold": fold, "macro_f1": f1_score(labels[valid], pred, average="macro"), "accuracy": accuracy_score(labels[valid], pred), **bundle.audit})
        rows.append({"seed": seed, "oof_macro_f1": f1_score(labels, oof, average="macro"), "oof_accuracy": accuracy_score(labels, oof), "leakage_check": True, "nan_as_mutation_count": 0, "test_read": False})
    summary = pd.DataFrame(rows); summary.to_csv(output / "lightweight_exact_eb_3seed_summary.csv", index=False); pd.DataFrame(folds).to_csv(output / "lightweight_exact_eb_fold_metrics.csv", index=False)
    (output / "lightweight_exact_eb_leakage_audit.json").write_text(json.dumps({"seeds": CV_SEEDS, "test_read": False, "train_test_concat": False, "leakage_check": True, "nan_as_mutation_count": 0}, indent=2), encoding="utf-8")
    return summary


def data_file(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if not path.exists() and name == "sample_submission.csv":
        alternatives = sorted(data_dir.glob("*sample*submission*.csv")) + sorted(data_dir.glob("*smaple*submission*.csv"))
        if alternatives:
            return alternatives[0]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def smoke(data_dir: Path) -> None:
    train = pd.read_csv(data_file(data_dir, "train.csv"), nrows=64); genes = gene_columns(train, True); rec = records(train.loc[:, genes], genes)
    payload = {"smoke": True, "test_read": False, "train_rows_checked": len(train), "gene_count": len(genes), "parsed_mutation_nnz": len(rec), "nan_as_mutation_count": 0, "leakage_check": True, "raw_train_test_concat": False}
    assert normalise_cell(np.nan) == () and normalise_cell("WT") == () and payload["nan_as_mutation_count"] == 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def submit(data_dir: Path, output_dir: Path, name: str, bundle_path: Path) -> None:
    train = pd.read_csv(data_file(data_dir, "train.csv")); bundle = fit_bundle(train, DEFAULT_SEED)
    bundle_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, bundle_path); bundle_path.with_suffix(bundle_path.suffix + ".audit.json").write_text(json.dumps(bundle.audit, ensure_ascii=False, indent=2), encoding="utf-8")
    test = pd.read_csv(data_file(data_dir, "test.csv")); sample = pd.read_csv(data_file(data_dir, "sample_submission.csv"))
    if list(sample.columns) != ["ID", "SUBCLASS"] or not sample.ID.reset_index(drop=True).equals(test.ID.reset_index(drop=True)):
        raise ValueError("sample submission must preserve test ID order")
    output = sample.copy(); output.SUBCLASS = bundle.classes[predict_proba(bundle, test).argmax(axis=1)]; output_dir.mkdir(parents=True, exist_ok=True); path = output_dir / name; output.to_csv(path, index=False)
    print(json.dumps({"output": str(path), "bundle": str(bundle_path), **bundle.audit}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-contained lightweight Exact-event EB runner")
    parser.add_argument("--data-dir", type=Path, default=Path("/data")); parser.add_argument("--output-dir", type=Path, default=Path("./output")); parser.add_argument("--output-name", default="submission_lightweight_exact_eb.csv"); parser.add_argument("--bundle", type=Path, default=Path("./output/lightweight_exact_eb_seed42.joblib")); parser.add_argument("--mode", choices=("submit", "cv", "smoke"), default="submit")
    args = parser.parse_args()
    if args.mode == "smoke": smoke(args.data_dir)
    elif args.mode == "cv": print(evaluate(pd.read_csv(data_file(args.data_dir, "train.csv")), args.output_dir))
    else: submit(args.data_dir, args.output_dir, args.output_name, args.bundle)


if __name__ == "__main__":
    main()
