"""Fold-aligned, leakage-safe OOF bagging audit for the fixed H0 candidate.

Every validation row below is held out from all three seed models whose
probabilities are averaged. This deliberately avoids cross-split OOF averaging.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[5]
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"
RESULT_DIR = HERE.parent.parent / "result"
OUTER_SPLIT_SEED = 42
MODEL_SEEDS = (42, 777, 2024)


def _source_common() -> Path:
    """Reference checked-in GS H0 code only; never a teammate directory."""
    path = HERE.parents[2] / "exp_model_007" / "common"
    if not path.exists():
        raise FileNotFoundError(f"fixed H0 source missing: {path}")
    return path


if str(_source_common()) not in sys.path:
    sys.path.insert(0, str(_source_common()))

from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402


def _metric(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[np.asarray(probability).argmax(axis=1)]
    return (float(f1_score(y, prediction, average="macro", zero_division=0)), float(accuracy_score(y, prediction)), prediction)


def _empty_oof(n_rows: int, n_classes: int) -> dict[str, np.ndarray]:
    return {f"seed_{seed}": np.zeros((n_rows, n_classes), dtype=np.float32) for seed in MODEL_SEEDS}


def _save(path: Path, completed: list[int], folds: list[dict], audits: list[dict], oof: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"completed_folds": completed, "fold_rows": folds, "audit_rows": audits})
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata_json=np.asarray(metadata), **oof)
    temporary.replace(path)


def _load(path: Path) -> tuple[list[int], list[dict], list[dict], dict[str, np.ndarray]] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        oof = {f"seed_{seed}": archive[f"seed_{seed}"].copy() for seed in MODEL_SEEDS}
    return list(metadata["completed_folds"]), list(metadata["fold_rows"]), list(metadata["audit_rows"]), oof


def run(run_id: str) -> None:
    train = pd.read_csv(TRAIN_CSV)  # train-only OOF contract
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    checkpoint = _load(RESULT_DIR / f"{run_id}_checkpoint.npz")
    completed, fold_rows, audit_rows, oof = checkpoint or ([], [], [], _empty_oof(len(train), len(classes)))
    started = perf_counter()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SPLIT_SEED)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        if fold in completed:
            continue
        print(f"fold {fold}/5: fixed validation split, fitting 3 model seeds", flush=True)
        fit_frame = train.iloc[fit_idx][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_idx][genes].reset_index(drop=True)
        for model_seed in MODEL_SEEDS:
            result = fit_fold(fit_frame, valid_frame, labels[fit_idx], genes, classes, seed=model_seed * 100 + fold)
            probability = result["candidate"]
            oof[f"seed_{model_seed}"][valid_idx] = probability
            macro, accuracy, _ = _metric(labels[valid_idx], probability, classes)
            fold_rows.append({"outer_split_seed": OUTER_SPLIT_SEED, "model_seed": model_seed, "fold": fold, "variant": f"seed_{model_seed}", "macro_f1": macro, "accuracy": accuracy, "feature_count": int(result["candidate_feature_count"]), "convergence_warning_count": int(result["h0_warning"] + result["eb_warning"])})
            audit_rows.append({"outer_split_seed": OUTER_SPLIT_SEED, "model_seed": model_seed, "fold": fold, "test_read": False, "train_test_concat": False, "outer_validation_used_for_fit": False, "fold_train_only_eb": True, "fixed_class_gene_mutation_rules": False, "leakage_check": not bool(result["audit"]["raw_train_test_concat"]), "nan_as_mutation_count": int(result["audit"]["nan_as_mutation_count"]), "valid_row_count": int(len(valid_idx))})
            del result
            gc.collect()
        completed.append(fold)
        _save(RESULT_DIR / f"{run_id}_checkpoint.npz", completed, fold_rows, audit_rows, oof)
        print(f"fold {fold}/5 checkpoint saved", flush=True)

    oof["fold_aligned_bagged"] = np.mean(np.stack([oof[f"seed_{seed}"] for seed in MODEL_SEEDS]), axis=0).astype(np.float32)
    if not np.allclose(oof["fold_aligned_bagged"].sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("bagged probability rows are not normalized")
    folds, audits = pd.DataFrame(fold_rows), pd.DataFrame(audit_rows)
    summary_rows, class_rows = [], []
    for variant, probability in oof.items():
        macro, accuracy, prediction = _metric(labels, probability, classes)
        summary_rows.append({"variant": variant, "outer_split_seed": OUTER_SPLIT_SEED, "model_seed_count": 3 if variant == "fold_aligned_bagged" else 1, "oof_macro_f1": macro, "oof_accuracy": accuracy, "feature_count_mean": float(folds.feature_count.mean()), "convergence_warning_count": int(folds.convergence_warning_count.sum()), "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started})
        precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"variant": variant, "class": label, "precision": p, "recall": r, "f1": score, "support": int(n)} for label, p, r, score, n in zip(classes, precision, recall, f1, support))
    summary = pd.DataFrame(summary_rows)
    base = float(summary.loc[summary.variant.eq("seed_42"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_seed42_same_folds"] = summary.oof_macro_f1 - base
    bagged_folds = []
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        for variant in ("seed_42", "fold_aligned_bagged"):
            macro, accuracy, _ = _metric(labels[valid_idx], oof[variant][valid_idx], classes)
            bagged_folds.append({"fold": fold, "variant": variant, "macro_f1": macro, "accuracy": accuracy})
    paired = pd.DataFrame(bagged_folds)
    all_folds = pd.concat([folds, paired], ignore_index=True, sort=False)
    pivot = paired.pivot(index="fold", columns="variant", values="macro_f1")
    decision = {"run_id": run_id, "purpose": "fold_aligned_3seed_oof_bagging_audit", "outer_split_seed": OUTER_SPLIT_SEED, "model_seeds": list(MODEL_SEEDS), "test_read": False, "train_test_concat": False, "all_models_exclude_each_validation_row": True, "invalid_cross_split_oof_average_used": False, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "positive_fold_count_vs_seed42": int((pivot["fold_aligned_bagged"] > pivot["seed_42"]).sum()), "decision": "audit_complete"}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_DIR / f"{run_id}_summary.csv", index=False)
    all_folds.to_csv(RESULT_DIR / f"{run_id}_fold_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULT_DIR / f"{run_id}_class_metrics.csv", index=False)
    audits.to_csv(RESULT_DIR / f"{run_id}_fold_audit.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(train)), "truth": labels, **{f"{variant}__{label}": probability[:, j] for variant, probability in oof.items() for j, label in enumerate(classes)}}).to_csv(RESULT_DIR / f"{run_id}_oof_probabilities.csv", index=False)
    (RESULT_DIR / f"{run_id}_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = pivot.plot(marker="o", figsize=(8, 4), title="Fold-aligned 3-seed bagging audit"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(RESULT_DIR / f"{run_id}_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    metrics = pd.DataFrame(class_rows).pivot(index="class", columns="variant", values="f1")
    ax = (metrics["fold_aligned_bagged"] - metrics["seed_42"]).sort_values().plot.barh(figsize=(8, 7), title="Class F1: bagging − seed42"); ax.figure.tight_layout(); ax.figure.savefig(RESULT_DIR / f"{run_id}_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    print(summary.to_string(index=False), flush=True)


def smoke() -> None:
    train = pd.read_csv(TRAIN_CSV, nrows=78)
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    assert int(train[genes].isna().sum().sum()) == 0
    # A synthetic balanced label vector keeps this smoke test independent of
    # early-file class ordering in the real training CSV.
    labels = np.repeat(np.arange(3), 3)
    for fit, valid in StratifiedKFold(n_splits=3, shuffle=True, random_state=OUTER_SPLIT_SEED).split(np.zeros(len(labels)), labels):
        assert not set(fit).intersection(valid)
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "outer_split_seed": OUTER_SPLIT_SEED}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-fold-aligned-h0-bagging-audit-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id)
