"""Seed42, fold-safe H0 complement using automatic mutation-profile prototypes.

The experiment reads the train dataset only.  Class prototypes, vocabulary, IDF,
priors and all H0 supervised transforms are fitted within each outer-fold train
split; validation rows are transformed and scored only.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[5]
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"
RESULT = HERE.parent.parent / "result"
SEED = 42
H0_REFERENCE = 0.547915
H0_WEIGHT = 0.80
PROTOTYPE_WEIGHT = 0.20
SCREEN_DELTA = 0.015
LOW_MARGIN = 0.05


def _add_gs_source(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"GS source missing: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_add_gs_source(HERE.parent)
_add_gs_source(HERE.parents[2] / "exp_model_006" / "common")
_add_gs_source(HERE.parents[2] / "exp_model_007" / "common")
from prototype_similarity_core import fit_train_only_prototype, normalize_cell, predict_prototype  # noqa: E402
from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402


def contract() -> dict:
    return {
        "test_read": False,
        "train_test_concat": False,
        "vocabulary_source": "outer_fold_train_only",
        "prototype_source": "outer_fold_train_only",
        "fixed_class_gene_mutation_rules": False,
        "outer_validation_used_for_fit": False,
        "leakage_check": True,
        "nan_as_mutation_count": 0,
        "h0_weight": H0_WEIGHT,
        "prototype_weight": PROTOTYPE_WEIGHT,
    }


def probability_blend(h0: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    if h0.shape != prototype.shape:
        raise ValueError("probability shape mismatch")
    probability = H0_WEIGHT * np.asarray(h0, dtype=np.float64) + PROTOTYPE_WEIGHT * np.asarray(prototype, dtype=np.float64)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability.astype(np.float32)


def metrics(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[np.asarray(probability).argmax(axis=1)]
    return (
        float(f1_score(labels, prediction, average="macro", zero_division=0)),
        float(accuracy_score(labels, prediction)),
        prediction,
    )


def _checkpoint_path(run_id: str) -> Path:
    return RESULT / f"{run_id}_seed42_checkpoint.npz"


def _save_checkpoint(path: Path, completed: set[int], oof: dict[str, np.ndarray], fold_rows: list[dict], audit_rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    payload = {f"oof__{name}": values for name, values in oof.items()}
    payload["metadata_json"] = np.asarray(json.dumps({"completed": sorted(completed), "fold_rows": fold_rows, "audit_rows": audit_rows}))
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    path.with_suffix(".progress.json").write_text(json.dumps({"completed": sorted(completed)}, indent=2), encoding="utf-8")


def _load_checkpoint(path: Path) -> tuple[set[int], dict[str, np.ndarray], list[dict], list[dict]] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        oof = {name.removeprefix("oof__"): archive[name].copy() for name in archive.files if name.startswith("oof__")}
    return set(metadata["completed"]), oof, metadata["fold_rows"], metadata["audit_rows"]


def _class_rows(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray, variant: str) -> list[dict]:
    _, _, prediction = metrics(labels, probability, classes)
    precision, recall, scores, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
    return [
        {"variant": variant, "class": label, "support": int(n), "precision": float(p), "recall": float(r), "f1": float(score)}
        for label, p, r, score, n in zip(classes, precision, recall, scores, support)
    ]


def _write_outputs(run_id: str, labels: np.ndarray, classes: np.ndarray, oof: dict[str, np.ndarray], folds: pd.DataFrame, audits: pd.DataFrame, started: float) -> None:
    h0_macro, _, h0_prediction = metrics(labels, oof["H0_selective_EB"], classes)
    summary_rows, class_rows = [], []
    margin = np.sort(oof["H0_selective_EB"], axis=1)[:, -1] - np.sort(oof["H0_selective_EB"], axis=1)[:, -2]
    low_mask = margin < LOW_MARGIN
    low_rows = []
    for variant, probability in oof.items():
        macro, accuracy, prediction = metrics(labels, probability, classes)
        recovered = int(((h0_prediction != labels) & (prediction == labels)).sum())
        broken = int(((h0_prediction == labels) & (prediction != labels)).sum())
        summary_rows.append({
            "variant": variant,
            "oof_macro_f1": macro,
            "oof_accuracy": accuracy,
            "feature_count": float(folds.loc[folds.variant.eq(variant), "feature_count"].mean()),
            "convergence_warning_count": int(audits.convergence_warning_count.sum()),
            "leakage_check": bool(audits.leakage_check.all()),
            "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()),
            "runtime_seconds": perf_counter() - started,
            "delta_vs_h0": macro - h0_macro,
            "recovered_h0_errors": recovered,
            "broken_h0_correct": broken,
        })
        class_rows.extend(_class_rows(labels, probability, classes, variant))
        low_rows.append({"variant": variant, "group": "h0_margin_lt_005", "support": int(low_mask.sum()), "macro_f1": float(f1_score(labels[low_mask], prediction[low_mask], average="macro", zero_division=0))})
    summary = pd.DataFrame(summary_rows)
    class_metrics = pd.DataFrame(class_rows)
    low_metrics = pd.DataFrame(low_rows)
    fold_pivot = folds.pivot(index="fold", columns="variant", values="macro_f1")
    candidate = summary.loc[summary.variant.eq("H0_plus_prototype")].iloc[0]
    h0_low = float(low_metrics.loc[low_metrics.variant.eq("H0_selective_EB"), "macro_f1"].iloc[0])
    candidate_low = float(low_metrics.loc[low_metrics.variant.eq("H0_plus_prototype"), "macro_f1"].iloc[0])
    h0_class = class_metrics.loc[class_metrics.variant.eq("H0_selective_EB")].set_index("class").f1
    candidate_class = class_metrics.loc[class_metrics.variant.eq("H0_plus_prototype")].set_index("class").f1
    class_delta = (candidate_class - h0_class).rename("delta_f1").reset_index(names="class")
    decision = {
        **contract(),
        "run_id": run_id,
        "h0_reference": H0_REFERENCE,
        "h0_oof_macro_f1": h0_macro,
        "h0_reference_match": abs(h0_macro - H0_REFERENCE) <= 0.001,
        "h0_reference_delta": h0_macro - H0_REFERENCE,
        "prototype_blend_oof_macro_f1": float(candidate.oof_macro_f1),
        "delta_vs_h0": float(candidate.delta_vs_h0),
        "positive_fold_count": int((fold_pivot["H0_plus_prototype"] > fold_pivot["H0_selective_EB"]).sum()),
        "low_margin_delta": candidate_low - h0_low,
        "recovered_h0_errors": int(candidate.recovered_h0_errors),
        "broken_h0_correct": int(candidate.broken_h0_correct),
        "class_f1_improved_count": int((class_delta.delta_f1 > 0).sum()),
        "screen_candidate": bool(
            candidate.delta_vs_h0 >= SCREEN_DELTA
            and int((fold_pivot["H0_plus_prototype"] > fold_pivot["H0_selective_EB"]).sum()) >= 4
            and candidate_low - h0_low >= -0.003
            and candidate.recovered_h0_errors > candidate.broken_h0_correct
        ),
    }
    prefix = RESULT / f"{run_id}_seed42"
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    folds.to_csv(prefix.with_name(prefix.name + "_fold_metrics.csv"), index=False)
    class_metrics.to_csv(prefix.with_name(prefix.name + "_class_metrics.csv"), index=False)
    low_metrics.to_csv(prefix.with_name(prefix.name + "_low_margin.csv"), index=False)
    audits.to_csv(prefix.with_name(prefix.name + "_fold_audit.csv"), index=False)
    class_delta.to_csv(prefix.with_name(prefix.name + "_class_f1_delta.csv"), index=False)
    pd.DataFrame({
        "row_index": np.arange(len(labels)),
        "truth": labels,
        **{f"{variant}__{label}": values[:, index] for variant, values in oof.items() for index, label in enumerate(classes)},
    }).to_csv(prefix.with_name(prefix.name + "_oof_probabilities.csv"), index=False)
    prefix.with_name(prefix.name + "_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = fold_pivot.plot(marker="o", title="H0 vs train-only prototype similarity"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_fold_macro_f1.png"), dpi=160); plt.close(ax.figure)
    ax = class_delta.sort_values("delta_f1").plot.barh(x="class", y="delta_f1", legend=False, title="Prototype blend: class F1 delta"); ax.axvline(0, color="black"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_class_f1_delta.png"), dpi=160); plt.close(ax.figure)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def run(run_id: str) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    train = pd.read_csv(TRAIN_CSV)
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    variants = ("H0_selective_EB", "prototype_similarity", "H0_plus_prototype")
    checkpoint = _load_checkpoint(_checkpoint_path(run_id))
    if checkpoint is None:
        completed: set[int] = set()
        oof = {variant: np.zeros((len(train), len(classes)), dtype=np.float32) for variant in variants}
        fold_rows: list[dict] = []
        audit_rows: list[dict] = []
    else:
        completed, oof, fold_rows, audit_rows = checkpoint
        if set(oof) != set(variants):
            raise ValueError("checkpoint variant mismatch")
        print(f"[prototype screen] resume completed folds {sorted(completed)}", flush=True)

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_index, valid_index) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        if fold in completed:
            continue
        print(f"[prototype screen] fold {fold}/5: H0 + fold-train prototype", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        fit_labels = labels[fit_index]
        h0 = fit_fold(fit_frame, valid_frame, fit_labels, genes, classes, seed=SEED * 100 + fold)
        prototype_artifacts = fit_train_only_prototype(fit_frame, fit_labels, genes, classes)
        prototype = predict_prototype(valid_frame, genes, prototype_artifacts)
        probability = {
            "H0_selective_EB": h0["candidate"],
            "prototype_similarity": prototype,
            "H0_plus_prototype": probability_blend(h0["candidate"], prototype),
        }
        for variant, values in probability.items():
            oof[variant][valid_index] = values
            macro, accuracy, _ = metrics(labels[valid_index], values, classes)
            count = h0["candidate_feature_count"] if variant != "prototype_similarity" else len(prototype_artifacts.vocabulary)
            fold_rows.append({"fold": fold, "variant": variant, "macro_f1": macro, "accuracy": accuracy, "feature_count": int(count)})
        audit_rows.append({
            "fold": fold,
            **contract(),
            "inner_eb_crossfit": True,
            "h0_convergence_warning_count": int(h0["h0_warning"]),
            "eb_convergence_warning_count": int(h0["eb_warning"]),
            "convergence_warning_count": int(h0["h0_warning"] + h0["eb_warning"]),
            "prototype_vocabulary_size": int(len(prototype_artifacts.vocabulary)),
            "outer_validation_used_for_h0_or_prototype_fit": False,
        })
        completed.add(fold)
        _save_checkpoint(_checkpoint_path(run_id), completed, oof, fold_rows, audit_rows)
        print(f"[prototype screen] fold {fold}/5 checkpoint saved", flush=True)
        del h0, prototype_artifacts, prototype, probability, fit_frame, valid_frame
        gc.collect()

    audits = pd.DataFrame(audit_rows)
    if not bool(audits.leakage_check.all()) or int(audits.nan_as_mutation_count.max()) != 0:
        raise AssertionError("leakage or NaN contract failed")
    _write_outputs(run_id, labels, classes, oof, pd.DataFrame(fold_rows), audits, started)


def smoke() -> None:
    train = pd.read_csv(TRAIN_CSV, nrows=48)
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    if len(genes) != 4384:
        raise AssertionError("unexpected gene column count")
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    if normalize_cell(np.nan) or normalize_cell("WT") or normalize_cell(""):
        raise AssertionError("non-event cell parsed as event")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    artifacts = fit_train_only_prototype(train.iloc[:32][genes], labels[:32], genes, classes)
    probability = predict_prototype(train.iloc[32:][genes], genes, artifacts)
    row_sum = float(np.round(probability.sum(axis=1).mean(), 8))
    print(json.dumps({"smoke": "ok", **contract(), "prototype_probability_row_sum": row_sum}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-h0-prototype-similarity-01")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    smoke() if arguments.smoke else run(arguments.run_id)

