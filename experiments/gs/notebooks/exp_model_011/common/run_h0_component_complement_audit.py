"""Seed42, fold-safe audit of fixed H0-internal probability combinations."""
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
ROOT = HERE.parents[5]
TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
RESULT = HERE.parent.parent / "result"
SEED = 42
H0_REFERENCE = 0.547915


def _add_source(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# Reuse only prior GS, rule-safe H0 implementation; no team-directory dependency.
_add_source(HERE.parents[2] / "exp_model_007" / "common")
_add_source(HERE.parents[2] / "exp_model_006" / "common")
from h0_selective_eb_replacement import fixed_branch_replacement  # noqa: E402
from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402


def equal_blend(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Fixed 0.5/0.5 probability blend; no weight search."""
    if left.shape != right.shape:
        raise ValueError("probability shape mismatch")
    blended = 0.5 * np.asarray(left, dtype=np.float64) + 0.5 * np.asarray(right, dtype=np.float64)
    return (blended / blended.sum(axis=1, keepdims=True)).astype(np.float32)


def recovery_breakage(truth: np.ndarray, h0_prediction: np.ndarray, candidate_prediction: np.ndarray) -> tuple[int, int]:
    h0_correct = h0_prediction == truth
    candidate_correct = candidate_prediction == truth
    return int(((~h0_correct) & candidate_correct).sum()), int((h0_correct & ~candidate_correct).sum())


def _metric(truth: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[np.asarray(probability).argmax(axis=1)]
    return float(f1_score(truth, prediction, average="macro", zero_division=0)), float(accuracy_score(truth, prediction)), prediction


def _checkpoint_path(run_id: str) -> Path:
    return RESULT / f"{run_id}_seed42_checkpoint.npz"


def _save_checkpoint(path: Path, completed: list[int], oof: dict[str, np.ndarray], folds: list[dict], audits: list[dict]) -> None:
    payload = {f"oof__{name}": value for name, value in oof.items()}
    payload["metadata_json"] = np.asarray(json.dumps({"completed": completed, "folds": folds, "audits": audits}))
    np.savez_compressed(path, **payload)


def _load_checkpoint(path: Path) -> tuple[set[int], dict[str, np.ndarray], list[dict], list[dict]] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        oof = {name.removeprefix("oof__"): archive[name].copy() for name in archive.files if name.startswith("oof__")}
    return set(metadata["completed"]), oof, metadata["folds"], metadata["audits"]


def run(run_id: str) -> None:
    started = perf_counter()
    train = pd.read_csv(TRAIN_CSV)  # Intentional: seed42 OOF never reads test.
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    variants = ("H0_selective_EB", "H0_non_EB", "H0_EB", "equal_H0_non_EB", "equal_H0_EB")
    RESULT.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint(_checkpoint_path(run_id))
    if checkpoint is None:
        completed: set[int] = set()
        oof = {name: np.zeros((len(train), len(classes)), dtype=np.float32) for name in variants}
        fold_rows: list[dict] = []
        audit_rows: list[dict] = []
    else:
        completed, oof, fold_rows, audit_rows = checkpoint
        if set(oof) != set(variants):
            raise ValueError("checkpoint variant mismatch")
        print(f"[H0 complement] resume folds {sorted(completed)}", flush=True)

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_index, valid_index) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        if fold in completed:
            continue
        print(f"[H0 complement] fold {fold}/5: train-only H0 branches", flush=True)
        result = fit_fold(
            train.iloc[fit_index][genes].reset_index(drop=True),
            train.iloc[valid_index][genes].reset_index(drop=True),
            labels[fit_index], genes, classes, seed=SEED * 100 + fold,
        )
        probability = {
            "H0_selective_EB": result["candidate"],
            "H0_non_EB": result["h0"],
            "H0_EB": fixed_branch_replacement(result["eb_lr"], result["specialist"]),
        }
        probability["equal_H0_non_EB"] = equal_blend(probability["H0_selective_EB"], probability["H0_non_EB"])
        probability["equal_H0_EB"] = equal_blend(probability["H0_selective_EB"], probability["H0_EB"])
        for name, value in probability.items():
            oof[name][valid_index] = value
            macro, accuracy, _ = _metric(labels[valid_index], value, classes)
            fold_rows.append({"fold": fold, "variant": name, "macro_f1": macro, "accuracy": accuracy, "feature_count": int(result["candidate_feature_count"])})
        audit_rows.append({
            "fold": fold,
            "test_read": False,
            "train_test_concat": False,
            "outer_validation_used_for_fit": False,
            "fold_train_only_eb": True,
            "fixed_class_gene_mutation_rules": False,
            "leakage_check": not bool(result["audit"]["raw_train_test_concat"]),
            "nan_as_mutation_count": int(result["audit"]["nan_as_mutation_count"]),
            "convergence_warning_count": int(result["h0_warning"] + result["eb_warning"]),
        })
        completed.add(fold)
        _save_checkpoint(_checkpoint_path(run_id), sorted(completed), oof, fold_rows, audit_rows)
        print(f"[H0 complement] fold {fold}/5 checkpoint saved", flush=True)
        del result, probability
        gc.collect()

    folds = pd.DataFrame(fold_rows)
    audits = pd.DataFrame(audit_rows)
    if not bool(audits.leakage_check.all()) or int(audits.nan_as_mutation_count.max()) != 0:
        raise AssertionError("leakage/NaN audit failed")
    h0_macro, _, h0_prediction = _metric(labels, oof["H0_selective_EB"], classes)
    summary_rows, class_rows, overlap_rows = [], [], []
    low_margin = np.sort(oof["H0_selective_EB"], axis=1)[:, -1] - np.sort(oof["H0_selective_EB"], axis=1)[:, -2]
    low_mask = low_margin < 0.05
    low_rows = []
    for name in variants:
        macro, accuracy, prediction = _metric(labels, oof[name], classes)
        recovered, broken = recovery_breakage(labels, h0_prediction, prediction)
        precision, recall, class_f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"variant": name, "class": label, "support": int(n), "precision": float(p), "recall": float(r), "f1": float(score)} for label, p, r, score, n in zip(classes, precision, recall, class_f1, support))
        summary_rows.append({"variant": name, "oof_macro_f1": macro, "oof_accuracy": accuracy, "feature_count": float(folds.loc[folds.variant.eq(name), "feature_count"].mean()), "convergence_warning_count": int(audits.convergence_warning_count.sum()), "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": perf_counter() - started, "delta_vs_h0": macro - h0_macro, "recovered_h0_errors": recovered, "broken_h0_correct": broken})
        low_rows.append({"variant": name, "group": "h0_margin_lt_005", "support": int(low_mask.sum()), "macro_f1": float(f1_score(labels[low_mask], prediction[low_mask], average="macro", zero_division=0))})
        if name != "H0_selective_EB":
            overlap_rows.append({"variant": name, "hard_prediction_disagreement_rate": float((prediction != h0_prediction).mean()), "h0_wrong_recovered": recovered, "h0_correct_broken": broken})
    summary, class_frame, low_frame, overlap = pd.DataFrame(summary_rows), pd.DataFrame(class_rows), pd.DataFrame(low_rows), pd.DataFrame(overlap_rows)
    candidate = summary.loc[summary.variant.ne("H0_selective_EB")].sort_values("oof_macro_f1", ascending=False).iloc[0]
    fold_pivot = folds.pivot(index="fold", columns="variant", values="macro_f1")
    candidate_low = float(low_frame.loc[low_frame.variant.eq(candidate.variant), "macro_f1"].iloc[0])
    h0_low = float(low_frame.loc[low_frame.variant.eq("H0_selective_EB"), "macro_f1"].iloc[0])
    decision = {
        "run_id": run_id,
        "h0_reference": H0_REFERENCE,
        "h0_reference_match": abs(h0_macro - H0_REFERENCE) <= 0.001,
        "h0_reference_delta": h0_macro - H0_REFERENCE,
        "test_read": False,
        "train_test_concat": False,
        "leakage_check": True,
        "nan_as_mutation_count": 0,
        "best_fixed_candidate": str(candidate.variant),
        "best_delta_vs_h0": float(candidate.delta_vs_h0),
        "positive_fold_count": int((fold_pivot[candidate.variant] > fold_pivot["H0_selective_EB"]).sum()),
        "recovered_exceeds_broken": bool(candidate.recovered_h0_errors > candidate.broken_h0_correct),
        "low_margin_delta": candidate_low - h0_low,
        "three_seed_candidate": bool(candidate.delta_vs_h0 >= .003 and int((fold_pivot[candidate.variant] > fold_pivot["H0_selective_EB"]).sum()) >= 4 and candidate.recovered_h0_errors > candidate.broken_h0_correct and candidate_low - h0_low >= -.003),
    }
    prefix = RESULT / f"{run_id}_seed42"
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    folds.to_csv(prefix.with_name(prefix.name + "_fold_metrics.csv"), index=False)
    class_frame.to_csv(prefix.with_name(prefix.name + "_class_metrics.csv"), index=False)
    low_frame.to_csv(prefix.with_name(prefix.name + "_low_margin.csv"), index=False)
    overlap.to_csv(prefix.with_name(prefix.name + "_error_overlap.csv"), index=False)
    audits.to_csv(prefix.with_name(prefix.name + "_fold_audit.csv"), index=False)
    pd.DataFrame({"row_index": np.arange(len(train)), "truth": labels, **{f"{name}__{label}": oof[name][:, index] for name in variants for index, label in enumerate(classes)}}).to_csv(prefix.with_name(prefix.name + "_oof_probabilities.csv"), index=False)
    prefix.with_name(prefix.name + "_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = folds.pivot(index="fold", columns="variant", values="macro_f1").plot(marker="o", title="H0 internal fixed-combination audit"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_fold_macro_f1.png"), dpi=160); plt.close(ax.figure)
    h0_class = class_frame.loc[class_frame.variant.eq("H0_selective_EB")].set_index("class").f1
    delta = class_frame.loc[class_frame.variant.eq(candidate.variant)].set_index("class").f1 - h0_class
    ax = delta.sort_values().plot.barh(title=f"{candidate.variant}: class F1 delta"); ax.axvline(0, color="black"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_class_f1_delta.png"), dpi=160); plt.close(ax.figure)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def smoke() -> None:
    train = pd.read_csv(TRAIN_CSV, nrows=32)
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    assert len(genes) == 4384 and int(train[genes].isna().sum().sum()) == 0
    test_probability = equal_blend(np.array([[.9, .1]], dtype=np.float32), np.array([[.1, .9]], dtype=np.float32))
    assert np.allclose(test_probability.sum(axis=1), 1.0)
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "fixed_blends": ["0.5/0.5"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-h0-component-complement-audit-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id)
