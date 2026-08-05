"""Faithful H0 versus fixed Selective-EB LR branch replacement.

OOF mode reads train.csv only.  All supervised event statistics and the
automatic LGBM specialist are fitted inside each outer-fold training split.
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
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from fold_checkpoint import experiment_result_dir, load_checkpoint, save_checkpoint
from h0_selective_eb_replacement import (
    SELECTIVE_MARGIN,
    empirical_bayes_features,
    fixed_branch_replacement,
    selective_probability,
)

HERE = Path(__file__).resolve()
DEFAULT_SEEDS = (42,)
H0_WEIGHT = 0.80
SPECIALIST_WEIGHT = 0.20
SCREEN_DELTA = 0.003


def _h0_common() -> Path:
    path = HERE.parents[2] / "exp_model_006" / "common"
    if not path.exists():
        raise FileNotFoundError("GS faithful H0 source was not found")
    return path


if str(_h0_common()) not in sys.path:
    sys.path.insert(0, str(_h0_common()))

from h0_faithful_pipeline import _aligned_probability, _hard_specialist, build_design_matrices  # noqa: E402


def result_directory(runner_path: Path = HERE) -> Path:
    return experiment_result_dir(runner_path)


def summary_columns() -> list[str]:
    return [
        "seed", "variant", "oof_macro_f1", "oof_accuracy", "feature_count_mean",
        "convergence_warning_count", "leakage_check", "nan_as_mutation_count",
        "runtime_seconds", "delta_vs_h0",
    ]


def split_checkpoint_oof(payload: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Separate the row-wise gate mask before iterating model probability matrices."""
    copied = {name: np.asarray(value).copy() for name, value in payload.items()}
    if "gate_usage" not in copied:
        raise KeyError("checkpoint is missing gate_usage")
    return {name: value for name, value in copied.items() if name != "gate_usage"}, copied["gate_usage"].astype(bool)


def project_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "data" / "raw" / "train.csv").exists():
            return candidate
    raise FileNotFoundError("data/raw/train.csv was not found")


def _fit_lr_probability(x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_apply: sparse.csr_matrix, classes: np.ndarray, *, seed: int) -> tuple[np.ndarray, int]:
    model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_fit, y_fit)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return _aligned_probability(model, model.predict_proba(x_apply), classes).astype(np.float32), int(warning_count)


def fit_fold(
    fit_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    y_fit: np.ndarray,
    genes: list[str],
    classes: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Fit unchanged H0 and the EB-replaced LR branch using one outer train fold."""
    x_fit, x_valid, names, audit = build_design_matrices(fit_frame, valid_frame, y_fit, genes, seed=seed)
    h0_lr, h0_warning = _fit_lr_probability(x_fit, y_fit, x_valid, classes, seed=seed)

    eb_fit, eb_valid = empirical_bayes_features(fit_frame, valid_frame, y_fit, classes, genes, seed=seed)
    eb_x_fit = sparse.hstack([x_fit, sparse.csr_matrix(eb_fit)], format="csr")
    eb_x_valid = sparse.hstack([x_valid, sparse.csr_matrix(eb_valid)], format="csr")
    candidate_feature_count = int(eb_x_fit.shape[1])
    eb_lr, eb_warning = _fit_lr_probability(eb_x_fit, y_fit, eb_x_valid, classes, seed=seed)
    gated_lr, use_non_eb = selective_probability(h0_lr, eb_lr)

    lgbm = LGBMClassifier(
        objective="multiclass", boosting_type="gbdt", num_class=len(classes),
        n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10,
        min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced",
        random_state=seed, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    lgbm.fit(x_fit, y_fit)
    specialist, pairs = _hard_specialist(
        x_fit, y_fit, x_valid, _aligned_probability(lgbm, lgbm.predict_proba(x_valid), classes), classes, names, seed,
    )
    h0 = fixed_branch_replacement(h0_lr, specialist)
    candidate = fixed_branch_replacement(gated_lr, specialist)
    del lgbm, eb_x_fit, eb_x_valid
    gc.collect()
    return {
        "h0_lr": h0_lr, "eb_lr": eb_lr, "gated_lr": gated_lr, "specialist": specialist,
        "h0": h0, "candidate": candidate, "use_non_eb": use_non_eb,
        "pairs": pairs, "feature_count": int(x_fit.shape[1]), "candidate_feature_count": candidate_feature_count,
        "audit": audit, "h0_warning": h0_warning, "eb_warning": eb_warning,
    }


def _metrics(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    predicted = classes[np.asarray(probability).argmax(axis=1)]
    return (
        float(f1_score(labels, predicted, average="macro", zero_division=0)),
        float(accuracy_score(labels, predicted)),
        predicted,
    )


def _empty_oof(n_rows: int, n_classes: int) -> dict[str, np.ndarray]:
    return {name: np.zeros((n_rows, n_classes), dtype=np.float32) for name in ("h0", "candidate", "h0_lr", "eb_lr", "gated_lr")}


def run_seed(train: pd.DataFrame, genes: list[str], labels: np.ndarray, classes: np.ndarray, seed: int, checkpoint_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    started = perf_counter()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        oof, gate_usage, fold_rows, audit_rows, completed = _empty_oof(len(train), len(classes)), np.zeros(len(train), dtype=bool), [], [], set()
    else:
        oof, gate_usage = split_checkpoint_oof(checkpoint["oof"])
        fold_rows, audit_rows, completed = checkpoint["fold_rows"], checkpoint["audit_rows"], set(checkpoint["completed_folds"])
        print(f"[H0 selective EB] seed {seed}: resume folds {sorted(completed)}", flush=True)

    for fold, (fit_index, valid_index) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        if fold in completed:
            continue
        print(f"[H0 selective EB] seed {seed}, fold {fold}/5: fold-train feature fit", flush=True)
        result = fit_fold(
            train.iloc[fit_index][genes].reset_index(drop=True), train.iloc[valid_index][genes].reset_index(drop=True),
            labels[fit_index], genes, classes, seed=seed * 100 + fold,
        )
        for name in oof:
            oof[name][valid_index] = result[name]
        gate_usage[valid_index] = result["use_non_eb"]
        for name, probability, count in (
            ("H0", result["h0"], result["feature_count"]),
            ("H0_EB_LR", fixed_branch_replacement(result["eb_lr"], result["specialist"]), result["candidate_feature_count"]),
            ("H0_selective_EB", result["candidate"], result["candidate_feature_count"]),
        ):
            macro_f1, accuracy, _ = _metrics(labels[valid_index], probability, classes)
            fold_rows.append({"seed": seed, "fold": fold, "variant": name, "macro_f1": macro_f1, "accuracy": accuracy, "feature_count": count})
        audit_rows.append({
            "seed": seed, "fold": fold, "test_read": False, "raw_train_test_concat": bool(result["audit"]["raw_train_test_concat"]),
            "vocabulary_source": result["audit"]["vocabulary_source"], "outer_validation_used_for_fit": False,
            "fixed_class_gene_mutation_rules": False, "fold_train_only_eb": True,
            "leakage_check": not bool(result["audit"]["raw_train_test_concat"]), "nan_as_mutation_count": int(result["audit"]["nan_as_mutation_count"]),
            "h0_convergence_warning_count": result["h0_warning"], "eb_convergence_warning_count": result["eb_warning"],
            "specialist_pairs": repr(result["pairs"]), "gate_non_eb_rows": int(result["use_non_eb"].sum()),
        })
        completed.add(fold)
        save_checkpoint(checkpoint_path, {"completed_folds": list(completed), "fold_rows": fold_rows, "audit_rows": audit_rows, "oof": {**oof, "gate_usage": gate_usage}})
        print(f"[H0 selective EB] seed {seed}, fold {fold}/5 checkpoint saved", flush=True)
        del result
        gc.collect()

    fold_frame, audit_frame = pd.DataFrame(fold_rows), pd.DataFrame(audit_rows)
    warning_count = int(audit_frame.h0_convergence_warning_count.sum() + audit_frame.eb_convergence_warning_count.sum())
    summary_rows, class_rows = [], []
    for name, probability in (("H0", oof["h0"]), ("H0_selective_EB", oof["candidate"])):
        macro_f1, accuracy, prediction = _metrics(labels, probability, classes)
        summary_rows.append({"seed": seed, "variant": name, "oof_macro_f1": macro_f1, "oof_accuracy": accuracy, "feature_count_mean": float(fold_frame.loc[fold_frame.variant.eq(name), "feature_count"].mean()), "convergence_warning_count": warning_count, "leakage_check": bool(audit_frame.leakage_check.all()), "nan_as_mutation_count": int(audit_frame.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started})
        precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"seed": seed, "variant": name, "class": label, "precision": p, "recall": r, "f1": score, "support": int(n)} for label, p, r, score, n in zip(classes, precision, recall, f1, support))
    summary = pd.DataFrame(summary_rows)
    h0_score = float(summary.loc[summary.variant.eq("H0"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_h0"] = summary.oof_macro_f1 - h0_score
    return summary, fold_frame, audit_frame, pd.DataFrame(class_rows), oof, gate_usage


def smoke() -> None:
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv", nrows=12)
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    non_eb = np.asarray([[.6, .4], [.1, .9]], dtype=np.float32)
    eb = np.asarray([[.51, .49], [.1, .9]], dtype=np.float32)
    gated, use_non_eb = selective_probability(non_eb, eb)
    assert bool(use_non_eb[0]) and not bool(use_non_eb[1])
    np.testing.assert_allclose(fixed_branch_replacement(gated, non_eb).sum(axis=1), 1.0)
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "fixed_class_gene_mutation_rules": False}), flush=True)


def run(run_id: str, seeds: tuple[int, ...]) -> None:
    root = project_root()
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    result = result_directory(); result.mkdir(exist_ok=True)
    outputs = [run_seed(train, genes, labels, classes, seed, result / f"{run_id}_seed{seed}_checkpoint.npz") for seed in seeds]
    summary = pd.concat([item[0] for item in outputs], ignore_index=True)
    folds = pd.concat([item[1] for item in outputs], ignore_index=True)
    audits = pd.concat([item[2] for item in outputs], ignore_index=True)
    class_metrics = pd.concat([item[3] for item in outputs], ignore_index=True)
    aggregate = summary.groupby("variant", as_index=False).agg(seed_count=("seed", "nunique"), oof_macro_f1_mean=("oof_macro_f1", "mean"), oof_macro_f1_std=("oof_macro_f1", "std"), delta_vs_h0_mean=("delta_vs_h0", "mean"), delta_vs_h0_std=("delta_vs_h0", "std"), convergence_warning_count=("convergence_warning_count", "sum"), leakage_check=("leakage_check", "all"), nan_as_mutation_count=("nan_as_mutation_count", "max"))
    pivot = folds.pivot_table(index=["seed", "fold"], columns="variant", values="macro_f1")
    candidate = summary.loc[summary.variant.eq("H0_selective_EB")].sort_values("seed")
    screen_pass = len(seeds) == 1 and float(candidate.delta_vs_h0.iloc[0]) >= SCREEN_DELTA and int((pivot["H0_selective_EB"] > pivot["H0"]).sum()) >= 4
    three_seed_pass = len(seeds) == 3 and bool(np.all(candidate.delta_vs_h0.to_numpy() > 0.0)) and float(candidate.delta_vs_h0.mean()) >= .003 and int((pivot["H0_selective_EB"] > pivot["H0"]).sum()) >= 11
    decision = {"run_id": run_id, "seeds": list(seeds), "h0_weight": H0_WEIGHT, "specialist_weight": SPECIALIST_WEIGHT, "selective_margin": SELECTIVE_MARGIN, "threshold_retuned": False, "screen_pass": bool(screen_pass), "three_seed_pass": bool(three_seed_pass), "positive_fold_count": int((pivot["H0_selective_EB"] > pivot["H0"]).sum()), "test_read": False, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "decision": "screen_candidate" if screen_pass else ("accepted_3seed" if three_seed_pass else "rejected_or_not_detected")}
    summary.to_csv(result / f"{run_id}_seed_summary.csv", index=False)
    aggregate.to_csv(result / f"{run_id}_aggregate_summary.csv", index=False)
    folds.to_csv(result / f"{run_id}_fold_metrics.csv", index=False)
    audits.to_csv(result / f"{run_id}_fold_audit.csv", index=False)
    class_metrics.to_csv(result / f"{run_id}_class_metrics.csv", index=False)
    for seed, output in zip(seeds, outputs):
        oof, usage = output[4], output[5]
        pd.DataFrame({"row_index": np.arange(len(train)), "truth": labels, "gate_uses_non_eb": usage, **{f"h0__{label}": oof["h0"][:, index] for index, label in enumerate(classes)}, **{f"candidate__{label}": oof["candidate"][:, index] for index, label in enumerate(classes)}}).to_csv(result / f"{run_id}_seed{seed}_oof_probabilities.csv", index=False)
    (result / f"{run_id}_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = pivot.plot(marker="o", figsize=(10, 4), title="H0 vs Selective-EB branch replacement"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    class_pivot = class_metrics.pivot_table(index=["seed", "class"], columns="variant", values="f1")
    delta = (class_pivot["H0_selective_EB"] - class_pivot["H0"]).groupby("class").mean().sort_values()
    ax = delta.plot.barh(figsize=(8, 7), title="Class F1 delta: H0 Selective-EB − H0"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-h0-selective-eb-branch-replacement-01")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id, tuple(args.seeds))


if __name__ == "__main__":
    main()
