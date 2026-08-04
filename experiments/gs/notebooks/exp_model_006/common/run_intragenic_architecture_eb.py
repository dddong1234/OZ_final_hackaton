"""Seed42 H0 + intragenic architecture Empirical-Bayes screen; train-only."""
from __future__ import annotations

import argparse
import gc
import json
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from h0_faithful_pipeline import build_design_matrices
from h1_auto_confusion_moe import _align, _h0_specialist_probability, discover_similarity_pairs, fit_h0_fold
from intragenic_architecture import architecture_token_sets, cross_fitted_architecture_scores


DEFAULT_SEEDS = (42, 777, 2024)
H0_REFERENCE_SEED42 = .544744
RESULT_COLUMNS = {
    "summary": {"variant", "oof_macro_f1", "oof_accuracy", "feature_count", "convergence_warning_count", "leakage_check", "nan_as_mutation_count", "delta_vs_h0"},
    "fold": {"fold", "variant", "macro_f1", "accuracy", "feature_count", "delta_vs_h0"},
}


def project_root() -> Path:
    for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv not found")


def _metrics(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[probability.argmax(axis=1)]
    return float(f1_score(labels, prediction, average="macro", zero_division=0)), float(accuracy_score(labels, prediction)), prediction


def _fit_architecture_h0(fit_frame, valid_frame, labels, all_labels, genes, classes, token_sets, fit_index, valid_index, seed):
    x_fit, x_valid, names, audit = build_design_matrices(fit_frame, valid_frame, labels, genes, seed=seed)
    architecture_fit, architecture_valid, architecture_names, vocabulary_size = cross_fitted_architecture_scores(token_sets, all_labels, classes, fit_index, valid_index, seed=seed)
    x_fit = sparse.hstack([x_fit, sparse.csr_matrix(architecture_fit)], format="csr")
    x_valid = sparse.hstack([x_valid, sparse.csr_matrix(architecture_valid)], format="csr")
    names = [*names, *architecture_names]
    lr = LogisticRegression(solver="lbfgs", C=.07, max_iter=2000, class_weight="balanced", random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr.fit(x_fit, labels)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    lr_probability = _align(lr, lr.predict_proba(x_valid), classes)
    main = LGBMClassifier(objective="multiclass", boosting_type="gbdt", num_class=len(classes), n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10, min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced", random_state=42, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1)
    main.fit(x_fit, labels)
    pairs = discover_similarity_pairs(x_fit, names, labels, classes)
    specialist = _h0_specialist_probability(_align(main, main.predict_proba(x_valid), classes), x_fit, labels, x_valid, classes, pairs, seed=42)
    probability = .80 * lr_probability + .20 * specialist
    audit = {**audit, "architecture_vocabulary_size": vocabulary_size, "architecture_feature_count": len(architecture_names), "architecture_fit_scope": "outer_fold_train_only"}
    return probability, len(names), warning_count, audit


def _decision(h0: float, candidate: float, fold_delta: np.ndarray) -> dict:
    delta, positive = float(candidate - h0), int((fold_delta > 0).sum())
    if delta >= .015 and positive >= 4:
        label = "strong_validation_candidate"
    elif delta >= .008 and positive >= 4:
        label = "validation_candidate"
    elif delta > 0:
        label = "not_detected"
    else:
        label = "rejected"
    return {"decision": label, "delta_vs_h0": delta, "positive_fold_count": positive}


def smoke() -> None:
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv", nrows=8)
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    assert len(genes) == 4384
    assert int(train[genes].isna().sum().sum()) == 0
    assert architecture_token_sets(train[genes], genes)
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "fixed_class_gene_mutation_rules": False}))


def run_one_seed(train: pd.DataFrame, genes: list[str], all_labels: np.ndarray, classes: np.ndarray, token_sets: list[set[str]], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    started = perf_counter()
    h0_oof = np.zeros((len(train), len(classes))); architecture_oof = np.zeros_like(h0_oof)
    fold_rows, audit_rows, warnings_h0, warnings_arch = [], [], 0, 0
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (fit_index, valid_index) in enumerate(outer.split(np.zeros(len(train)), all_labels), 1):
        print(f"[architecture-eb] seed {seed}, outer fold {fold}/5: H0 and architecture-EB", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        h0 = fit_h0_fold(fit_frame, valid_frame, all_labels[fit_index], genes, classes, seed=seed * 100 + fold)
        arch_probability, arch_count, arch_warning, arch_audit = _fit_architecture_h0(fit_frame, valid_frame, all_labels[fit_index], all_labels, genes, classes, token_sets, fit_index, valid_index, seed * 100 + fold)
        if h0.audit["raw_train_test_concat"] or arch_audit["raw_train_test_concat"]:
            raise AssertionError("train/test concatenation contract violation")
        h0_oof[valid_index], architecture_oof[valid_index] = h0.probability, arch_probability
        h0_f1, _, _ = _metrics(all_labels[valid_index], h0.probability, classes)
        arch_f1, _, _ = _metrics(all_labels[valid_index], arch_probability, classes)
        for name, values, score, count in (("H0", h0.probability, h0_f1, len(h0.names)), ("H0_intragenic_architecture_EB", arch_probability, arch_f1, arch_count)):
            fold_rows.append({"seed": seed, "fold": fold, "variant": name, "macro_f1": score, "accuracy": float(accuracy_score(all_labels[valid_index], classes[values.argmax(axis=1)])), "feature_count": count, "delta_vs_h0": score - h0_f1})
        audit_rows.append({"seed": seed, "fold": fold, "architecture_vocabulary_size": arch_audit["architecture_vocabulary_size"], "architecture_feature_count": arch_audit["architecture_feature_count"], "outer_validation_used_for_eb_fit": False, "leakage_check": True, "nan_as_mutation_count": 0})
        warnings_h0 += h0.convergence_warnings; warnings_arch += arch_warning
        del h0, arch_probability, fit_frame, valid_frame
        gc.collect()
    h0_score, h0_acc, h0_prediction = _metrics(all_labels, h0_oof, classes)
    arch_score, arch_acc, arch_prediction = _metrics(all_labels, architecture_oof, classes)
    folds = pd.DataFrame(fold_rows)
    h0_folds = folds[folds.variant.eq("H0")].sort_values("fold").macro_f1.to_numpy()
    arch_folds = folds[folds.variant.eq("H0_intragenic_architecture_EB")].sort_values("fold").macro_f1.to_numpy()
    verdict = _decision(h0_score, arch_score, arch_folds - h0_folds)
    summary = pd.DataFrame([
        {"seed": seed, "variant": "H0", "oof_macro_f1": h0_score, "oof_accuracy": h0_acc, "feature_count": float(folds[folds.variant.eq("H0")].feature_count.mean()), "convergence_warning_count": warnings_h0, "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": 0.0},
        {"seed": seed, "variant": "H0_intragenic_architecture_EB", "oof_macro_f1": arch_score, "oof_accuracy": arch_acc, "feature_count": float(folds[folds.variant.eq("H0_intragenic_architecture_EB")].feature_count.mean()), "convergence_warning_count": warnings_arch, "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": arch_score - h0_score},
    ])
    class_rows = []
    for label in classes:
        class_rows.append({"seed": seed, "class": str(label), "support": int((all_labels == label).sum()), "H0_f1": f1_score(all_labels == label, h0_prediction == label, zero_division=0), "architecture_f1": f1_score(all_labels == label, arch_prediction == label, zero_division=0)})
    class_frame = pd.DataFrame(class_rows); class_frame["delta"] = class_frame.architecture_f1 - class_frame.H0_f1
    oof_frame = pd.DataFrame({"seed": seed, "true_class": all_labels, **{f"h0__{label}": h0_oof[:, index] for index, label in enumerate(classes)}, **{f"architecture__{label}": architecture_oof[:, index] for index, label in enumerate(classes)}})
    audit = {"seed": seed, "outer_splits": 5, "architecture_eb_inner_splits": 5, "test_read": False, "train_test_concat": False, "fixed_class_gene_mutation_rules": False, "architecture_vocabulary_source": "outer_fold_train_only", "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": perf_counter() - started, **verdict}
    return summary, folds, pd.DataFrame(audit_rows), class_frame, {"audit": audit, "oof": oof_frame}


def run(run_id: str, seeds: tuple[int, ...]) -> None:
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    all_labels = train.SUBCLASS.to_numpy(); classes = np.asarray(sorted(np.unique(all_labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    print("[architecture-eb] train-only architecture token cache", flush=True)
    token_sets = architecture_token_sets(train[genes], genes)
    summaries, folds, audits, class_metrics, oofs, seed_audits = [], [], [], [], [], []
    for seed in seeds:
        summary, fold_metrics, audit_rows, classes_frame, payload = run_one_seed(train, genes, all_labels, classes, token_sets, seed)
        summaries.append(summary); folds.append(fold_metrics); audits.append(audit_rows); class_metrics.append(classes_frame); oofs.append(payload["oof"]); seed_audits.append(payload["audit"])
    result = Path(__file__).parent.parent / "result"; result.mkdir(exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True); folds = pd.concat(folds, ignore_index=True); class_frame = pd.concat(class_metrics, ignore_index=True); audit_frame = pd.concat(audits, ignore_index=True); oof_frame = pd.concat(oofs, ignore_index=True)
    if not RESULT_COLUMNS["summary"].issubset(summary.columns) or not RESULT_COLUMNS["fold"].issubset(folds.columns):
        raise AssertionError("result schema failure")
    aggregate = summary.groupby("variant", as_index=False).agg(seed_count=("seed", "nunique"), oof_macro_f1_mean=("oof_macro_f1", "mean"), oof_macro_f1_std=("oof_macro_f1", "std"), delta_vs_h0_mean=("delta_vs_h0", "mean"), delta_vs_h0_std=("delta_vs_h0", "std"), convergence_warning_count=("convergence_warning_count", "sum"), leakage_check=("leakage_check", "all"), nan_as_mutation_count=("nan_as_mutation_count", "max"))
    candidate_folds = folds[folds.variant.eq("H0_intragenic_architecture_EB")].sort_values(["seed", "fold"]).macro_f1.to_numpy()
    baseline_folds = folds[folds.variant.eq("H0")].sort_values(["seed", "fold"]).macro_f1.to_numpy()
    delta = candidate_folds - baseline_folds
    final_decision = "accepted" if bool(np.all(summary[summary.variant.eq("H0_intragenic_architecture_EB")].delta_vs_h0 > 0)) and float(delta.mean()) >= .008 and int((delta > 0).sum()) >= 11 else "rejected_or_not_detected"
    decision = {"seeds": list(seeds), "test_read": False, "train_test_concat": False, "fixed_class_gene_mutation_rules": False, "leakage_check": True, "nan_as_mutation_count": 0, "positive_fold_count": int((delta > 0).sum()), "mean_fold_delta": float(delta.mean()), "all_seed_delta_positive": bool(np.all(summary[summary.variant.eq("H0_intragenic_architecture_EB")].delta_vs_h0 > 0)), "decision": final_decision, "seed_audits": seed_audits}
    summary.to_csv(result / f"{run_id}_seed_summary.csv", index=False)
    aggregate.to_csv(result / f"{run_id}_3seed_summary.csv", index=False)
    folds.to_csv(result / f"{run_id}_fold_metrics.csv", index=False)
    audit_frame.to_csv(result / f"{run_id}_architecture_audit.csv", index=False)
    class_frame.to_csv(result / f"{run_id}_class_metrics.csv", index=False)
    oof_frame.to_csv(result / f"{run_id}_oof_probabilities.csv", index=False)
    (result / f"{run_id}_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = folds.pivot_table(index=["seed", "fold"], columns="variant", values="macro_f1").plot(marker="o", title="H0 vs intragenic architecture EB")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    ax = class_frame.groupby("class").delta.mean().sort_values().plot.barh(title="Architecture EB mean class F1 delta vs H0")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-intragenic-architecture-eb-01")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id, tuple(args.seeds))


if __name__ == "__main__":
    main()
