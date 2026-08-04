"""Seed42 safe 3-way ensemble screen; train-only by construction."""
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

from h1_auto_confusion_moe import fit_h0_fold
from safe_3way_ensemble import WEIGHTS, align_probability, fixed_three_way_probability


SEED = 42
H0_REFERENCE = .544744
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


def _base_lgbm(seed: int, class_count: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass", boosting_type="gbdt", num_class=class_count,
        n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10,
        min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0,
        class_weight="balanced", random_state=seed, n_jobs=-1,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )


def _decision(h0_score: float, ensemble_score: float, fold_delta: np.ndarray) -> dict:
    delta = float(ensemble_score - h0_score)
    positive = int((fold_delta > 0).sum())
    if delta >= .015 and positive >= 4:
        label = "strong_validation_candidate"
    elif delta >= .005 and positive >= 4:
        label = "validation_candidate"
    elif delta > 0:
        label = "not_detected"
    else:
        label = "rejected"
    return {"decision": label, "delta_vs_h0": delta, "positive_fold_count": positive, "h0_reference": H0_REFERENCE, "h0_reference_delta": float(h0_score - H0_REFERENCE)}


def smoke() -> None:
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv", nrows=8)
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    assert len(genes) == 4384
    assert int(train[genes].isna().sum().sum()) == 0
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "weights": WEIGHTS}))


def run(run_id: str) -> None:
    started = perf_counter()
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    probability = {name: np.zeros((len(train), len(classes)), dtype=np.float64) for name in ("H0", "multinomial_LR", "OVR_LR", "base_LGBM", "safe_3way")}
    fold_rows: list[dict] = []
    warnings_total = {name: 0 for name in probability}
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_index, valid_index) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        print(f"[safe-3way] outer fold {fold}/5: feature fit and model training", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        y_fit = labels[fit_index]
        h0 = fit_h0_fold(fit_frame, valid_frame, y_fit, genes, classes, seed=SEED * 100 + fold)
        if h0.audit["raw_train_test_concat"]:
            raise AssertionError("train/test concatenation contract violation")
        multinomial = LogisticRegression(solver="lbfgs", C=.07, max_iter=2000, class_weight="balanced", random_state=42)
        ovr = OneVsRestClassifier(LogisticRegression(solver="lbfgs", C=.07, max_iter=2000, class_weight="balanced", random_state=42), n_jobs=-1)
        lgbm = _base_lgbm(42, len(classes))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            multinomial.fit(h0.x_fit, y_fit)
            ovr.fit(h0.x_fit, y_fit)
        local_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        lgbm.fit(h0.x_fit, y_fit)
        p_multi = align_probability(multinomial, multinomial.predict_proba(h0.x_apply), classes)
        p_ovr = align_probability(ovr, ovr.predict_proba(h0.x_apply), classes)
        p_lgbm = align_probability(lgbm, lgbm.predict_proba(h0.x_apply), classes)
        p_safe = fixed_three_way_probability(p_multi, p_ovr, p_lgbm)
        local = {"H0": h0.probability, "multinomial_LR": p_multi, "OVR_LR": p_ovr, "base_LGBM": p_lgbm, "safe_3way": p_safe}
        h0_fold, _, _ = _metrics(labels[valid_index], h0.probability, classes)
        for name, values in local.items():
            score, accuracy, _ = _metrics(labels[valid_index], values, classes)
            probability[name][valid_index] = values
            fold_rows.append({"fold": fold, "variant": name, "macro_f1": score, "accuracy": accuracy, "feature_count": len(h0.names), "delta_vs_h0": score - h0_fold})
            warnings_total[name] += h0.convergence_warnings if name == "H0" else local_warnings if name in ("multinomial_LR", "OVR_LR", "safe_3way") else 0
        del h0, multinomial, ovr, lgbm, p_multi, p_ovr, p_lgbm, p_safe, local, fit_frame, valid_frame
        gc.collect()
    folds = pd.DataFrame(fold_rows)
    scores: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    summary_rows: list[dict] = []
    feature_count = float(folds.feature_count.mean())
    for name, values in probability.items():
        score, accuracy, prediction = _metrics(labels, values, classes)
        scores[name], predictions[name] = score, prediction
        summary_rows.append({"variant": name, "oof_macro_f1": score, "oof_accuracy": accuracy, "feature_count": feature_count, "convergence_warning_count": warnings_total[name], "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": score - scores.get("H0", score)})
    summary = pd.DataFrame(summary_rows)
    h0_fold = folds[folds.variant.eq("H0")].sort_values("fold").macro_f1.to_numpy()
    e0_fold = folds[folds.variant.eq("safe_3way")].sort_values("fold").macro_f1.to_numpy()
    verdict = _decision(scores["H0"], scores["safe_3way"], e0_fold - h0_fold)
    class_rows = []
    for label in classes:
        class_rows.append({"class": str(label), "support": int((labels == label).sum()), "H0_f1": f1_score(labels == label, predictions["H0"] == label, zero_division=0), "safe_3way_f1": f1_score(labels == label, predictions["safe_3way"] == label, zero_division=0)})
    class_frame = pd.DataFrame(class_rows); class_frame["delta"] = class_frame.safe_3way_f1 - class_frame.H0_f1
    result = Path(__file__).parent.parent / "result"; result.mkdir(exist_ok=True)
    if not RESULT_COLUMNS["summary"].issubset(summary.columns) or not RESULT_COLUMNS["fold"].issubset(folds.columns):
        raise AssertionError("result schema failure")
    summary.to_csv(result / f"{run_id}_seed42_summary.csv", index=False)
    folds.to_csv(result / f"{run_id}_seed42_fold_metrics.csv", index=False)
    class_frame.to_csv(result / f"{run_id}_seed42_class_metrics.csv", index=False)
    pd.DataFrame({"true_class": labels, **{f"{name}__{label}": values[:, index] for name, values in probability.items() for index, label in enumerate(classes)}}).to_csv(result / f"{run_id}_seed42_oof_probabilities.csv", index=False)
    audit = {"seed": SEED, "outer_splits": 5, "weights": WEIGHTS, "test_read": False, "train_test_concat": False, "fixed_class_gene_mutation_rules": False, "feature_fit_scope": "outer_fold_train_only", "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": perf_counter() - started, **verdict}
    (result / f"{run_id}_seed42_leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = folds.pivot(index="fold", columns="variant", values="macro_f1").plot(marker="o", figsize=(9, 4), title="Safe 3-way vs H0 by fold")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    ax = class_frame.set_index("class").delta.sort_values().plot.barh(figsize=(7, 7), title="Safe 3-way class F1 delta vs H0")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-safe-3way-ensemble-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id)


if __name__ == "__main__":
    main()
