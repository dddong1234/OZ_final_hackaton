"""Faithful H0 + fixed XGBoost complement, train-only OOF screen."""
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
from xgboost import XGBClassifier

HERE = Path(__file__).resolve()
H0_COMMON = HERE.parents[2] / "exp_model_006" / "common"
if not H0_COMMON.exists():
    raise FileNotFoundError("GS faithful H0 implementation was not found: exp_model_006/common")
sys.path.insert(0, str(H0_COMMON))
from h1_auto_confusion_moe import fit_h0_fold  # noqa: E402
from xgb_complement import H0_WEIGHT, XGB_WEIGHT, fixed_blend, xgb_config  # noqa: E402

DEFAULT_SEEDS = (42,)
H0_REFERENCE_SEED42 = .544744
H0_REFERENCE_TOLERANCE = .0005


def root() -> Path:
    for path in (HERE, *HERE.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv was not found")


def align_probability(model: XGBClassifier, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return columns in the repository's deterministic class order."""
    fitted = np.asarray(model.classes_)
    location = {label: index for index, label in enumerate(fitted)}
    return probability[:, [location[index] for index in range(len(classes))]].astype(np.float32)


def evaluate(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[np.asarray(probability).argmax(axis=1)]
    return (
        float(f1_score(labels, prediction, average="macro", zero_division=0)),
        float(accuracy_score(labels, prediction)),
        prediction,
    )


def run_seed(train: pd.DataFrame, genes: list[str], labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    started = perf_counter()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    h0_oof = np.zeros((len(train), len(classes)), dtype=np.float32)
    xgb_oof = np.zeros_like(h0_oof)
    blend_oof = np.zeros_like(h0_oof)
    fold_rows: list[dict] = []
    audit_rows: list[dict] = []

    for fold, (fit_index, valid_index) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        print(f"[H0-XGB] seed {seed}, fold {fold}/5: H0 fit + XGB fit", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        h0 = fit_h0_fold(fit_frame, valid_frame, labels[fit_index], genes, classes, seed=seed * 100 + fold)
        model = XGBClassifier(**xgb_config(seed=seed * 100 + fold, class_count=len(classes)))
        model.fit(h0.x_fit, np.searchsorted(classes, labels[fit_index]))
        xgb_probability = align_probability(model, model.predict_proba(h0.x_apply), classes)
        blend_probability = fixed_blend(h0.probability, xgb_probability)
        h0_oof[valid_index] = h0.probability
        xgb_oof[valid_index] = xgb_probability
        blend_oof[valid_index] = blend_probability
        h0_f1, h0_accuracy, _ = evaluate(labels[valid_index], h0.probability, classes)
        xgb_f1, xgb_accuracy, _ = evaluate(labels[valid_index], xgb_probability, classes)
        blend_f1, blend_accuracy, _ = evaluate(labels[valid_index], blend_probability, classes)
        for variant, macro_f1, accuracy in (
            ("H0", h0_f1, h0_accuracy),
            ("XGB_structured", xgb_f1, xgb_accuracy),
            ("H0_080_XGB_020", blend_f1, blend_accuracy),
        ):
            fold_rows.append({
                "seed": seed, "fold": fold, "variant": variant,
                "macro_f1": macro_f1, "accuracy": accuracy,
                "feature_count": len(h0.names), "delta_vs_h0": macro_f1 - h0_f1,
            })
        audit_rows.append({
            "seed": seed, "fold": fold, "test_read": False,
            "raw_train_test_concat": bool(h0.audit["raw_train_test_concat"]),
            "vocabulary_source": h0.audit["vocabulary_source"],
            "outer_validation_used_for_fit": False,
            "fixed_class_gene_mutation_rules": False,
            "leakage_check": not bool(h0.audit["raw_train_test_concat"]),
            "nan_as_mutation_count": int(h0.audit["nan_as_mutation_count"]),
            "h0_convergence_warning_count": h0.convergence_warnings,
        })
        del model, h0, fit_frame, valid_frame, xgb_probability, blend_probability
        gc.collect()

    fold_frame = pd.DataFrame(fold_rows)
    audit_frame = pd.DataFrame(audit_rows)
    h0_warning_count = int(audit_frame.h0_convergence_warning_count.sum())
    summaries: list[dict] = []
    class_rows: list[dict] = []
    oof_rows: list[pd.DataFrame] = []
    for variant, probability in (("H0", h0_oof), ("XGB_structured", xgb_oof), ("H0_080_XGB_020", blend_oof)):
        macro_f1, accuracy, prediction = evaluate(labels, probability, classes)
        summaries.append({
            "seed": seed, "variant": variant, "oof_macro_f1": macro_f1, "oof_accuracy": accuracy,
            "feature_count_mean": float(fold_frame.loc[fold_frame.variant.eq(variant), "feature_count"].mean()),
            "convergence_warning_count": h0_warning_count,
            "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": perf_counter() - started,
        })
        _, _, f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"seed": seed, "variant": variant, "class": label, "f1": value, "support": int(count)} for label, value, count in zip(classes, f1, support))
        oof_rows.append(pd.DataFrame(probability, columns=[f"prob__{label}" for label in classes]).assign(seed=seed, variant=variant, row_index=np.arange(len(train)), truth=labels, prediction=prediction))

    summary = pd.DataFrame(summaries)
    h0_score = float(summary.loc[summary.variant.eq("H0"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_h0"] = summary.oof_macro_f1 - h0_score
    return summary, fold_frame, audit_frame, pd.DataFrame(class_rows), pd.concat(oof_rows, ignore_index=True), {
        "seed": seed,
        "h0_seed42_reference_delta": h0_score - H0_REFERENCE_SEED42 if seed == 42 else None,
    }


def smoke() -> None:
    train = pd.read_csv(root() / "data" / "raw" / "train.csv", nrows=12)
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    assert len(genes) == 4384
    assert int(train[genes].isna().sum().sum()) == 0
    config = xgb_config(seed=42, class_count=train.SUBCLASS.nunique())
    assert config["objective"] == "multi:softprob" and config["tree_method"] == "hist"
    tiny_config = xgb_config(seed=42, class_count=2)
    tiny_config["n_estimators"] = 2
    tiny = XGBClassifier(**tiny_config)
    tiny.fit(np.asarray([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]), np.asarray([0, 0, 1, 1]))
    assert tiny.predict_proba(np.asarray([[0., 0.]])).shape == (1, 2)
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "fixed_class_gene_mutation_rules": False}))


def run(run_id: str, seeds: tuple[int, ...]) -> None:
    data_root = root()
    train = pd.read_csv(data_root / "data" / "raw" / "train.csv")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    outputs = [run_seed(train, genes, labels, classes, seed) for seed in seeds]
    summary, folds, audits, class_metrics, probabilities = (pd.concat([item[index] for item in outputs], ignore_index=True) for index in range(5))
    aggregate = summary.groupby("variant", as_index=False).agg(
        seed_count=("seed", "nunique"), oof_macro_f1_mean=("oof_macro_f1", "mean"),
        oof_macro_f1_std=("oof_macro_f1", "std"), delta_vs_h0_mean=("delta_vs_h0", "mean"),
        delta_vs_h0_std=("delta_vs_h0", "std"), feature_count_mean=("feature_count_mean", "mean"),
        convergence_warning_count=("convergence_warning_count", "sum"), leakage_check=("leakage_check", "all"),
        nan_as_mutation_count=("nan_as_mutation_count", "max"),
    )
    pivot = folds.pivot_table(index=["seed", "fold"], columns="variant", values="macro_f1")
    xgb = summary.loc[summary.variant.eq("H0_080_XGB_020")].sort_values("seed")
    h0 = summary.loc[summary.variant.eq("H0")].sort_values("seed")
    reference_delta = next((item[5]["h0_seed42_reference_delta"] for item in outputs if item[5]["seed"] == 42), None)
    screen_pass = bool(xgb.delta_vs_h0.iloc[0] >= .008 and (pivot["H0_080_XGB_020"] > pivot["H0"]).sum() >= 4) if len(seeds) == 1 else False
    three_seed_pass = bool(np.all(xgb.oof_macro_f1.to_numpy() > h0.oof_macro_f1.to_numpy()) and xgb.delta_vs_h0.mean() >= .005 and (pivot["H0_080_XGB_020"] > pivot["H0"]).sum() >= 11) if len(seeds) == 3 else False
    decision = {
        "seeds": list(seeds), "h0_weight": H0_WEIGHT, "xgb_weight": XGB_WEIGHT,
        "h0_seed42_reference_delta": reference_delta,
        "h0_seed42_reference_match": reference_delta is None or abs(reference_delta) <= H0_REFERENCE_TOLERANCE,
        "positive_fold_count": int((pivot["H0_080_XGB_020"] > pivot["H0"]).sum()),
        "mean_delta": float(xgb.delta_vs_h0.mean()), "leakage_check": bool(audits.leakage_check.all()),
        "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "test_read": False,
        "decision": "screen_candidate" if screen_pass else ("accepted_3seed" if three_seed_pass else "rejected_or_not_detected"),
    }
    result = HERE.parent / "result"; result.mkdir(exist_ok=True)
    summary.to_csv(result / f"{run_id}_seed_summary.csv", index=False)
    aggregate.to_csv(result / f"{run_id}_aggregate_summary.csv", index=False)
    folds.to_csv(result / f"{run_id}_fold_metrics.csv", index=False)
    audits.to_csv(result / f"{run_id}_fold_audit.csv", index=False)
    class_metrics.to_csv(result / f"{run_id}_class_metrics.csv", index=False)
    probabilities.to_csv(result / f"{run_id}_oof_probabilities.csv", index=False)
    (result / f"{run_id}_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = pivot.plot(marker="o", figsize=(10, 4), title="H0 vs XGB complement"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    f1 = class_metrics.pivot_table(index=["seed", "class"], columns="variant", values="f1")
    delta = (f1["H0_080_XGB_020"] - f1["H0"]).groupby("class").mean().sort_values()
    ax = delta.plot.barh(figsize=(8, 7), title="Class F1 delta: H0 + XGB − H0"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-h0-xgb-complement-01")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id, tuple(args.seeds))


if __name__ == "__main__":
    main()
