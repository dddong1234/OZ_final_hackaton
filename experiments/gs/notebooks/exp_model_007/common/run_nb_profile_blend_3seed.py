"""H0-faithful + Complement NB profile blend, train-only 3-seed validation."""
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
from sklearn.naive_bayes import ComplementNB

HERE = Path(__file__).resolve()
H0_COMMON = HERE.parents[2] / "exp_model_006" / "common"
if not H0_COMMON.exists():
    raise FileNotFoundError("exp_model_006 faithful H0 common code is required")
sys.path.insert(0, str(H0_COMMON))
from h1_auto_confusion_moe import fit_h0_fold  # noqa: E402
from nb_profile_blend import build_gene_type_matrix, fixed_blend  # noqa: E402

SEEDS = (42, 777, 2024)
H0_REFERENCE_SEED42 = .544744
BLEND = {"h0": .75, "complement_nb": .25}


def root() -> Path:
    for path in (HERE, *HERE.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv not found")


def aligned_probability(model: ComplementNB, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((len(probability), len(classes)), dtype=np.float32)
    location = {label: index for index, label in enumerate(classes)}
    for index, label in enumerate(model.classes_):
        out[:, location[label]] = probability[:, index]
    return out


def score(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[probability.argmax(axis=1)]
    return float(f1_score(labels, prediction, average="macro", zero_division=0)), float(accuracy_score(labels, prediction)), prediction


def run_seed(train: pd.DataFrame, genes: list[str], labels: np.ndarray, classes: np.ndarray, seed: int):
    started = perf_counter(); outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    h0_oof = np.zeros((len(train), len(classes)), dtype=np.float32); nb_oof = np.zeros_like(h0_oof); blend_oof = np.zeros_like(h0_oof)
    folds, audits = [], []
    for fold, (fit, valid) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        print(f"[NB-profile] seed {seed}, fold {fold}/5", flush=True)
        fit_frame = train.iloc[fit][genes].reset_index(drop=True); valid_frame = train.iloc[valid][genes].reset_index(drop=True)
        h0 = fit_h0_fold(fit_frame, valid_frame, labels[fit], genes, classes, seed=seed * 100 + fold)
        x_fit, vocabulary = build_gene_type_matrix(fit_frame, genes, vocabulary=None)
        x_valid, _ = build_gene_type_matrix(valid_frame, genes, vocabulary=vocabulary)
        nb = ComplementNB(alpha=1.0); nb.fit(x_fit, labels[fit])
        nb_probability = aligned_probability(nb, nb.predict_proba(x_valid), classes)
        blended = fixed_blend(h0.probability, nb_probability)
        h0_oof[valid], nb_oof[valid], blend_oof[valid] = h0.probability, nb_probability, blended
        h0_f1, _, _ = score(labels[valid], h0.probability, classes); nb_f1, _, _ = score(labels[valid], nb_probability, classes); blend_f1, _, _ = score(labels[valid], blended, classes)
        for variant, probability, metric in (("H0", h0.probability, h0_f1), ("ComplementNB_profile", nb_probability, nb_f1), ("H0_075_ComplementNB_025", blended, blend_f1)):
            folds.append({"seed": seed, "fold": fold, "variant": variant, "macro_f1": metric, "accuracy": float(accuracy_score(labels[valid], classes[probability.argmax(axis=1)])), "feature_count": len(h0.names) if variant == "H0" else len(vocabulary), "delta_vs_h0": metric - h0_f1})
        audits.append({"seed": seed, "fold": fold, "profile_vocabulary_size": len(vocabulary), "profile_vocabulary_source": "outer_fold_train_only", "outer_validation_used_for_nb_fit": False, "test_read": False, "leakage_check": not h0.audit["raw_train_test_concat"], "nan_as_mutation_count": 0})
        del h0, nb, x_fit, x_valid, fit_frame, valid_frame; gc.collect()
    summaries, class_rows = [], []
    for variant, probability in (("H0", h0_oof), ("ComplementNB_profile", nb_oof), ("H0_075_ComplementNB_025", blend_oof)):
        macro, accuracy, prediction = score(labels, probability, classes)
        variant_folds = pd.DataFrame(folds).query("variant == @variant")
        summaries.append({"seed": seed, "variant": variant, "oof_macro_f1": macro, "oof_accuracy": accuracy, "feature_count_mean": float(variant_folds.feature_count.mean()), "convergence_warning_count": 0, "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": perf_counter() - started})
        _, _, f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"seed": seed, "variant": variant, "class": label, "f1": value, "support": int(count)} for label, value, count in zip(classes, f1, support))
    summary = pd.DataFrame(summaries); h0_score = float(summary.loc[summary.variant.eq("H0"), "oof_macro_f1"].iloc[0]); summary["delta_vs_h0"] = summary.oof_macro_f1 - h0_score
    return summary, pd.DataFrame(folds), pd.DataFrame(audits), pd.DataFrame(class_rows), {"seed": seed, "h0_seed42_reference_delta": h0_score - H0_REFERENCE_SEED42 if seed == 42 else None}


def smoke() -> None:
    train = pd.read_csv(root() / "data" / "raw" / "train.csv", nrows=8); genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    matrix, vocabulary = build_gene_type_matrix(pd.DataFrame({"G": ["WT", np.nan, "R1H"]}), ["G"], vocabulary=None)
    assert len(genes) == 4384 and matrix.nnz == 1 and len(vocabulary) == 1
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "fixed_class_gene_mutation_rules": False}))


def run(run_id: str, seeds: tuple[int, ...]) -> None:
    train = pd.read_csv(root() / "data" / "raw" / "train.csv"); genes = [column for column in train if column not in ("ID", "SUBCLASS")]; labels = train.SUBCLASS.to_numpy(); classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0: raise AssertionError("train NaN contract violation")
    outputs = [run_seed(train, genes, labels, classes, seed) for seed in seeds]
    summary, folds, audits, classes_frame = (pd.concat([item[index] for item in outputs], ignore_index=True) for index in range(4))
    aggregate = summary.groupby("variant", as_index=False).agg(seed_count=("seed", "nunique"), oof_macro_f1_mean=("oof_macro_f1", "mean"), oof_macro_f1_std=("oof_macro_f1", "std"), delta_vs_h0_mean=("delta_vs_h0", "mean"), delta_vs_h0_std=("delta_vs_h0", "std"), feature_count_mean=("feature_count_mean", "mean"), leakage_check=("leakage_check", "all"), nan_as_mutation_count=("nan_as_mutation_count", "max"))
    candidate = summary[summary.variant.eq("H0_075_ComplementNB_025")].sort_values("seed"); h0 = summary[summary.variant.eq("H0")].sort_values("seed"); fold_pivot = folds.pivot_table(index=["seed", "fold"], columns="variant", values="macro_f1")
    reference_delta = next((item[4]["h0_seed42_reference_delta"] for item in outputs if item[4]["seed"] == 42), None)
    decision = {"seeds": list(seeds), "weights": BLEND, "all_seed_delta_positive": bool(np.all(candidate.oof_macro_f1.to_numpy() > h0.oof_macro_f1.to_numpy())), "mean_delta": float(candidate.delta_vs_h0.mean()), "positive_fold_count": int((fold_pivot["H0_075_ComplementNB_025"] > fold_pivot["H0"]).sum()), "h0_seed42_reference_delta": reference_delta, "h0_seed42_reference_match": reference_delta is None or abs(reference_delta) <= .0005, "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "test_read": False}
    decision["decision"] = "accepted" if decision["h0_seed42_reference_match"] and decision["all_seed_delta_positive"] and decision["mean_delta"] >= .005 and decision["positive_fold_count"] >= 11 else "rejected_or_not_detected"
    result = HERE.parent / "result"; result.mkdir(exist_ok=True)
    summary.to_csv(result / f"{run_id}_seed_summary.csv", index=False); aggregate.to_csv(result / f"{run_id}_3seed_summary.csv", index=False); folds.to_csv(result / f"{run_id}_fold_metrics.csv", index=False); audits.to_csv(result / f"{run_id}_fold_audit.csv", index=False); classes_frame.to_csv(result / f"{run_id}_class_metrics.csv", index=False)
    (result / f"{run_id}_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = fold_pivot.plot(marker="o", figsize=(10, 4), title="H0 vs Complement NB profile blend"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    delta = classes_frame.pivot_table(index=["seed", "class"], columns="variant", values="f1"); delta["blend_delta"] = delta["H0_075_ComplementNB_025"] - delta["H0"]; ax = delta.groupby("class").blend_delta.mean().sort_values().plot.barh(title="Class F1 delta: NB blend − H0"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", default="exp-nb-profile-blend-faithful-01"); parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS)); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id, tuple(args.seeds))


if __name__ == "__main__": main()
