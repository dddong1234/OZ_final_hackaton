"""Checkpointed seed OOF screen: faithful H0 Selective-EB plus Complement NB.

OOF mode reads train.csv only.  Vocabulary and NB fitting use each outer-fold
train split only; validation is transformed and predicted only.
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
from sklearn.naive_bayes import ComplementNB

from h0_complement_nb_profile import H0_WEIGHT, NB_ALPHA, NB_WEIGHT, profile_blend

HERE = Path(__file__).resolve()
DEFAULT_SEEDS = (42,)
SCREEN_DELTA = 0.003


def _gs_common(model: str) -> Path:
    path = HERE.parents[2] / model / "common"
    if not path.exists():
        raise FileNotFoundError(f"GS dependency missing: {path}")
    return path


for dependency in (_gs_common("exp_model_006"), _gs_common("exp_model_007")):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from h0_faithful_pipeline import _aligned_probability, fit_vocabulary, normalise_cell, transform_rows  # noqa: E402
from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402


def project_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "data" / "raw" / "train.csv").exists():
            return candidate
    raise FileNotFoundError("data/raw/train.csv was not found")


def result_directory() -> Path:
    path = HERE.parent.parent / "result"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_contract() -> dict:
    return {
        "test_read": False,
        "raw_train_test_concat": False,
        "fixed_class_gene_exact_mutation_rules": False,
        "outer_validation_used_for_fit": False,
        "leakage_check": True,
        "nan_as_mutation_count": 0,
        "h0_weight": H0_WEIGHT,
        "nb_weight": NB_WEIGHT,
        "nb_alpha": NB_ALPHA,
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    arrays = {f"oof__{name}": np.asarray(value) for name, value in payload["oof"].items()}
    metadata = json.dumps({key: payload[key] for key in ("completed_folds", "fold_rows", "audit_rows")})
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata_json=np.asarray(metadata), **arrays)
    temporary.replace(path)
    path.with_suffix(".progress.json").write_text(json.dumps({"completed_folds": payload["completed_folds"]}, indent=2), encoding="utf-8")


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        oof = {name.removeprefix("oof__"): archive[name].copy() for name in archive.files if name.startswith("oof__")}
    return {**metadata, "oof": oof}


def _nb_probability(fit_frame: pd.DataFrame, apply_frame: pd.DataFrame, labels: np.ndarray, genes: list[str], classes: np.ndarray) -> tuple[np.ndarray, int]:
    """Fit the profile model on fold-train binary genes only."""
    vocabulary = fit_vocabulary(fit_frame, genes)
    fit = transform_rows(fit_frame, genes, vocabulary).mutation
    apply = transform_rows(apply_frame, genes, vocabulary).mutation
    active = np.flatnonzero(np.asarray(fit.getnnz(axis=0)).ravel() > 0)
    if not len(active):
        raise AssertionError("outer train has no mutation features")
    model = ComplementNB(alpha=NB_ALPHA, norm=True)
    model.fit(fit[:, active], labels)
    return _aligned_probability(model, model.predict_proba(apply[:, active]), classes).astype(np.float32), int(len(active))


def _metrics(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[np.asarray(probability).argmax(axis=1)]
    return float(f1_score(labels, prediction, average="macro", zero_division=0)), float(accuracy_score(labels, prediction)), prediction


def _write_outputs(result: Path, run_id: str, seed: int, labels: np.ndarray, classes: np.ndarray, oof: dict[str, np.ndarray], folds: pd.DataFrame, audits: pd.DataFrame, started: float) -> pd.DataFrame:
    summary_rows, class_rows = [], []
    for variant, probability, count in (("H0_selective_EB", oof["h0"], "h0"), ("Complement_NB", oof["nb"], "nb"), ("H0_plus_Complement_NB", oof["blend"], "blend")):
        macro_f1, accuracy, prediction = _metrics(labels, probability, classes)
        summary_rows.append({"seed": seed, "variant": variant, "oof_macro_f1": macro_f1, "oof_accuracy": accuracy, "feature_count_mean": float(folds.loc[folds.variant.eq(variant), "feature_count"].mean()), "convergence_warning_count": int(audits.h0_convergence_warning_count.sum() + audits.eb_convergence_warning_count.sum()), "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started})
        precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"seed": seed, "variant": variant, "class": label, "precision": float(p), "recall": float(r), "f1": float(score), "support": int(n)} for label, p, r, score, n in zip(classes, precision, recall, f1, support))
    summary = pd.DataFrame(summary_rows)
    h0_score = float(summary.loc[summary.variant.eq("H0_selective_EB"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_h0"] = summary.oof_macro_f1 - h0_score
    class_frame = pd.DataFrame(class_rows)
    h0_class = class_frame.loc[class_frame.variant.eq("H0_selective_EB")].set_index("class")
    blend_class = class_frame.loc[class_frame.variant.eq("H0_plus_Complement_NB")].set_index("class")
    class_metrics = pd.DataFrame({"class": classes, "support": [int(h0_class.loc[label, "support"]) for label in classes], "h0_f1": [float(h0_class.loc[label, "f1"]) for label in classes], "blend_f1": [float(blend_class.loc[label, "f1"]) for label in classes]})
    class_metrics["delta_f1"] = class_metrics.blend_f1 - class_metrics.h0_f1
    prefix = result / f"{run_id}_seed{seed}"
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    folds.to_csv(prefix.with_name(prefix.name + "_fold_metrics.csv"), index=False)
    class_metrics.to_csv(prefix.with_name(prefix.name + "_class_metrics.csv"), index=False)
    audits.to_csv(prefix.with_name(prefix.name + "_fold_audit.csv"), index=False)
    pd.DataFrame({"row_index": np.arange(len(labels)), "truth": labels, **{f"h0__{label}": oof["h0"][:, index] for index, label in enumerate(classes)}, **{f"nb__{label}": oof["nb"][:, index] for index, label in enumerate(classes)}, **{f"blend__{label}": oof["blend"][:, index] for index, label in enumerate(classes)}}).to_csv(prefix.with_name(prefix.name + "_oof_probabilities.csv"), index=False)
    fold_pivot = folds.pivot(index="fold", columns="variant", values="macro_f1")
    ax = fold_pivot.plot(marker="o", title="H0 vs Complement NB profile blend"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_fold_macro_f1.png"), dpi=160); plt.close(ax.figure)
    ax = class_metrics.sort_values("delta_f1").plot.barh(x="class", y="delta_f1", title="Complement NB blend: class F1 delta"); ax.axvline(0, color="black"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_class_f1_delta.png"), dpi=160); plt.close(ax.figure)
    decision = {**run_contract(), "run_id": run_id, "seed": seed, "h0_macro_f1": h0_score, "blend_macro_f1": float(summary.loc[summary.variant.eq("H0_plus_Complement_NB"), "oof_macro_f1"].iloc[0]), "screen_delta": float(summary.loc[summary.variant.eq("H0_plus_Complement_NB"), "delta_vs_h0"].iloc[0]), "positive_fold_count": int((fold_pivot["H0_plus_Complement_NB"] > fold_pivot["H0_selective_EB"]).sum()), "screen_candidate": bool(float(summary.loc[summary.variant.eq("H0_plus_Complement_NB"), "delta_vs_h0"].iloc[0]) >= SCREEN_DELTA and int((fold_pivot["H0_plus_Complement_NB"] > fold_pivot["H0_selective_EB"]).sum()) >= 4)}
    prefix.with_name(prefix.name + "_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return summary


def run_seed(train: pd.DataFrame, genes: list[str], labels: np.ndarray, classes: np.ndarray, seed: int, run_id: str) -> pd.DataFrame:
    started = perf_counter(); result = result_directory(); checkpoint_path = result / f"{run_id}_seed{seed}_checkpoint.npz"
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is None:
        oof = {name: np.zeros((len(train), len(classes)), dtype=np.float32) for name in ("h0", "nb", "blend")}; completed, fold_rows, audit_rows = set(), [], []
    else:
        oof, completed, fold_rows, audit_rows = checkpoint["oof"], set(checkpoint["completed_folds"]), checkpoint["fold_rows"], checkpoint["audit_rows"]
        print(f"[H0+NB] seed {seed}: resume completed folds {sorted(completed)}", flush=True)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (fit_index, valid_index) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        if fold in completed:
            continue
        print(f"[H0+NB] seed {seed}, fold {fold}/5: H0 and train-only profile NB", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True); valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        h0_result = fit_fold(fit_frame, valid_frame, labels[fit_index], genes, classes, seed=seed * 100 + fold)
        nb_probability, nb_feature_count = _nb_probability(fit_frame, valid_frame, labels[fit_index], genes, classes)
        blend = profile_blend(h0_result["candidate"], nb_probability)
        for name, probability, feature_count in (("H0_selective_EB", h0_result["candidate"], h0_result["candidate_feature_count"]), ("Complement_NB", nb_probability, nb_feature_count), ("H0_plus_Complement_NB", blend, h0_result["candidate_feature_count"] + nb_feature_count)):
            oof[{"H0_selective_EB": "h0", "Complement_NB": "nb", "H0_plus_Complement_NB": "blend"}[name]][valid_index] = probability
            macro, accuracy, _ = _metrics(labels[valid_index], probability, classes)
            fold_rows.append({"seed": seed, "fold": fold, "variant": name, "macro_f1": macro, "accuracy": accuracy, "feature_count": int(feature_count)})
        audit_rows.append({"seed": seed, "fold": fold, **run_contract(), "inner_eb_crossfit": True, "outer_validation_used_for_fit": False, "h0_convergence_warning_count": int(h0_result["h0_warning"]), "eb_convergence_warning_count": int(h0_result["eb_warning"]), "nb_vocabulary_source": "outer_fold_train_only", "nb_feature_count": nb_feature_count, "auto_specialist_pairs": repr(h0_result["pairs"])})
        completed.add(fold); _save_checkpoint(checkpoint_path, {"completed_folds": list(completed), "fold_rows": fold_rows, "audit_rows": audit_rows, "oof": oof})
        print(f"[H0+NB] seed {seed}, fold {fold}/5 checkpoint saved", flush=True)
        del h0_result, nb_probability, blend; gc.collect()
    audits = pd.DataFrame(audit_rows)
    if not bool(audits.leakage_check.all()) or int(audits.nan_as_mutation_count.max()) != 0:
        raise AssertionError("leakage/NaN audit failed")
    return _write_outputs(result, run_id, seed, labels, classes, oof, pd.DataFrame(fold_rows), audits, started)


def smoke() -> None:
    root = project_root(); train = pd.read_csv(root / "data" / "raw" / "train.csv", nrows=64); genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    assert int(train[genes].isna().sum().sum()) == 0
    assert normalise_cell(np.nan) == () and normalise_cell("WT") == () and normalise_cell("") == ()
    labels = train.SUBCLASS.to_numpy(); classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    # Uses a compact deterministic slice solely to test parser/vocabulary/NB interfaces.
    fit_frame, apply_frame = train.iloc[:48][genes], train.iloc[48:][genes]
    probability, feature_count = _nb_probability(fit_frame, apply_frame, labels[:48], genes, classes)
    assert probability.shape == (16, len(classes)) and feature_count > 0
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    print(json.dumps({"smoke": "ok", **run_contract(), "test_read": False}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", default="exp-h0-complement-nb-profile-blend-01"); parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS)); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke(); return
    root = project_root(); train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]; labels = train.SUBCLASS.to_numpy(); classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    outputs = [run_seed(train, genes, labels, classes, seed, args.run_id) for seed in args.seeds]
    combined = pd.concat(outputs, ignore_index=True)
    combined.to_csv(result_directory() / f"{args.run_id}_seed_summary.csv", index=False)
    aggregate = combined.groupby("variant", as_index=False).agg(seed_count=("seed", "nunique"), oof_macro_f1_mean=("oof_macro_f1", "mean"), oof_macro_f1_std=("oof_macro_f1", "std"), delta_vs_h0_mean=("delta_vs_h0", "mean"), delta_vs_h0_std=("delta_vs_h0", "std"), convergence_warning_count=("convergence_warning_count", "sum"), leakage_check=("leakage_check", "all"), nan_as_mutation_count=("nan_as_mutation_count", "max"))
    aggregate.to_csv(result_directory() / f"{args.run_id}_aggregate_summary.csv", index=False)


if __name__ == "__main__":
    main()
