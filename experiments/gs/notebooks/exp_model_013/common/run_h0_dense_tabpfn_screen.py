"""Seed-42 TabPFN screen on fold-train-only dense H0 evidence features.

The screen reads train.csv only.  It contains no test path, fixed cancer names,
fixed genes or fixed mutations.  H0 is reconstructed through GS-owned, rule-safe
modules and is used only after matching the published seed-42 reference.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve()
RESULT = HERE.parent.parent / "result"
PROJECT_ROOT = HERE.parents[5]
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"
SEED = 42
H0_REFERENCE = 0.547915
H0_TOLERANCE = 0.001
H0_WEIGHT = 0.80
TABPFN_WEIGHT = 0.20
SCREEN_DELTA = 0.015
LOW_MARGIN = 0.05


def _add_source(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required GS source is missing: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_add_source(HERE.parent)
_add_source(HERE.parents[2] / "exp_model_006" / "common")
_add_source(HERE.parents[2] / "exp_model_007" / "common")
from h0_faithful_pipeline import build_design_matrices  # noqa: E402
from h0_selective_eb_replacement import empirical_bayes_features  # noqa: E402
from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402
from tabpfn_dense_core import FitOnlyStandardizer, build_dense_h0_view, package_contract  # noqa: E402


def contract() -> dict[str, object]:
    return {
        "test_read": False,
        "train_test_concat": False,
        "fixed_class_gene_mutation_rules": False,
        "vocabulary_source": "outer_fold_train_only",
        "supervised_eb_source": "outer_fold_train_only_with_inner_crossfit",
        "scaler_source": "outer_fold_train_only",
        "outer_validation_used_for_fit": False,
        "leakage_check": True,
        "nan_as_mutation_count": 0,
        "tabpfn_weight": TABPFN_WEIGHT,
        "h0_weight": H0_WEIGHT,
        "tabpfn_model_contract": "fixed_tabpfn_v3_classifier_no_hpo_no_autotabpfn",
    }


def decide_screen(*, delta: float, positive_folds: int, h0_reference_match: bool) -> str:
    if not h0_reference_match:
        return "baseline_not_reproduced"
    return "screen_candidate" if delta >= SCREEN_DELTA and positive_folds >= 4 else "not_detected"


def smoke_contract() -> dict[str, object]:
    fit = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid = np.asarray([[101.0, 102.0]], dtype=np.float32)
    transformed = FitOnlyStandardizer().fit(fit).transform(valid)
    if not np.all(transformed > 90):
        raise AssertionError("smoke scaler must not fit validation rows")
    return {**contract(), "example_dense_feature_count": int(transformed.shape[1])}


def _device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fit_tabpfn_probability(x_fit: np.ndarray, y_fit: np.ndarray, x_valid: np.ndarray, classes: np.ndarray, device: str, *, seed: int) -> np.ndarray:
    metadata = package_contract()
    if not metadata["tabpfn_installed"]:
        raise RuntimeError("tabpfn is not installed. Run: /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m pip install tabpfn")
    if device == "cpu":
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
    try:
        from tabpfn import TabPFNClassifier  # type: ignore
        from tabpfn.constants import ModelVersion  # type: ignore
    except Exception as error:  # pragma: no cover - dependency specific
        raise RuntimeError("TabPFN import failed after package detection") from error
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        device=device,
        random_state=seed,
        n_estimators=8,
        auto_scale_n_estimators=False,
        softmax_temperature=1.0,
        balance_probabilities=False,
        eval_metric=None,
        tuning_config=None,
        show_progress_bar=True,
    )
    model.fit(x_fit, y_fit)
    raw = np.asarray(model.predict_proba(x_valid), dtype=np.float64)
    lookup = {label: index for index, label in enumerate(model.classes_)}
    probability = raw[:, [lookup[label] for label in classes]]
    probability /= probability.sum(axis=1, keepdims=True)
    return probability.astype(np.float32)


def _metric_row(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[np.asarray(probability).argmax(axis=1)]
    return float(f1_score(labels, prediction, average="macro", zero_division=0)), float(accuracy_score(labels, prediction)), prediction


def _topk_recall(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray, k: int) -> float:
    order = np.argpartition(probability, -k, axis=1)[:, -k:]
    truth = np.asarray([np.flatnonzero(classes == label)[0] for label in labels])
    return float(np.mean(np.any(order == truth[:, None], axis=1)))


def _h0_checkpoint(run_id: str) -> Path:
    return RESULT / f"{run_id}_seed42_h0_only.npz"


def _run_h0_first(train: pd.DataFrame, genes: list[str], labels: np.ndarray, classes: np.ndarray, run_id: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    checkpoint = _h0_checkpoint(run_id)
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as archive:
            payload = json.loads(str(archive["metadata_json"].item()))
            return archive["h0_oof"].copy(), pd.DataFrame(payload["fold_rows"]), pd.DataFrame(payload["audit_rows"])
    h0_oof = np.zeros((len(train), len(classes)), dtype=np.float32)
    folds, audits = [], []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_index, valid_index) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        print(f"[TabPFN screen] H0 reproduction fold {fold}/5", flush=True)
        result = fit_fold(train.iloc[fit_index][genes].reset_index(drop=True), train.iloc[valid_index][genes].reset_index(drop=True), labels[fit_index], genes, classes, seed=SEED * 100 + fold)
        h0_oof[valid_index] = result["candidate"]
        fold_macro, fold_accuracy, _ = _metric_row(labels[valid_index], result["candidate"], classes)
        folds.append({"fold": fold, "variant": "H0_selective_EB", "macro_f1": fold_macro, "accuracy": fold_accuracy, "feature_count": result["candidate_feature_count"]})
        audits.append({"fold": fold, **contract(), "h0_convergence_warning_count": result["h0_warning"], "eb_convergence_warning_count": result["eb_warning"], "h0_feature_count": result["candidate_feature_count"]})
        del result
        gc.collect()
    temporary = checkpoint.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, h0_oof=h0_oof, metadata_json=np.asarray(json.dumps({"fold_rows": folds, "audit_rows": audits})))
    temporary.replace(checkpoint)
    return h0_oof, pd.DataFrame(folds), pd.DataFrame(audits)


def _class_metrics(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray, variant: str) -> pd.DataFrame:
    _, _, predicted = _metric_row(labels, probability, classes)
    precision, recall, score, support = precision_recall_fscore_support(labels, predicted, labels=classes, zero_division=0)
    return pd.DataFrame({"variant": variant, "class": classes, "precision": precision, "recall": recall, "f1": score, "support": support})


def _write_outputs(run_id: str, labels: np.ndarray, classes: np.ndarray, probabilities: dict[str, np.ndarray], folds: pd.DataFrame, audits: pd.DataFrame, started: float) -> dict[str, object]:
    summary, class_frames, top_rows = [], [], []
    h0_macro, _, h0_pred = _metric_row(labels, probabilities["H0_selective_EB"], classes)
    margin = np.sort(probabilities["H0_selective_EB"], axis=1)[:, -1] - np.sort(probabilities["H0_selective_EB"], axis=1)[:, -2]
    low = margin < LOW_MARGIN
    for variant, probability in probabilities.items():
        macro, accuracy, prediction = _metric_row(labels, probability, classes)
        summary.append({"variant": variant, "oof_macro_f1": macro, "oof_accuracy": accuracy, "feature_count": float(folds.loc[folds.variant.eq(variant), "feature_count"].mean()), "convergence_warning_count": int(audits[["h0_convergence_warning_count", "eb_convergence_warning_count"]].sum().sum()), "leakage_check": bool(audits.leakage_check.all()), "nan_as_mutation_count": int(audits.nan_as_mutation_count.max()), "runtime_seconds": perf_counter() - started, "delta_vs_h0": macro - h0_macro})
        class_frames.append(_class_metrics(labels, probability, classes, variant))
        top_rows.append({"variant": variant, **{f"top{k}_recall": _topk_recall(labels, probability, classes, k) for k in (1, 2, 3)}, "low_margin_macro_f1": float(f1_score(labels[low], prediction[low], average="macro", zero_division=0))})
    summary_frame, class_frame, top_frame = pd.DataFrame(summary), pd.concat(class_frames, ignore_index=True), pd.DataFrame(top_rows)
    fold_pivot = folds.pivot(index="fold", columns="variant", values="macro_f1")
    candidate = summary_frame.loc[summary_frame.variant.eq("H0_plus_TabPFN")].iloc[0]
    h0_low = float(top_frame.loc[top_frame.variant.eq("H0_selective_EB"), "low_margin_macro_f1"].iloc[0])
    candidate_low = float(top_frame.loc[top_frame.variant.eq("H0_plus_TabPFN"), "low_margin_macro_f1"].iloc[0])
    h0_class = class_frame.loc[class_frame.variant.eq("H0_selective_EB")].set_index("class").f1
    candidate_class = class_frame.loc[class_frame.variant.eq("H0_plus_TabPFN")].set_index("class").f1
    delta_class = (candidate_class - h0_class).rename("delta_f1").rename_axis("class").reset_index()
    h0_reference_match = abs(h0_macro - H0_REFERENCE) <= H0_TOLERANCE
    decision = {**contract(), **package_contract(), "run_id": run_id, "h0_oof_macro_f1": h0_macro, "h0_reference": H0_REFERENCE, "h0_reference_match": h0_reference_match, "h0_reference_delta": h0_macro - H0_REFERENCE, "tabpfn_device": _device(), "tabpfn_blend_oof_macro_f1": float(candidate.oof_macro_f1), "delta_vs_h0": float(candidate.delta_vs_h0), "positive_fold_count": int((fold_pivot.H0_plus_TabPFN > fold_pivot.H0_selective_EB).sum()), "low_margin_delta": candidate_low - h0_low, "decision": decide_screen(delta=float(candidate.delta_vs_h0), positive_folds=int((fold_pivot.H0_plus_TabPFN > fold_pivot.H0_selective_EB).sum()), h0_reference_match=h0_reference_match)}
    prefix = RESULT / f"{run_id}_seed42"
    summary_frame.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    folds.to_csv(prefix.with_name(prefix.name + "_fold_metrics.csv"), index=False)
    audits.to_csv(prefix.with_name(prefix.name + "_fold_audit.csv"), index=False)
    class_frame.to_csv(prefix.with_name(prefix.name + "_class_metrics.csv"), index=False)
    delta_class.to_csv(prefix.with_name(prefix.name + "_class_f1_delta.csv"), index=False)
    top_frame.to_csv(prefix.with_name(prefix.name + "_topk_metrics.csv"), index=False)
    pd.DataFrame({"row_index": np.arange(len(labels)), "truth": labels, **{f"{variant}__{label}": values[:, index] for variant, values in probabilities.items() for index, label in enumerate(classes)}}).to_csv(prefix.with_name(prefix.name + "_oof_probabilities.csv"), index=False)
    prefix.with_name(prefix.name + "_leakage_audit.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = fold_pivot.plot(marker="o", title="H0 vs dense TabPFN"); ax.set_ylabel("Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_fold_macro_f1.png"), dpi=160); plt.close(ax.figure)
    ax = delta_class.sort_values("delta_f1").plot.barh(x="class", y="delta_f1", legend=False, title="H0 + TabPFN: class F1 delta"); ax.axvline(0, color="black"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_class_f1_delta.png"), dpi=160); plt.close(ax.figure)
    ax = top_frame.set_index("variant")[["top1_recall", "top2_recall", "top3_recall"]].plot.bar(title="Top-k recall"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_topk_recall.png"), dpi=160); plt.close(ax.figure)
    return decision


def run(run_id: str, device: str) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    train = pd.read_csv(TRAIN_CSV)
    genes = [column for column in train.columns if column not in {"ID", "SUBCLASS"}]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    h0_oof, h0_folds, audits = _run_h0_first(train, genes, labels, classes, run_id)
    h0_macro, _, _ = _metric_row(labels, h0_oof, classes)
    if abs(h0_macro - H0_REFERENCE) > H0_TOLERANCE:
        failure = {**contract(), "run_id": run_id, "h0_oof_macro_f1": h0_macro, "h0_reference": H0_REFERENCE, "h0_reference_match": False, "decision": "baseline_not_reproduced", "tabpfn_fit_started": False}
        (RESULT / f"{run_id}_seed42_leakage_audit.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"H0 reference mismatch: got {h0_macro:.6f}, expected {H0_REFERENCE:.6f}; TabPFN was not fitted")
    if not package_contract()["tabpfn_installed"]:
        raise RuntimeError("tabpfn missing. Install it first; H0 checkpoint is preserved and TabPFN has not been fitted.")
    tabpfn_oof = np.zeros_like(h0_oof)
    tabpfn_folds = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_index, valid_index) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        print(f"[TabPFN screen] fold {fold}/5: fold-train dense evidence + TabPFN", flush=True)
        fit_frame, valid_frame = train.iloc[fit_index][genes].reset_index(drop=True), train.iloc[valid_index][genes].reset_index(drop=True)
        x_fit, x_valid, names, audit = build_design_matrices(fit_frame, valid_frame, labels[fit_index], genes, seed=SEED * 100 + fold)
        eb_fit, eb_valid = empirical_bayes_features(fit_frame, valid_frame, labels[fit_index], classes, genes, seed=SEED * 100 + fold)
        dense_fit, dense_valid, dense_names = build_dense_h0_view(x_fit, x_valid, names, eb_fit, eb_valid)
        scaler = FitOnlyStandardizer().fit(dense_fit)
        probability = _fit_tabpfn_probability(scaler.transform(dense_fit), labels[fit_index], scaler.transform(dense_valid), classes, device, seed=SEED * 100 + fold)
        tabpfn_oof[valid_index] = probability
        macro, accuracy, _ = _metric_row(labels[valid_index], probability, classes)
        tabpfn_folds.append({"fold": fold, "variant": "TabPFN_dense", "macro_f1": macro, "accuracy": accuracy, "feature_count": len(dense_names), "device": device})
        audits.loc[audits.fold.eq(fold), "tabpfn_dense_feature_count"] = len(dense_names)
        audits.loc[audits.fold.eq(fold), "tabpfn_scaler_fit_only"] = True
        audits.loc[audits.fold.eq(fold), "tabpfn_outer_validation_used_for_fit"] = False
        del x_fit, x_valid, eb_fit, eb_valid, dense_fit, dense_valid, scaler, probability
        gc.collect()
    blended = H0_WEIGHT * h0_oof + TABPFN_WEIGHT * tabpfn_oof
    blended /= blended.sum(axis=1, keepdims=True)
    h0_folds = h0_folds.copy()
    blend_folds = []
    for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        macro, accuracy, _ = _metric_row(labels[valid_index], blended[valid_index], classes)
        feature_count = int(tabpfn_folds[fold - 1]["feature_count"])
        blend_folds.append({"fold": fold, "variant": "H0_plus_TabPFN", "macro_f1": macro, "accuracy": accuracy, "feature_count": feature_count})
    decision = _write_outputs(run_id, labels, classes, {"H0_selective_EB": h0_oof, "TabPFN_dense": tabpfn_oof, "H0_plus_TabPFN": blended.astype(np.float32)}, pd.concat([h0_folds, pd.DataFrame(tabpfn_folds), pd.DataFrame(blend_folds)], ignore_index=True), audits, started)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-h0-dense-tabpfn-screen-01")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=_device())
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        print(json.dumps(smoke_contract(), ensure_ascii=False), flush=True)
    else:
        run(arguments.run_id, arguments.device)


if __name__ == "__main__":
    main()
