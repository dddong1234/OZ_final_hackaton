"""Seed42 train-only screen for inner-OOF validated binary pair specialists."""
from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from auto_validated_pair_specialist import apply_pair_probability, select_non_overlapping_pairs, top_two_pair_mask
from h1_auto_confusion_moe import fit_h0_fold


SEED = 42
H0_REFERENCE = .544744
INNER_SPLITS = 3
MAX_CANDIDATES = 8
MAX_SELECTED_PAIRS = 2
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


def _binary_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", boosting_type="gbdt", n_estimators=100, learning_rate=.02,
        num_leaves=20, min_child_samples=10, reg_alpha=0.0, reg_lambda=0.0,
        class_weight="balanced", random_state=seed, n_jobs=-1,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )


def _align_pair(model: LGBMClassifier, probability: np.ndarray, pair: tuple[str, str]) -> np.ndarray:
    lookup = {str(label): index for index, label in enumerate(model.classes_)}
    output = probability[:, [lookup[pair[0]], lookup[pair[1]]]]
    output /= output.sum(axis=1, keepdims=True)
    return output


def _confusion_candidates(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> list[tuple[str, str]]:
    prediction = classes[probability.argmax(axis=1)]
    rows = []
    for left_index, left in enumerate(classes):
        for right in classes[left_index + 1:]:
            swapped = int(((labels == left) & (prediction == right)).sum() + ((labels == right) & (prediction == left)).sum())
            if swapped:
                rows.append((-swapped, str(left), str(right)))
    return [(left, right) for _, left, right in sorted(rows)[:MAX_CANDIDATES]]


def _save_inner_artifact(directory: Path, fold: int, h0, y_fit: np.ndarray, valid_index: np.ndarray) -> dict:
    x_fit_path, x_valid_path = directory / f"inner_{fold}_fit.npz", directory / f"inner_{fold}_valid.npz"
    sparse.save_npz(x_fit_path, h0.x_fit); sparse.save_npz(x_valid_path, h0.x_apply)
    return {"fold": fold, "x_fit": x_fit_path, "x_valid": x_valid_path, "y_fit": np.asarray(y_fit), "valid_index": np.asarray(valid_index), "probability": h0.probability, "warning_count": h0.convergence_warnings, "feature_count": len(h0.names)}


def _evaluate_pair_inner(pair: tuple[str, str], artifacts: list[dict], inner_probability: np.ndarray, inner_labels: np.ndarray, classes: np.ndarray, seed: int) -> dict:
    truth, base_prediction, candidate_prediction = [], [], []
    routed_total = 0
    for artifact in artifacts:
        x_fit, x_valid = sparse.load_npz(artifact["x_fit"]), sparse.load_npz(artifact["x_valid"])
        y_fit, valid_index = artifact["y_fit"], artifact["valid_index"]
        mask = np.isin(y_fit, pair)
        model = _binary_lgbm(seed + artifact["fold"])
        model.fit(x_fit[mask], y_fit[mask])
        expert = _align_pair(model, model.predict_proba(x_valid), pair)
        base = inner_probability[valid_index]
        route = top_two_pair_mask(base, classes, pair)
        adjusted = apply_pair_probability(base, expert, classes, pair, route=route)
        local_truth = inner_labels[valid_index]
        pair_rows = np.isin(local_truth, pair)
        truth.extend(local_truth[pair_rows])
        base_prediction.extend(classes[base[pair_rows].argmax(axis=1)])
        candidate_prediction.extend(classes[adjusted[pair_rows].argmax(axis=1)])
        routed_total += int(route.sum())
        del model, x_fit, x_valid, expert, adjusted
        gc.collect()
    truth_array, base_array, candidate_array = np.asarray(truth), np.asarray(base_prediction), np.asarray(candidate_prediction)
    base_f1 = float(f1_score(truth_array, base_array, labels=list(pair), average="macro", zero_division=0))
    pair_f1 = float(f1_score(truth_array, candidate_array, labels=list(pair), average="macro", zero_division=0))
    recovered = int(((base_array != truth_array) & (candidate_array == truth_array)).sum())
    broken = int(((base_array == truth_array) & (candidate_array != truth_array)).sum())
    return {"pair": pair, "inner_base_pair_f1": base_f1, "inner_specialist_pair_f1": pair_f1, "pair_f1_delta": pair_f1 - base_f1, "recovered": recovered, "broken": broken, "inner_routed_rows": routed_total}


def _inner_selection(fit_frame: pd.DataFrame, y_fit: np.ndarray, genes: list[str], classes: np.ndarray, seed: int) -> tuple[list[tuple[str, str]], list[dict], list[dict], int]:
    """Generate all selection signals using only one outer-fold training partition."""
    inner_probability = np.zeros((len(fit_frame), len(classes)), dtype=np.float64)
    artifacts: list[dict] = []
    audit_rows: list[dict] = []
    warnings_count = 0
    with tempfile.TemporaryDirectory(prefix="auto_pair_inner_") as temporary:
        directory = Path(temporary)
        splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
        for inner_fold, (inner_fit, inner_valid) in enumerate(splitter.split(np.zeros(len(y_fit)), y_fit), 1):
            h0 = fit_h0_fold(fit_frame.iloc[inner_fit].reset_index(drop=True), fit_frame.iloc[inner_valid].reset_index(drop=True), y_fit[inner_fit], genes, classes, seed=seed * 100 + inner_fold)
            if h0.audit["raw_train_test_concat"]:
                raise AssertionError("inner train/test concatenation contract violation")
            inner_probability[inner_valid] = h0.probability
            artifacts.append(_save_inner_artifact(directory, inner_fold, h0, y_fit[inner_fit], inner_valid))
            audit_rows.append({"inner_fold": inner_fold, "fit_rows": int(len(inner_fit)), "validation_rows": int(len(inner_valid)), "outer_validation_used_for_pair_fit": False, "leakage_check": True, "nan_as_mutation_count": 0})
            warnings_count += h0.convergence_warnings
            del h0
            gc.collect()
        candidates = _confusion_candidates(y_fit, inner_probability, classes)
        candidate_rows = [_evaluate_pair_inner(pair, artifacts, inner_probability, y_fit, classes, seed * 1000) for pair in candidates]
        selected = select_non_overlapping_pairs(candidate_rows, maximum=MAX_SELECTED_PAIRS)
    return selected, candidate_rows, audit_rows, warnings_count


def _decision(h0: float, candidate: float, fold_delta: np.ndarray, low_delta: float) -> dict:
    delta, positive = float(candidate - h0), int((fold_delta > 0).sum())
    if delta >= .015 and positive >= 4 and low_delta >= -.003:
        verdict = "strong_validation_candidate"
    elif delta >= .008 and positive >= 4 and low_delta >= -.003:
        verdict = "validation_candidate"
    elif delta > 0:
        verdict = "not_detected"
    else:
        verdict = "rejected"
    return {"decision": verdict, "delta_vs_h0": delta, "positive_fold_count": positive, "low_margin_delta": float(low_delta), "h0_reference": H0_REFERENCE, "h0_reference_delta": float(h0 - H0_REFERENCE)}


def smoke() -> None:
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv", nrows=8)
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    assert len(genes) == 4384
    assert int(train[genes].isna().sum().sum()) == 0
    assert MAX_SELECTED_PAIRS == 2 and INNER_SPLITS == 3
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "fixed_class_gene_mutation_rules": False}))


def run(run_id: str) -> None:
    started = perf_counter()
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    labels = train.SUBCLASS.to_numpy(); classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    h0_oof, specialist_oof = np.zeros((len(train), len(classes))), np.zeros((len(train), len(classes)))
    fold_rows, selection_rows, applied_rows, inner_audit_rows = [], [], [], []
    warning_count = 0
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for outer_fold, (fit_index, valid_index) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        print(f"[auto-pair] outer fold {outer_fold}/5: inner OOF selection", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        selected, candidates, audits, inner_warnings = _inner_selection(fit_frame, labels[fit_index], genes, classes, SEED * 10 + outer_fold)
        print(f"[auto-pair] outer fold {outer_fold}/5: selected={selected}", flush=True)
        h0 = fit_h0_fold(fit_frame, valid_frame, labels[fit_index], genes, classes, seed=SEED * 100 + outer_fold)
        final_probability = h0.probability.copy()
        original_probability = h0.probability.copy()
        for pair_index, pair in enumerate(selected):
            mask = np.isin(labels[fit_index], pair)
            model = _binary_lgbm(SEED * 10000 + outer_fold * 10 + pair_index)
            model.fit(h0.x_fit[mask], labels[fit_index][mask])
            expert = _align_pair(model, model.predict_proba(h0.x_apply), pair)
            route = top_two_pair_mask(original_probability, classes, pair)
            final_probability = apply_pair_probability(final_probability, expert, classes, pair, route=route)
            applied_rows.append({"outer_fold": outer_fold, "pair": "|".join(pair), "routed_rows": int(route.sum()), "fit_support": int(mask.sum())})
            del model, expert
        h0_oof[valid_index], specialist_oof[valid_index] = h0.probability, final_probability
        h0_f1, _, _ = _metrics(labels[valid_index], h0.probability, classes)
        candidate_f1, _, _ = _metrics(labels[valid_index], final_probability, classes)
        for name, values, score in (("H0", h0.probability, h0_f1), ("auto_validated_pair_specialist", final_probability, candidate_f1)):
            fold_rows.append({"fold": outer_fold, "variant": name, "macro_f1": score, "accuracy": float(accuracy_score(labels[valid_index], classes[values.argmax(axis=1)])), "feature_count": len(h0.names), "delta_vs_h0": score - h0_f1})
        for row in candidates:
            selection_rows.append({"outer_fold": outer_fold, "selected": tuple(row["pair"]) in selected, "pair": "|".join(row["pair"]), **{key: value for key, value in row.items() if key != "pair"}})
        inner_audit_rows.extend({"outer_fold": outer_fold, **row} for row in audits)
        warning_count += h0.convergence_warnings + inner_warnings
        del h0, final_probability, original_probability, fit_frame, valid_frame
        gc.collect()
    h0_score, h0_accuracy, h0_prediction = _metrics(labels, h0_oof, classes)
    candidate_score, candidate_accuracy, candidate_prediction = _metrics(labels, specialist_oof, classes)
    folds = pd.DataFrame(fold_rows)
    h0_folds = folds[folds.variant.eq("H0")].sort_values("fold").macro_f1.to_numpy()
    candidate_folds = folds[folds.variant.eq("auto_validated_pair_specialist")].sort_values("fold").macro_f1.to_numpy()
    margin = np.sort(h0_oof, axis=1)[:, -1] - np.sort(h0_oof, axis=1)[:, -2]
    low = margin < .05
    low_h0 = float(f1_score(labels[low], h0_prediction[low], average="macro", zero_division=0))
    low_candidate = float(f1_score(labels[low], candidate_prediction[low], average="macro", zero_division=0))
    verdict = _decision(h0_score, candidate_score, candidate_folds - h0_folds, low_candidate - low_h0)
    summary = pd.DataFrame([
        {"variant": "H0", "oof_macro_f1": h0_score, "oof_accuracy": h0_accuracy, "feature_count": float(folds.feature_count.mean()), "convergence_warning_count": warning_count, "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": 0.0},
        {"variant": "auto_validated_pair_specialist", "oof_macro_f1": candidate_score, "oof_accuracy": candidate_accuracy, "feature_count": float(folds.feature_count.mean()), "convergence_warning_count": warning_count, "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": candidate_score - h0_score},
    ])
    class_rows = []
    for label in classes:
        class_rows.append({"class": str(label), "support": int((labels == label).sum()), "H0_f1": f1_score(labels == label, h0_prediction == label, zero_division=0), "specialist_f1": f1_score(labels == label, candidate_prediction == label, zero_division=0)})
    class_frame = pd.DataFrame(class_rows); class_frame["delta"] = class_frame.specialist_f1 - class_frame.H0_f1
    low_frame = pd.DataFrame([{"group": "H0_margin_<0.05", "support": int(low.sum()), "H0_macro_f1": low_h0, "specialist_macro_f1": low_candidate, "delta": low_candidate - low_h0}])
    result = Path(__file__).parent.parent / "result"; result.mkdir(exist_ok=True)
    if not RESULT_COLUMNS["summary"].issubset(summary.columns) or not RESULT_COLUMNS["fold"].issubset(folds.columns):
        raise AssertionError("result schema failure")
    summary.to_csv(result / f"{run_id}_seed42_summary.csv", index=False)
    folds.to_csv(result / f"{run_id}_seed42_fold_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(result / f"{run_id}_seed42_inner_candidate_audit.csv", index=False)
    pd.DataFrame(applied_rows).to_csv(result / f"{run_id}_seed42_applied_pairs.csv", index=False)
    pd.DataFrame(inner_audit_rows).to_csv(result / f"{run_id}_seed42_inner_outer_audit.csv", index=False)
    class_frame.to_csv(result / f"{run_id}_seed42_class_metrics.csv", index=False)
    low_frame.to_csv(result / f"{run_id}_seed42_low_margin.csv", index=False)
    pd.DataFrame({"true_class": labels, **{f"h0__{label}": h0_oof[:, index] for index, label in enumerate(classes)}, **{f"specialist__{label}": specialist_oof[:, index] for index, label in enumerate(classes)}}).to_csv(result / f"{run_id}_seed42_oof_probabilities.csv", index=False)
    audit = {"seed": SEED, "outer_splits": 5, "inner_splits": INNER_SPLITS, "maximum_candidates": MAX_CANDIDATES, "maximum_selected_pairs": MAX_SELECTED_PAIRS, "test_read": False, "train_test_concat": False, "fixed_class_gene_mutation_rules": False, "pair_source": "outer_train_inner_oof_only", "leakage_check": True, "nan_as_mutation_count": 0, "convergence_warning_count": warning_count, "runtime_seconds": perf_counter() - started, **verdict}
    (result / f"{run_id}_seed42_leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = folds.pivot(index="fold", columns="variant", values="macro_f1").plot(marker="o", title="H0 vs auto-validated pair specialist")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    ax = class_frame.set_index("class").delta.sort_values().plot.barh(title="Auto-pair specialist class F1 delta")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-auto-validated-pair-specialist-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id)


if __name__ == "__main__":
    main()
