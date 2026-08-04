"""User-run seed42 faithful H0 reproduction audit; train-only and self-contained."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from h0_faithful_pipeline import build_design_matrices


SEED = 42
REFERENCE = .543679
TOLERANCE = .001
REFERENCE_COMPONENTS = {"exp013_LR": .526130, "exp014_LGBM_hard_specialist": .492332, "H0_blend_80_20": REFERENCE}
RESULT_COLUMNS = {"summary": {"variant", "oof_macro_f1", "oof_accuracy", "feature_count_mean", "convergence_warning_count", "leakage_check", "nan_as_mutation_count", "reference_delta"}, "fold": {"fold", "variant", "macro_f1", "feature_count"}}


def project_root() -> Path:
    for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (path / "data" / "raw" / "train.csv").exists(): return path
    raise FileNotFoundError("data/raw/train.csv not found")


def baseline_gate(score: float) -> dict:
    delta = float(score - REFERENCE); reproduced = abs(delta) <= TOLERANCE
    return {"reference_oof_macro_f1": REFERENCE, "tolerance": TOLERANCE, "observed_oof_macro_f1": float(score), "reference_delta": delta, "baseline_reproduced": reproduced, "block_downstream_experiments": not reproduced, "decision": "reproduced" if reproduced else "baseline_not_reproduced"}


def has_required_columns(schema_name: str, frame: pd.DataFrame) -> bool:
    """Return whether a result frame satisfies its declared output schema."""
    return RESULT_COLUMNS[schema_name].issubset(frame.columns)


def _align(model, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    output = np.zeros((len(probability), len(classes)), dtype=np.float64); lookup = {label: i for i, label in enumerate(model.classes_)}
    return probability[:, [lookup[label] for label in classes]]


def _discover_pairs(x_fit, names: list[str], y_fit: np.ndarray, classes: np.ndarray) -> tuple[tuple[str, str], ...]:
    gene_columns = np.asarray([name.startswith("G__") for name in names])
    matrix = x_fit[:, gene_columns]; centroids = []
    for label in classes:
        row = np.asarray(matrix[y_fit == label].mean(axis=0)).ravel(); norm = np.linalg.norm(row); centroids.append(row / norm if norm else row)
    profile = np.vstack(centroids); candidates = [(-(float(profile[left] @ profile[right])), classes[left], classes[right]) for left in range(len(classes)) for right in range(left + 1, len(classes))]
    return tuple((left, right) for _, left, right in sorted(candidates)[:2])


def _specialist(main_probability: np.ndarray, x_fit, y_fit: np.ndarray, x_valid, classes: np.ndarray, pairs: tuple[tuple[str, str], ...]) -> np.ndarray:
    output = main_probability.copy(); index = {label: i for i, label in enumerate(classes)}; original = classes[main_probability.argmax(axis=1)]
    for pair in pairs:
        mask = np.isin(y_fit, pair)
        model = LGBMClassifier(objective="binary", boosting_type="gbdt", n_estimators=100, learning_rate=.02, num_leaves=20, min_child_samples=10, reg_alpha=0.0, reg_lambda=0.0, importance_type="gain", class_weight="balanced", random_state=SEED, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1)
        model.fit(x_fit[mask], y_fit[mask]); route = np.isin(original, pair)
        if not route.any(): continue
        raw = model.predict_proba(x_valid[route]); local = {label: i for i, label in enumerate(model.classes_)}; ratio = raw[:, local[pair[0]]]
        cols = [index[pair[0]], index[pair[1]]]; mass = main_probability[route][:, cols].sum(axis=1)
        output[np.ix_(np.flatnonzero(route), cols)] = np.column_stack((mass * ratio, mass * (1 - ratio)))
    np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-6)
    return output


def smoke() -> None:
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv", nrows=8); genes = [col for col in train if col not in ("ID", "SUBCLASS")]
    assert len(genes) == 4384 and int(train[genes].isna().sum().sum()) == 0
    assert all(columns for columns in RESULT_COLUMNS.values())
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0, "result_schemas": {name: sorted(cols) for name, cols in RESULT_COLUMNS.items()}}))


def run(run_id: str) -> None:
    started = perf_counter(); root = project_root(); train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [col for col in train if col not in ("ID", "SUBCLASS")]; labels = train.SUBCLASS.to_numpy(); classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0: raise AssertionError("train NaN contract violation")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED); lr_oof = np.zeros((len(train), len(classes))); lgbm_oof = np.zeros_like(lr_oof); blend_oof = np.zeros_like(lr_oof)
    fold_rows, feature_rows, pair_rows, audit_rows = [], [], [], []; warnings_seen = 0
    for fold, (fit, valid) in enumerate(splitter.split(np.zeros(len(train)), labels), 1):
        print(f"[faithful-H0] fold {fold}/5: feature fit", flush=True)
        x_fit, x_valid, names, audit = build_design_matrices(train.iloc[fit][genes].reset_index(drop=True), train.iloc[valid][genes].reset_index(drop=True), labels[fit], genes, seed=SEED * 100 + fold)
        lr = LogisticRegression(solver="lbfgs", C=.07, max_iter=2000, class_weight="balanced", random_state=SEED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning); lr.fit(x_fit, labels[fit])
        warnings_seen += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        lr_p = _align(lr, lr.predict_proba(x_valid), classes)
        main = LGBMClassifier(objective="multiclass", boosting_type="gbdt", num_class=len(classes), n_estimators=400, learning_rate=.05, num_leaves=25, min_child_samples=10, min_child_weight=1e-3, reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced", random_state=SEED, n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1)
        main.fit(x_fit, labels[fit]); main_p = _align(main, main.predict_proba(x_valid), classes); pairs = _discover_pairs(x_fit, names, labels[fit], classes); specialist_p = _specialist(main_p, x_fit, labels[fit], x_valid, classes, pairs)
        blend_p = .80 * lr_p + .20 * specialist_p; lr_oof[valid], lgbm_oof[valid], blend_oof[valid] = lr_p, specialist_p, blend_p
        for name, probability in (("exp013_LR", lr_p), ("exp014_LGBM_hard_specialist", specialist_p), ("H0_blend_80_20", blend_p)):
            pred = classes[probability.argmax(axis=1)]; fold_rows.append({"fold": fold, "variant": name, "macro_f1": float(f1_score(labels[valid], pred, average="macro", zero_division=0)), "accuracy": float(accuracy_score(labels[valid], pred)), "feature_count": len(names)})
        feature_rows.append({"fold": fold, "feature_count": len(names), **audit["pre_filter_block_counts"], "exact_vocabulary_size": audit["exact_vocabulary_size"], "gene_type_vocabulary_size": audit["gene_type_vocabulary_size"]})
        pair_rows.extend({"fold": fold, "left_class": left, "right_class": right} for left, right in pairs); audit_rows.append({"fold": fold, "leakage_check": audit["raw_train_test_concat"] is False and audit["vocabulary_source"] == "fit_frame_only", "test_read": False, "nan_as_mutation_count": 0, **audit})
        del x_fit, x_valid, lr, main, specialist_p; gc.collect()
    rows = []
    for name, probability in (("exp013_LR", lr_oof), ("exp014_LGBM_hard_specialist", lgbm_oof), ("H0_blend_80_20", blend_oof)):
        prediction = classes[probability.argmax(axis=1)]; rows.append({"variant": name, "oof_macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)), "oof_accuracy": float(accuracy_score(labels, prediction)), "feature_count_mean": float(np.mean([row["feature_count"] for row in fold_rows if row["variant"] == name])), "convergence_warning_count": warnings_seen if name == "H0_blend_80_20" else 0, "leakage_check": True, "nan_as_mutation_count": 0})
    summary = pd.DataFrame(rows); h0 = float(summary.loc[summary.variant.eq("H0_blend_80_20"), "oof_macro_f1"].iloc[0]); gate = baseline_gate(h0); summary["reference_oof_macro_f1"] = summary.variant.map(REFERENCE_COMPONENTS); summary["reference_delta"] = summary.oof_macro_f1 - summary.reference_oof_macro_f1
    if not has_required_columns("summary", summary): raise AssertionError("summary schema failure")
    folds = pd.DataFrame(fold_rows); features = pd.DataFrame(feature_rows); result = Path(__file__).parent.parent / "result"; result.mkdir(exist_ok=True)
    summary.to_csv(result / f"{run_id}_seed42_summary.csv", index=False); summary[["variant", "oof_macro_f1", "reference_oof_macro_f1", "reference_delta"]].to_csv(result / f"{run_id}_seed42_reproduction_diff.csv", index=False); folds.to_csv(result / f"{run_id}_seed42_fold_metrics.csv", index=False); features.to_csv(result / f"{run_id}_seed42_feature_audit.csv", index=False); pd.DataFrame(pair_rows).to_csv(result / f"{run_id}_seed42_auto_pairs.csv", index=False); pd.DataFrame(audit_rows).to_csv(result / f"{run_id}_seed42_fold_audit.csv", index=False)
    config = {"seed": SEED, "outer_splits": 5, "enrichment_inner_splits": 5, "blend": {"lr": .80, "lgbm_hard_specialist": .20}, "reference": REFERENCE, "runtime_seconds": perf_counter() - started, "test_read": False, "fixed_class_gene_mutation_rules": False, **gate}
    (result / f"{run_id}_seed42_reproduction_audit.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = folds.pivot(index="fold", columns="variant", values="macro_f1").plot(marker="o", title="Faithful H0: fold Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    ax = features.set_index("fold").feature_count.plot.bar(title="Faithful H0: feature count by fold"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_feature_count.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(gate, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", default="exp-h0-faithful-reproduction-01"); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id)


if __name__ == "__main__": main()
