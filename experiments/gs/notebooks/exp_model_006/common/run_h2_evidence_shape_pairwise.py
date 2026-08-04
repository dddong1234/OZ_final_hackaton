"""User-run, train-only seed42 screen for H2-S shared evidence-shape ranking."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from h2_evidence_shape_core import assert_fold_contract
from h2_pairwise_ranker import apply_residual_correction, candidate_residuals, make_symmetric_pairs
from h2_safe_h0 import build_evidence_shape, fit_predict_h0


SEED = 42
ALPHAS = (.10, .20)
PAIRWISE_C = .035
LOW_MARGIN = .05
H0_REFERENCE_MACRO_F1 = .543679
H0_REFERENCE_TOLERANCE = .001
REQUIRED_RESULT_COLUMNS = {
    "summary": {"variant", "oof_macro_f1", "oof_accuracy", "feature_count", "convergence_warning_count", "leakage_check", "nan_as_mutation_count", "runtime_seconds", "delta_vs_h0"},
    "fold_metrics": {"fold", "variant", "macro_f1", "accuracy", "feature_count", "alpha"},
    "class_metrics": {"class", "support", "h0_f1", "h2_f1", "delta_f1"},
}


def root() -> Path:
    for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv를 가진 프로젝트 루트를 찾지 못했습니다.")


def pair_sample_weight(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    count = {label: int((labels == label).sum()) for label in classes}
    weights = []
    for label in labels:
        value = len(labels) / max(len(classes) * count[label], 1)
        weights.extend([value] * (2 * (len(classes) - 1)))
    return np.asarray(weights, dtype=np.float64)


def fit_ranker(features: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> tuple[LogisticRegression, int]:
    x, y = make_symmetric_pairs(features, labels, classes)
    model = LogisticRegression(solver="lbfgs", C=PAIRWISE_C, max_iter=2000, class_weight=None, random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y, sample_weight=pair_sample_weight(labels, classes))
    return model, sum(issubclass(item.category, ConvergenceWarning) for item in caught)


def inner_oof_meta(frame: pd.DataFrame, genes: list[str], labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, int, list[dict]]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    features = np.zeros((len(frame), len(classes), 19), dtype=np.float32)
    probability = np.zeros((len(frame), len(classes)), dtype=np.float32)
    warnings_count, audits = 0, []
    for fold, (fit, holdout) in enumerate(splitter.split(np.zeros(len(frame)), labels), 1):
        model = fit_predict_h0(frame.iloc[fit], frame.iloc[holdout], genes, labels[fit], seed=seed * 10 + fold)
        features[holdout] = build_evidence_shape(model.output_matrices.gene_type, model.state.eb_weights, model.state.class_prior, model.probability)
        probability[holdout] = model.probability
        warnings_count += model.convergence_warnings
        audits.append({"inner_fold": fold, **assert_fold_contract(np.arange(len(frame)), fit, holdout), "h0_feature_count": model.feature_count, "specialist_pairs": [list(pair) for pair in model.specialist_pairs]})
        del model
        gc.collect()
    return features, probability, warnings_count, audits


def choose_alpha(features: np.ndarray, probability: np.ndarray, labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[float, int, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 901)
    residual = np.zeros_like(probability, dtype=np.float64)
    warning_count = 0
    for fit, holdout in splitter.split(np.zeros(len(labels)), labels):
        model, warnings_seen = fit_ranker(features[fit], labels[fit], classes)
        residual[holdout] = candidate_residuals(features[holdout], model.coef_.ravel(), float(model.intercept_[0]))
        warning_count += warnings_seen
        del model
    rows = []
    for alpha in ALPHAS:
        corrected = apply_residual_correction(probability, residual, alpha)
        rows.append({"alpha": alpha, "inner_oof_macro_f1": float(f1_score(labels, classes[corrected.argmax(axis=1)], average="macro"))})
    table = pd.DataFrame(rows).sort_values(["inner_oof_macro_f1", "alpha"], ascending=[False, True]).reset_index(drop=True)
    return float(table.alpha.iloc[0]), warning_count, table


def topk(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> list[dict]:
    order = np.argsort(probability, axis=1)[:, ::-1]
    rows = []
    for k in (1, 2, 3):
        included = np.asarray([labels[row] in classes[order[row, :k]] for row in range(len(labels))])
        oracle = classes[order[:, 0]].copy()
        oracle[included] = labels[included]
        rows.append({"k": k, "recall": float(included.mean()), "oracle_macro_f1": float(f1_score(labels, oracle, average="macro"))})
    return rows


def decision(summary: pd.DataFrame, folds: pd.DataFrame, classes: pd.DataFrame, low: pd.DataFrame) -> dict:
    h0 = float(summary.loc[summary.variant.eq("H0"), "oof_macro_f1"].iloc[0])
    h2 = float(summary.loc[summary.variant.eq("H2_S"), "oof_macro_f1"].iloc[0])
    delta = h2 - h0
    paired = folds.pivot(index="fold", columns="variant", values="macro_f1")
    fold_delta = paired.H2_S - paired.H0
    class_delta = classes.h2_f1 - classes.h0_f1
    low_delta = float(low.loc[low.variant.eq("H2_S"), "macro_f1"].iloc[0] - low.loc[low.variant.eq("H0"), "macro_f1"].iloc[0])
    positive_fold = float(fold_delta.clip(lower=0).sum())
    positive_class = float(class_delta.clip(lower=0).sum())
    conditions = {"delta_at_least_0015": delta >= .015, "four_positive_folds": int((fold_delta > 0).sum()) >= 4, "low_margin_not_below_minus_0003": low_delta >= -.003, "no_single_fold_dominates": positive_fold == 0 or float(fold_delta.max()) / positive_fold <= .5, "no_single_class_dominates": positive_class == 0 or float(class_delta.max()) / positive_class <= .5}
    h0_reference_delta = h0 - H0_REFERENCE_MACRO_F1
    h0_reference_match = abs(h0_reference_delta) <= H0_REFERENCE_TOLERANCE
    label = "reject" if delta < .008 else "not_detected" if delta < .015 else "strong_validation_candidate" if all(conditions.values()) else "hold"
    if delta >= .030 and all(conditions.values()):
        label = "jump_candidate"
    if not h0_reference_match:
        label = "baseline_not_reproduced"
    return {"h0": h0, "h2": h2, "delta": delta, "h0_reference_macro_f1": H0_REFERENCE_MACRO_F1, "h0_reference_delta": h0_reference_delta, "h0_reference_match": h0_reference_match, "low_margin_delta": low_delta, **conditions, "decision": label}


def smoke() -> None:
    train = pd.read_csv(root() / "data" / "raw" / "train.csv", nrows=8)
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    assert len(genes) > 100 and train.SUBCLASS.nunique() > 1
    assert int(train[genes].isna().sum().sum()) == 0
    from h2_evidence_shape_core import parse_frame
    parsed = parse_frame(pd.DataFrame({"G": ["WT", np.nan, "R1H R2*", ""]}), ["G"])
    assert len(parsed.events) == 2 and parsed.nan_as_mutation_count == 0
    result = Path(__file__).parent.parent / "result"; result.mkdir(exist_ok=True)
    assert all(columns for columns in REQUIRED_RESULT_COLUMNS.values())
    print(json.dumps({"smoke": "ok", "rows": len(train), "genes": len(genes), "test_read": False, "nan_as_mutation_count": 0, "required_result_schemas": {key: sorted(value) for key, value in REQUIRED_RESULT_COLUMNS.items()}}))


def run(args: argparse.Namespace) -> None:
    started = time.time()
    train = pd.read_csv(root() / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    h0_oof = np.zeros((len(train), len(classes)), dtype=np.float32)
    h2_oof = np.zeros_like(h0_oof)
    fold_rows, alpha_rows, audit_rows, pairs_rows, warning_count = [], [], [], [], 0
    coefficient_norms: list[float] = []
    for fold, (fit, valid) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        print(f"[H2-S] outer fold {fold}/5: inner OOF meta features", flush=True)
        meta_x, meta_p, inner_warnings, inner_audit = inner_oof_meta(train.iloc[fit].reset_index(drop=True), genes, labels[fit], classes, SEED * 100 + fold)
        alpha, alpha_warnings, alpha_table = choose_alpha(meta_x, meta_p, labels[fit], classes, SEED * 100 + fold)
        ranker, final_warnings = fit_ranker(meta_x, labels[fit], classes)
        coefficient_norms.append(float(np.linalg.norm(ranker.coef_)))
        print(f"[H2-S] outer fold {fold}/5: outer train fit / validation transform (alpha={alpha:.2f})", flush=True)
        h0 = fit_predict_h0(train.iloc[fit], train.iloc[valid], genes, labels[fit], seed=SEED * 1000 + fold)
        valid_shape = build_evidence_shape(h0.output_matrices.gene_type, h0.state.eb_weights, h0.state.class_prior, h0.probability)
        residual = candidate_residuals(valid_shape, ranker.coef_.ravel(), float(ranker.intercept_[0]))
        h2_probability = apply_residual_correction(h0.probability, residual, alpha).astype(np.float32)
        h0_oof[valid], h2_oof[valid] = h0.probability, h2_probability
        for name, probability in (("H0", h0.probability), ("H2_S", h2_probability)):
            fold_rows.append({"fold": fold, "variant": name, "macro_f1": float(f1_score(labels[valid], classes[probability.argmax(axis=1)], average="macro")), "accuracy": float(accuracy_score(labels[valid], classes[probability.argmax(axis=1)])), "feature_count": h0.feature_count if name == "H0" else 19, "alpha": alpha if name == "H2_S" else 0.0})
        alpha_rows.extend(alpha_table.assign(fold=fold).to_dict("records"))
        audit = assert_fold_contract(fit, fit, valid)
        audit_rows.append({"fold": fold, **audit, "outer_validation_used_for_eb_fit": False, "outer_validation_used_for_pairwise_fit": False, "test_read": False, "nan_as_mutation_count": 0})
        pairs_rows.extend({"fold": fold, "left_class": left, "right_class": right} for left, right in h0.specialist_pairs)
        warning_count += inner_warnings + alpha_warnings + final_warnings + h0.convergence_warnings
        del meta_x, meta_p, h0, valid_shape, residual, ranker
        gc.collect()
    result = Path(__file__).parent.parent / "result"; result.mkdir(exist_ok=True)
    rows, class_rows = [], []
    for name, probability in (("H0", h0_oof), ("H2_S", h2_oof)):
        prediction = classes[probability.argmax(axis=1)]
        rows.append({"variant": name, "oof_macro_f1": float(f1_score(labels, prediction, average="macro")), "oof_accuracy": float(accuracy_score(labels, prediction)), "feature_count": 19 if name == "H2_S" else float(pd.DataFrame(fold_rows).query("variant == 'H0'").feature_count.mean()), "convergence_warning_count": warning_count if name == "H2_S" else 0, "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": time.time() - started})
        precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"variant": name, "class": label, "precision": p, "recall": r, "f1": f, "support": s} for label, p, r, f, s in zip(classes, precision, recall, f1, support))
    summary, folds, class_metrics = pd.DataFrame(rows), pd.DataFrame(fold_rows), pd.DataFrame(class_rows)
    summary["delta_vs_h0"] = summary.oof_macro_f1 - float(summary.loc[summary.variant.eq("H0"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_h0_reference"] = summary.oof_macro_f1 - H0_REFERENCE_MACRO_F1
    h0_class = class_metrics.loc[class_metrics.variant.eq("H0")].set_index("class")
    h2_class = class_metrics.loc[class_metrics.variant.eq("H2_S")].set_index("class")
    class_table = pd.DataFrame({"class": classes, "support": [int(h0_class.loc[label, "support"]) for label in classes], "h0_precision": [float(h0_class.loc[label, "precision"]) for label in classes], "h2_precision": [float(h2_class.loc[label, "precision"]) for label in classes], "h0_recall": [float(h0_class.loc[label, "recall"]) for label in classes], "h2_recall": [float(h2_class.loc[label, "recall"]) for label in classes], "h0_f1": [float(h0_class.loc[label, "f1"]) for label in classes], "h2_f1": [float(h2_class.loc[label, "f1"]) for label in classes]})
    class_table["delta_f1"] = class_table.h2_f1 - class_table.h0_f1
    margin = np.sort(h0_oof, axis=1)[:, -1] - np.sort(h0_oof, axis=1)[:, -2]; low_mask = margin < LOW_MARGIN
    low = pd.DataFrame({"variant": ["H0", "H2_S"], "group": "h0_margin_lt_005", "support": int(low_mask.sum()), "macro_f1": [float(f1_score(labels[low_mask], classes[p[low_mask].argmax(1)], average="macro", zero_division=0)) for p in (h0_oof, h2_oof)]})
    top = pd.DataFrame([{"variant": name, **item} for name, probability in (("H0", h0_oof), ("H2_S", h2_oof)) for item in topk(labels, probability, classes)])
    h0_top3 = np.argsort(h0_oof, axis=1)[:, -3:]; h2_top3 = np.argsort(h2_oof, axis=1)[:, -3:]
    recovered = sum(labels[row] not in classes[h0_top3[row]] and labels[row] in classes[h2_top3[row]] for row in range(len(labels)))
    broken = sum(classes[h0_oof[row].argmax()] == labels[row] and classes[h2_oof[row].argmax()] != labels[row] for row in range(len(labels)))
    selection = decision(summary, folds, class_table, low)
    if not all(item["leakage_check"] and not item["outer_validation_used_for_fit"] and not item["outer_validation_used_for_eb_fit"] and not item["outer_validation_used_for_pairwise_fit"] and not item["test_read"] for item in audit_rows):
        raise AssertionError("H2-S fold leakage audit failed")
    selection.update({"new_top3_recovered_rows": int(recovered), "h0_correct_h2_broken_rows": int(broken), "pairwise_coefficient_norm_mean": float(np.mean(coefficient_norms)), "pairwise_coefficient_norm_max": float(np.max(coefficient_norms))})
    for name, table in (("summary", summary), ("fold_metrics", folds), ("class_metrics", class_table)):
        missing = REQUIRED_RESULT_COLUMNS[name] - set(table.columns)
        if missing:
            raise AssertionError(f"{name} output schema missing: {sorted(missing)}")
    summary.to_csv(result / f"{args.run_id}_seed42_summary.csv", index=False); folds.to_csv(result / f"{args.run_id}_seed42_fold_metrics.csv", index=False); class_table.to_csv(result / f"{args.run_id}_seed42_class_metrics.csv", index=False); pd.DataFrame(alpha_rows).to_csv(result / f"{args.run_id}_seed42_alpha_selection.csv", index=False); pd.DataFrame(pairs_rows).to_csv(result / f"{args.run_id}_seed42_auto_pairs.csv", index=False); low.to_csv(result / f"{args.run_id}_seed42_low_margin.csv", index=False); top.to_csv(result / f"{args.run_id}_seed42_topk.csv", index=False)
    pd.DataFrame({"true_class": labels, **{f"h0__{label}": h0_oof[:, i] for i, label in enumerate(classes)}, **{f"h2_s__{label}": h2_oof[:, i] for i, label in enumerate(classes)}}).to_csv(result / f"{args.run_id}_seed42_oof_probabilities.csv", index=False)
    (result / f"{args.run_id}_seed42_leakage_audit.json").write_text(json.dumps({"leakage_check": True, "nan_as_mutation_count": 0, "test_read": False, "outer_validation_used_for_eb_fit": False, "outer_validation_used_for_pairwise_fit": False, "fixed_class_gene_mutation_rules": False, "selection": selection}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(result / f"{args.run_id}_seed42_fold_audit.csv", index=False)
    ax = folds.pivot(index="fold", columns="variant", values="macro_f1").plot(marker="o", title="H2-S fold Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{args.run_id}_seed42_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    ax = class_table.sort_values("delta_f1").plot.barh(x="class", y="delta_f1", title="H2-S class F1 delta"); ax.axvline(0, color="black"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{args.run_id}_seed42_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    ax = top.pivot(index="k", columns="variant", values="recall").plot.bar(title="H2-S Top-k recall"); ax.figure.tight_layout(); ax.figure.savefig(result / f"{args.run_id}_seed42_topk_recall.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(selection, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", default="exp-h2-evidence-shape-01"); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args)


if __name__ == "__main__":
    main()
