"""Seed42 big-jump screen: faithful H0 Selective-EB plus shared candidate ranker.

OOF mode reads train.csv only.  The ranker sees only inner-OOF evidence for
outer-train rows; outer validation is transform/evaluation only.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve()
NOTEBOOKS = HERE.parents[2]
for path in (NOTEBOOKS / "exp_model_007" / "common", NOTEBOOKS / "exp_model_006" / "common", HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from faithful_h0_ranker_core import apply_residual, build_evidence_shape, candidate_residual, make_symmetric_pairs  # noqa: E402
from h0_faithful_pipeline import fit_vocabulary, transform_rows  # noqa: E402
from h0_selective_eb_replacement import fit_empirical_bayes  # noqa: E402
from h0_selective_eb_replacement_runner import fit_fold  # noqa: E402

SEED = 42
PAIRWISE_C = 0.035
ALPHAS = (0.10, 0.20)
LOW_MARGIN = 0.05


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
        "outer_validation_used_for_eb_fit": False,
        "outer_validation_used_for_ranker_fit": False,
        "fixed_class_gene_exact_mutation_rules": False,
        "nan_as_mutation_count": 0,
        "leakage_check": True,
    }


def summary_columns() -> list[str]:
    return [
        "variant", "oof_macro_f1", "oof_accuracy", "feature_count", "convergence_warning_count",
        "leakage_check", "nan_as_mutation_count", "runtime_seconds", "delta_vs_h0",
    ]


def _save_checkpoint(path: Path, payload: dict) -> None:
    arrays = {f"oof__{name}": np.asarray(value) for name, value in payload["oof"].items()}
    metadata = json.dumps({key: payload[key] for key in ("completed_folds", "fold_rows", "audit_rows", "alpha_rows")})
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata_json=np.asarray(metadata), **arrays)
    os.replace(temporary, path)
    path.with_suffix(".progress.json").write_text(json.dumps({"completed_folds": payload["completed_folds"]}, indent=2), encoding="utf-8")


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        oof = {name.removeprefix("oof__"): archive[name].copy() for name in archive.files if name.startswith("oof__")}
    return {**metadata, "oof": oof}


def _shape_from_train(
    fit_frame: pd.DataFrame,
    apply_frame: pd.DataFrame,
    labels: np.ndarray,
    classes: np.ndarray,
    genes: list[str],
    probability: np.ndarray,
) -> np.ndarray:
    """Fit EB evidence only on fit_frame, then describe apply_frame candidates."""
    vocabulary = fit_vocabulary(fit_frame, genes)
    fit_parsed = transform_rows(fit_frame, genes, vocabulary)
    apply_parsed = transform_rows(apply_frame, genes, vocabulary)
    state = fit_empirical_bayes(fit_parsed.gene_type, labels, classes)
    selected_apply = apply_parsed.gene_type[:, state.selected]
    priors = np.asarray([(labels == label).mean() for label in classes], dtype=np.float32)
    return build_evidence_shape(selected_apply, state.weights, priors, probability)


def _pair_sample_weight(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    count = {label: int((labels == label).sum()) for label in classes}
    values: list[float] = []
    for label in labels:
        weight = len(labels) / max(len(classes) * count[label], 1)
        values.extend([weight] * (2 * (len(classes) - 1)))
    return np.asarray(values, dtype=np.float64)


def _fit_ranker(features: np.ndarray, labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[LogisticRegression, int]:
    x, y = make_symmetric_pairs(features, labels, classes)
    model = LogisticRegression(solver="lbfgs", C=PAIRWISE_C, max_iter=2000, class_weight=None, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y, sample_weight=_pair_sample_weight(labels, classes))
    return model, int(sum(issubclass(item.category, ConvergenceWarning) for item in caught))


def _inner_oof_meta(
    frame: pd.DataFrame,
    labels: np.ndarray,
    genes: list[str],
    classes: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, list[dict]]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    shapes = np.zeros((len(frame), len(classes), 19), dtype=np.float32)
    probabilities = np.zeros((len(frame), len(classes)), dtype=np.float32)
    warning_count, audits = 0, []
    for inner_fold, (fit_index, holdout_index) in enumerate(splitter.split(np.zeros(len(frame)), labels), 1):
        fit_frame = frame.iloc[fit_index][genes].reset_index(drop=True)
        holdout_frame = frame.iloc[holdout_index][genes].reset_index(drop=True)
        result = fit_fold(fit_frame, holdout_frame, labels[fit_index], genes, classes, seed=seed * 10 + inner_fold)
        probabilities[holdout_index] = result["candidate"]
        shapes[holdout_index] = _shape_from_train(fit_frame, holdout_frame, labels[fit_index], classes, genes, result["candidate"])
        audit = {
            "inner_fold": inner_fold,
            "inner_fit_rows": int(len(fit_index)), "inner_holdout_rows": int(len(holdout_index)),
            "inner_disjoint": not bool(set(fit_index) & set(holdout_index)),
            "outer_validation_used_for_fit": False,
            "test_read": False,
            "nan_as_mutation_count": int(result["audit"]["nan_as_mutation_count"]),
        }
        if not audit["inner_disjoint"] or audit["nan_as_mutation_count"] != 0:
            raise AssertionError("inner OOF contract failed")
        audits.append(audit)
        warning_count += int(result["h0_warning"] + result["eb_warning"])
        del result
        gc.collect()
    return shapes, probabilities, warning_count, audits


def _select_alpha(features: np.ndarray, probability: np.ndarray, labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[float, int, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 911)
    residual = np.zeros_like(probability, dtype=np.float32)
    warnings_count = 0
    for fit_index, holdout_index in splitter.split(np.zeros(len(labels)), labels):
        ranker, warning_count_fold = _fit_ranker(features[fit_index], labels[fit_index], classes, seed)
        residual[holdout_index] = candidate_residual(features[holdout_index], ranker.coef_.ravel(), float(ranker.intercept_[0]))
        warnings_count += warning_count_fold
        del ranker
    rows = []
    for alpha in ALPHAS:
        corrected = apply_residual(probability, residual, alpha)
        rows.append({"alpha": alpha, "inner_oof_macro_f1": float(f1_score(labels, classes[corrected.argmax(axis=1)], average="macro", zero_division=0))})
    table = pd.DataFrame(rows).sort_values(["inner_oof_macro_f1", "alpha"], ascending=[False, True]).reset_index(drop=True)
    return float(table.alpha.iloc[0]), warnings_count, table


def _topk(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> list[dict]:
    order = np.argsort(probability, axis=1)[:, ::-1]
    rows = []
    for k in (1, 2, 3):
        included = np.asarray([labels[row] in classes[order[row, :k]] for row in range(len(labels))])
        rows.append({"k": k, "recall": float(included.mean())})
    return rows


def _decision(summary: pd.DataFrame, folds: pd.DataFrame, class_metrics: pd.DataFrame, low_margin: pd.DataFrame) -> dict:
    h0 = float(summary.loc[summary.variant.eq("H0_selective_EB"), "oof_macro_f1"].iloc[0])
    candidate = float(summary.loc[summary.variant.eq("H3_candidate_ranker"), "oof_macro_f1"].iloc[0])
    paired = folds.pivot(index="fold", columns="variant", values="macro_f1")
    fold_delta = paired.H3_candidate_ranker - paired.H0_selective_EB
    class_delta = class_metrics.candidate_f1 - class_metrics.h0_f1
    low_delta = float(low_margin.loc[low_margin.variant.eq("H3_candidate_ranker"), "macro_f1"].iloc[0] - low_margin.loc[low_margin.variant.eq("H0_selective_EB"), "macro_f1"].iloc[0])
    positive_fold_sum = float(fold_delta.clip(lower=0).sum())
    positive_class_sum = float(class_delta.clip(lower=0).sum())
    criteria = {
        "delta_at_least_0015": candidate - h0 >= .015,
        "four_positive_folds": int((fold_delta > 0).sum()) >= 4,
        "low_margin_not_below_minus_0003": low_delta >= -.003,
        "no_single_fold_dominates": positive_fold_sum == 0 or float(fold_delta.max()) / positive_fold_sum <= .5,
        "no_single_class_dominates": positive_class_sum == 0 or float(class_delta.max()) / positive_class_sum <= .5,
    }
    decision = "strong_validation_candidate" if all(criteria.values()) else "rejected_or_not_detected"
    return {"h0": h0, "candidate": candidate, "delta": candidate - h0, "low_margin_delta": low_delta, **criteria, "decision": decision}


def smoke() -> None:
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv")
    # Retain every class sufficiently often for all nested 3/5-fold fits.
    train = train.groupby("SUBCLASS", group_keys=False).head(15).reset_index(drop=True)
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if int(train[genes].isna().sum().sum()) != 0 or len(genes) < 100 or train.SUBCLASS.value_counts().min() < 15:
        raise AssertionError("train schema/no-NaN contract failed")
    shape, probability, warnings_count, audits = _inner_oof_meta(train, train.SUBCLASS.to_numpy(), genes, np.asarray(sorted(train.SUBCLASS.unique()), dtype=object), SEED)
    if not np.isfinite(shape).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("smoke meta output failed")
    if warnings_count or not all(item["inner_disjoint"] for item in audits):
        raise AssertionError("smoke fit contract failed")
    print(json.dumps({"smoke": "ok", **run_contract(), "shape": list(shape.shape)}), flush=True)


def run(args: argparse.Namespace) -> None:
    started = perf_counter()
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("training data must not contain NaN")
    result = result_directory()
    checkpoint_path = result / f"{args.run_id}_seed{args.seed}_checkpoint.npz"
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is None:
        oof = {"h0": np.zeros((len(train), len(classes)), dtype=np.float32), "candidate": np.zeros((len(train), len(classes)), dtype=np.float32)}
        completed, fold_rows, audit_rows, alpha_rows = set(), [], [], []
    else:
        oof = checkpoint["oof"]; completed = set(checkpoint["completed_folds"]); fold_rows = checkpoint["fold_rows"]; audit_rows = checkpoint["audit_rows"]; alpha_rows = checkpoint["alpha_rows"]
        print(f"[H3] resume completed folds: {sorted(completed)}", flush=True)

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (fit_index, valid_index) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        if fold in completed:
            continue
        print(f"[H3] outer fold {fold}/5: inner OOF H0 Selective-EB meta-features", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        meta_shape, meta_probability, meta_warning, inner_audits = _inner_oof_meta(fit_frame, labels[fit_index], genes, classes, args.seed * 100 + fold)
        alpha, alpha_warning, alpha_table = _select_alpha(meta_shape, meta_probability, labels[fit_index], classes, args.seed * 100 + fold)
        ranker, ranker_warning = _fit_ranker(meta_shape, labels[fit_index], classes, args.seed * 1000 + fold)
        print(f"[H3] outer fold {fold}/5: faithful H0 fit and candidate residual (alpha={alpha:.2f})", flush=True)
        h0_result = fit_fold(fit_frame, valid_frame, labels[fit_index], genes, classes, seed=args.seed * 1000 + fold)
        h0_probability = h0_result["candidate"]
        valid_shape = _shape_from_train(fit_frame, valid_frame, labels[fit_index], classes, genes, h0_probability)
        candidate_probability = apply_residual(h0_probability, candidate_residual(valid_shape, ranker.coef_.ravel(), float(ranker.intercept_[0])), alpha)
        oof["h0"][valid_index], oof["candidate"][valid_index] = h0_probability, candidate_probability
        for name, probability in (("H0_selective_EB", h0_probability), ("H3_candidate_ranker", candidate_probability)):
            prediction = classes[probability.argmax(axis=1)]
            fold_rows.append({"seed": args.seed, "fold": fold, "variant": name, "macro_f1": float(f1_score(labels[valid_index], prediction, average="macro", zero_division=0)), "accuracy": float(accuracy_score(labels[valid_index], prediction)), "feature_count": int(h0_result["candidate_feature_count"] if name == "H0_selective_EB" else 19), "alpha": alpha if name == "H3_candidate_ranker" else 0.0})
        audit_rows.append({"seed": args.seed, "fold": fold, **run_contract(), "inner_oof_rows": int(len(fit_index)), "outer_validation_rows": int(len(valid_index)), "inner_audit_all_disjoint": bool(all(item["inner_disjoint"] for item in inner_audits)), "h0_convergence_warning_count": int(h0_result["h0_warning"] + h0_result["eb_warning"]), "ranker_convergence_warning_count": int(meta_warning + alpha_warning + ranker_warning), "ranker_training_rows_are_inner_oof": True, "pairwise_coefficient_norm": float(np.linalg.norm(ranker.coef_))})
        alpha_rows.extend(alpha_table.assign(seed=args.seed, fold=fold).to_dict("records"))
        completed.add(fold)
        _save_checkpoint(checkpoint_path, {"completed_folds": list(completed), "fold_rows": fold_rows, "audit_rows": audit_rows, "alpha_rows": alpha_rows, "oof": oof})
        print(f"[H3] outer fold {fold}/5 checkpoint saved", flush=True)
        del meta_shape, meta_probability, valid_shape, h0_result, ranker
        gc.collect()

    fold_frame, audit_frame = pd.DataFrame(fold_rows), pd.DataFrame(audit_rows)
    # Checkpoints created before leakage_check was added retain the atomic audit
    # fields. Reconstruct the derived flag so completed folds can be aggregated
    # without re-fitting any model.
    if "leakage_check" not in audit_frame.columns:
        audit_frame["leakage_check"] = (
            audit_frame["test_read"].eq(False)
            & audit_frame["raw_train_test_concat"].eq(False)
            & audit_frame["outer_validation_used_for_eb_fit"].eq(False)
            & audit_frame["outer_validation_used_for_ranker_fit"].eq(False)
            & audit_frame["inner_audit_all_disjoint"].eq(True)
            & audit_frame["nan_as_mutation_count"].eq(0)
        )
    if not bool(audit_frame.leakage_check.all()) or not bool(audit_frame.inner_audit_all_disjoint.all()) or int(audit_frame.nan_as_mutation_count.max()) != 0:
        raise AssertionError("fold leakage/NaN audit failed")
    warning_count = int(audit_frame.h0_convergence_warning_count.sum() + audit_frame.ranker_convergence_warning_count.sum())
    summary_rows, class_rows = [], []
    for name, probability in (("H0_selective_EB", oof["h0"]), ("H3_candidate_ranker", oof["candidate"])):
        prediction = classes[probability.argmax(axis=1)]
        summary_rows.append({"variant": name, "oof_macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)), "oof_accuracy": float(accuracy_score(labels, prediction)), "feature_count": float(fold_frame.loc[fold_frame.variant.eq(name), "feature_count"].mean()), "convergence_warning_count": warning_count, "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": perf_counter() - started})
        precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        class_rows.extend({"variant": name, "class": label, "precision": float(p), "recall": float(r), "f1": float(score), "support": int(n)} for label, p, r, score, n in zip(classes, precision, recall, f1, support))
    summary, class_frame = pd.DataFrame(summary_rows), pd.DataFrame(class_rows)
    h0_score = float(summary.loc[summary.variant.eq("H0_selective_EB"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_h0"] = summary.oof_macro_f1 - h0_score
    h0_class = class_frame.loc[class_frame.variant.eq("H0_selective_EB")].set_index("class")
    candidate_class = class_frame.loc[class_frame.variant.eq("H3_candidate_ranker")].set_index("class")
    class_metrics = pd.DataFrame({"class": classes, "support": [int(h0_class.loc[label, "support"]) for label in classes], "h0_f1": [float(h0_class.loc[label, "f1"]) for label in classes], "candidate_f1": [float(candidate_class.loc[label, "f1"]) for label in classes]})
    class_metrics["delta_f1"] = class_metrics.candidate_f1 - class_metrics.h0_f1
    margin = np.sort(oof["h0"], axis=1)[:, -1] - np.sort(oof["h0"], axis=1)[:, -2]
    low_mask = margin < LOW_MARGIN
    low_margin = pd.DataFrame({"variant": ["H0_selective_EB", "H3_candidate_ranker"], "group": "h0_margin_lt_005", "support": int(low_mask.sum()), "macro_f1": [float(f1_score(labels[low_mask], classes[probability[low_mask].argmax(axis=1)], average="macro", zero_division=0)) for probability in (oof["h0"], oof["candidate"])]})
    topk = pd.DataFrame([{"variant": name, **row} for name, probability in (("H0_selective_EB", oof["h0"]), ("H3_candidate_ranker", oof["candidate"])) for row in _topk(labels, probability, classes)])
    selection = _decision(summary, fold_frame, class_metrics, low_margin)
    for name, frame, required in (("summary", summary, set(summary_columns())), ("fold_metrics", fold_frame, {"fold", "variant", "macro_f1", "accuracy", "feature_count", "alpha"}), ("class_metrics", class_metrics, {"class", "support", "h0_f1", "candidate_f1", "delta_f1"})):
        if not required.issubset(frame.columns):
            raise AssertionError(f"{name} result schema failure")
    prefix = result / f"{args.run_id}_seed{args.seed}"
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False); fold_frame.to_csv(prefix.with_name(prefix.name + "_fold_metrics.csv"), index=False); class_metrics.to_csv(prefix.with_name(prefix.name + "_class_metrics.csv"), index=False); low_margin.to_csv(prefix.with_name(prefix.name + "_low_margin.csv"), index=False); topk.to_csv(prefix.with_name(prefix.name + "_topk.csv"), index=False); pd.DataFrame(alpha_rows).to_csv(prefix.with_name(prefix.name + "_alpha_selection.csv"), index=False); audit_frame.to_csv(prefix.with_name(prefix.name + "_fold_audit.csv"), index=False)
    pd.DataFrame({"true_class": labels, **{f"h0__{label}": oof["h0"][:, index] for index, label in enumerate(classes)}, **{f"h3__{label}": oof["candidate"][:, index] for index, label in enumerate(classes)}}).to_csv(prefix.with_name(prefix.name + "_oof_probabilities.csv"), index=False)
    prefix.with_name(prefix.name + "_leakage_audit.json").write_text(json.dumps({**run_contract(), "leakage_check": True, "selection": selection}, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = fold_frame.pivot(index="fold", columns="variant", values="macro_f1").plot(marker="o", title="Faithful H0 candidate ranker: fold Macro F1"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_fold_macro_f1.png"), dpi=160); plt.close(ax.figure)
    ax = class_metrics.sort_values("delta_f1").plot.barh(x="class", y="delta_f1", title="Candidate ranker: class F1 delta"); ax.axvline(0, color="black"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_class_f1_delta.png"), dpi=160); plt.close(ax.figure)
    ax = topk.pivot(index="k", columns="variant", values="recall").plot.bar(title="Candidate ranker: Top-k recall"); ax.figure.tight_layout(); ax.figure.savefig(prefix.with_name(prefix.name + "_topk_recall.png"), dpi=160); plt.close(ax.figure)
    print(json.dumps(selection, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-id", default="exp-faithful-h0-candidate-ranker-01")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.seed != SEED and not arguments.smoke:
        raise ValueError("this first screen is intentionally locked to seed42")
    smoke() if arguments.smoke else run(arguments)


if __name__ == "__main__":
    main()
