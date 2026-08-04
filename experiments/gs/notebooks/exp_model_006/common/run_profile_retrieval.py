"""Train-only exact mutation-profile posterior retrieval screen.

The runner deliberately never reads test data.  A profile vocabulary and its
class posterior are fitted independently inside each outer-fold train split.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from h1_auto_confusion_moe import fit_h0_fold
from profile_retrieval import build_profile_lookup, fixed_profile_blend, query_profile_posteriors


SEED = 42
PROFILE_BLEND_WEIGHT = .20
H0_REFERENCE = .544744
LOW_MARGIN = .05
RESULT_COLUMNS = {
    "summary": {
        "variant", "oof_macro_f1", "oof_accuracy", "feature_count",
        "convergence_warning_count", "leakage_check", "nan_as_mutation_count", "delta_vs_h0",
    },
    "fold": {"fold", "variant", "macro_f1", "accuracy", "feature_count", "delta_vs_h0"},
}


def project_root() -> Path:
    for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv not found")


def _topk_recall(probability: np.ndarray, labels: np.ndarray, classes: np.ndarray, k: int) -> float:
    truth = np.searchsorted(classes, labels)
    choices = np.argpartition(probability, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(choices == truth[:, None], axis=1)))


def _metrics(labels: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = classes[probability.argmax(axis=1)]
    return (
        float(f1_score(labels, prediction, average="macro", zero_division=0)),
        float(accuracy_score(labels, prediction)),
        prediction,
    )


def _decision(h0_score: float, blend_score: float, fold_deltas: np.ndarray, matched_rate: float, recovered: int, broken: int) -> dict:
    delta = float(blend_score - h0_score)
    positive_folds = int((fold_deltas > 0).sum())
    if delta >= .015 and positive_folds >= 4 and matched_rate >= .05 and recovered > broken:
        verdict = "strong_profile_retrieval_candidate"
    elif delta >= .005 and positive_folds >= 3 and recovered > broken:
        verdict = "profile_retrieval_candidate"
    elif delta > 0:
        verdict = "not_detected"
    else:
        verdict = "rejected"
    return {
        "decision": verdict, "delta_vs_h0": delta, "positive_fold_count": positive_folds,
        "matched_rate": float(matched_rate), "retrieval_recovered_rows": int(recovered),
        "retrieval_broken_rows": int(broken), "h0_reference": H0_REFERENCE,
        "h0_reference_delta": float(h0_score - H0_REFERENCE),
    }


def smoke() -> None:
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv", nrows=8)
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    assert len(genes) == 4384
    assert int(train[genes].isna().sum().sum()) == 0
    assert all(columns for columns in RESULT_COLUMNS.values())
    assert PROFILE_BLEND_WEIGHT == .20
    print(json.dumps({"smoke": "ok", "test_read": False, "nan_as_mutation_count": 0}))


def run(run_id: str) -> None:
    started = perf_counter()
    train = pd.read_csv(project_root() / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(np.unique(labels)), dtype=object)
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN contract violation")

    h0_oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    retrieval_oof = np.zeros_like(h0_oof)
    blend_oof = np.zeros_like(h0_oof)
    match_oof = np.zeros(len(train), dtype=bool)
    support_oof = np.zeros(len(train), dtype=np.int32)
    purity_oof = np.zeros(len(train), dtype=np.float64)
    fold_rows: list[dict] = []
    profile_rows: list[dict] = []
    warning_count = 0
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_index, valid_index) in enumerate(outer.split(np.zeros(len(train)), labels), 1):
        print(f"[profile-retrieval] outer fold {fold}/5: fit H0 and profile lookup", flush=True)
        fit_frame = train.iloc[fit_index][genes].reset_index(drop=True)
        valid_frame = train.iloc[valid_index][genes].reset_index(drop=True)
        y_fit = labels[fit_index]
        h0 = fit_h0_fold(fit_frame, valid_frame, y_fit, genes, classes, seed=SEED * 100 + fold)
        if h0.audit["raw_train_test_concat"]:
            raise AssertionError("H0 train/test concatenation contract violation")
        lookup = build_profile_lookup(fit_frame, y_fit, genes, classes)
        posterior, matched, support, purity = query_profile_posteriors(valid_frame, genes, lookup)
        blended = fixed_profile_blend(h0.probability, posterior, matched, weight=PROFILE_BLEND_WEIGHT)
        h0_oof[valid_index] = h0.probability
        retrieval_oof[valid_index] = posterior
        blend_oof[valid_index] = blended
        match_oof[valid_index], support_oof[valid_index], purity_oof[valid_index] = matched, support, purity
        h0_f1, _, _ = _metrics(labels[valid_index], h0.probability, classes)
        retrieval_f1, _, _ = _metrics(labels[valid_index], posterior, classes)
        blend_f1, _, _ = _metrics(labels[valid_index], blended, classes)
        for variant, probability, score in (
            ("H0", h0.probability, h0_f1),
            ("profile_retrieval", posterior, retrieval_f1),
            ("H0_profile_retrieval_blend", blended, blend_f1),
        ):
            fold_rows.append({
                "fold": fold, "variant": variant, "macro_f1": score,
                "accuracy": float(accuracy_score(labels[valid_index], classes[probability.argmax(axis=1)])),
                "feature_count": len(h0.names), "delta_vs_h0": score - h0_f1,
            })
        profile_rows.append({
            "fold": fold, "fit_rows": int(len(fit_index)), "validation_rows": int(len(valid_index)),
            "lookup_profile_count": len(lookup.counts), "matched_rows": int(matched.sum()),
            "matched_rate": float(matched.mean()), "mean_matched_support": float(support[matched].mean()) if matched.any() else 0.0,
            "mean_matched_purity": float(purity[matched].mean()) if matched.any() else 0.0,
            "outer_validation_used_for_lookup_fit": False,
            "leakage_check": True, "nan_as_mutation_count": 0,
        })
        warning_count += h0.convergence_warnings
        del h0, lookup, posterior, blended, fit_frame, valid_frame
        gc.collect()

    h0_f1, h0_accuracy, h0_prediction = _metrics(labels, h0_oof, classes)
    retrieval_f1, retrieval_accuracy, retrieval_prediction = _metrics(labels, retrieval_oof, classes)
    blend_f1, blend_accuracy, blend_prediction = _metrics(labels, blend_oof, classes)
    folds = pd.DataFrame(fold_rows)
    h0_folds = folds[folds.variant.eq("H0")].sort_values("fold").macro_f1.to_numpy()
    blend_folds = folds[folds.variant.eq("H0_profile_retrieval_blend")].sort_values("fold").macro_f1.to_numpy()
    recovered = match_oof & (h0_prediction != labels) & (retrieval_prediction == labels)
    broken = match_oof & (h0_prediction == labels) & (retrieval_prediction != labels)
    margin = np.sort(h0_oof, axis=1)[:, -1] - np.sort(h0_oof, axis=1)[:, -2]
    low = margin < LOW_MARGIN
    low_h0 = float(f1_score(labels[low], h0_prediction[low], average="macro", zero_division=0))
    low_blend = float(f1_score(labels[low], blend_prediction[low], average="macro", zero_division=0))
    verdict = _decision(h0_f1, blend_f1, blend_folds - h0_folds, match_oof.mean(), recovered.sum(), broken.sum())
    feature_mean = float(folds[folds.variant.eq("H0")].feature_count.mean())
    summary = pd.DataFrame([
        {"variant": "H0", "oof_macro_f1": h0_f1, "oof_accuracy": h0_accuracy, "feature_count": feature_mean, "convergence_warning_count": warning_count, "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": 0.0},
        {"variant": "profile_retrieval", "oof_macro_f1": retrieval_f1, "oof_accuracy": retrieval_accuracy, "feature_count": feature_mean, "convergence_warning_count": 0, "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": retrieval_f1 - h0_f1},
        {"variant": "H0_profile_retrieval_blend", "oof_macro_f1": blend_f1, "oof_accuracy": blend_accuracy, "feature_count": feature_mean, "convergence_warning_count": warning_count, "leakage_check": True, "nan_as_mutation_count": 0, "delta_vs_h0": blend_f1 - h0_f1},
    ])
    class_rows = []
    for label in classes:
        class_rows.append({
            "class": str(label), "support": int((labels == label).sum()),
            "H0_f1": f1_score(labels == label, h0_prediction == label, zero_division=0),
            "profile_retrieval_f1": f1_score(labels == label, retrieval_prediction == label, zero_division=0),
            "blend_f1": f1_score(labels == label, blend_prediction == label, zero_division=0),
        })
    class_frame = pd.DataFrame(class_rows)
    class_frame["blend_delta_vs_h0"] = class_frame.blend_f1 - class_frame.H0_f1
    recovery = pd.DataFrame({
        "row_index": np.arange(len(train)), "matched_profile": match_oof, "profile_support": support_oof,
        "profile_purity": purity_oof, "h0_prediction": h0_prediction, "retrieval_prediction": retrieval_prediction,
        "blend_prediction": blend_prediction, "true_class": labels, "h0_correct": h0_prediction == labels,
        "retrieval_recovers_h0_error": recovered, "retrieval_breaks_h0_correct": broken,
    })
    topk = pd.DataFrame([
        {"variant": name, **{f"top{k}_recall": _topk_recall(probability, labels, classes, k) for k in (1, 2, 3)}}
        for name, probability in (("H0", h0_oof), ("profile_retrieval", retrieval_oof), ("blend", blend_oof))
    ])
    low_frame = pd.DataFrame([{"group": "low_margin_<0.05", "support": int(low.sum()), "H0_macro_f1": low_h0, "blend_macro_f1": low_blend, "delta": low_blend - low_h0}])
    result = Path(__file__).parent.parent / "result"; result.mkdir(exist_ok=True)
    if not RESULT_COLUMNS["summary"].issubset(summary.columns) or not RESULT_COLUMNS["fold"].issubset(folds.columns):
        raise AssertionError("result schema failure")
    summary.to_csv(result / f"{run_id}_seed42_summary.csv", index=False)
    folds.to_csv(result / f"{run_id}_seed42_fold_metrics.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(result / f"{run_id}_seed42_profile_audit.csv", index=False)
    class_frame.to_csv(result / f"{run_id}_seed42_class_metrics.csv", index=False)
    recovery.to_csv(result / f"{run_id}_seed42_recovery_rows.csv", index=False)
    topk.to_csv(result / f"{run_id}_seed42_topk.csv", index=False)
    low_frame.to_csv(result / f"{run_id}_seed42_low_margin.csv", index=False)
    audit = {"seed": SEED, "outer_splits": 5, "profile_prior_strength": 1.0, "profile_blend_weight": PROFILE_BLEND_WEIGHT,
             "test_read": False, "train_test_concat": False, "fixed_class_gene_mutation_rules": False,
             "profile_vocabulary_source": "outer_fold_train_only", "leakage_check": True,
             "nan_as_mutation_count": 0, "runtime_seconds": perf_counter() - started,
             "low_margin_delta": float(low_blend - low_h0), **verdict}
    (result / f"{run_id}_seed42_leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    ax = folds.pivot(index="fold", columns="variant", values="macro_f1").plot(marker="o", title="H0 vs train-only profile retrieval")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_fold_macro_f1.png", dpi=160); plt.close(ax.figure)
    ax = class_frame.set_index("class").blend_delta_vs_h0.sort_values().plot.barh(title="Profile blend: class F1 delta vs H0")
    ax.figure.tight_layout(); ax.figure.savefig(result / f"{run_id}_seed42_class_f1_delta.png", dpi=160); plt.close(ax.figure)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-profile-retrieval-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run(args.run_id)


if __name__ == "__main__":
    main()
