"""P1+EB 기준선을 변경하지 않고 신규 축을 OOF로 비교하는 실행기."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score

from p1_eb_axes import (
    build_event_rows,
    cross_fitted_scores,
    exp002_common,
    fit_point_process,
    legacy_runner_module,
    lowrank_weight_builder,
    parser_tables,
    point_score_matrix,
    project_root,
    support_tables,
    summarize_ranks,
    token_sets_from_events,
    apply_weight_scores,
)


def result_dir() -> Path:
    output = Path(__file__).parent.parent / "result"
    output.mkdir(exist_ok=True)
    return output


def context():
    legacy = legacy_runner_module()
    base, enrichment_ref, cache, tokens, labels, classes = legacy.legacy_context()
    train = pd.read_csv(project_root() / "data/raw/train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert int(train[genes].isna().sum().sum()) == 0, "train NaN 계약 위반"
    events = build_event_rows(train[genes])
    return legacy, base, enrichment_ref, cache, tokens, labels, classes, events


def legacy_token_sets(tokens: pd.DataFrame, n_rows: int) -> list[set[str]]:
    output = [set() for _ in range(n_rows)]
    for row, token in tokens.itertuples(index=False):
        output[int(row)].add(token)
    return output


def eb_scores(legacy, token_sets, fit_idx, out_idx, labels, classes, seed, *, inner: bool) -> np.ndarray:
    return legacy.enriched(token_sets, fit_idx, out_idx, labels, classes, seed, empirical=True, inner=inner)


def standardize(train_score: np.ndarray, valid_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_score.mean(axis=0, keepdims=True)
    std = np.maximum(train_score.std(axis=0, keepdims=True), 1e-6)
    return (train_score - mean) / std, (valid_score - mean) / std


def eb_base_parts(legacy, base, cache, base_tokens, train_idx, valid_idx, labels, classes, seed, fold):
    matrix, _ = base._matrix(cache, train_idx, labels[train_idx], contrast=True, functional=False, scale_numeric=False)
    inner = eb_scores(legacy, base_tokens, train_idx, train_idx, labels, classes, seed + fold, inner=True)
    valid = eb_scores(legacy, base_tokens, train_idx, valid_idx, labels, classes, seed + fold, inner=False)
    inner, valid = standardize(inner, valid)
    return matrix, inner, valid


def point_cv(seed: int):
    legacy, base, enrichment_ref, cache, tokens, labels, classes, events = context()
    _, baseline_probability, baseline_folds, baseline_warn = legacy.eb_cv((base, enrichment_ref, cache, tokens, labels, classes), seed)
    base_tokens = legacy_token_sets(tokens, len(labels))
    probability = np.zeros_like(baseline_probability)
    folds = []
    warnings = 0
    for fold, (train_idx, valid_idx) in enumerate(legacy.fixed_folds(labels, seed), 1):
        matrix, eb_train, eb_valid = eb_base_parts(legacy, base, cache, base_tokens, train_idx, valid_idx, labels, classes, seed, fold)
        make_model = lambda idx: fit_point_process(events, idx, labels, classes, legacy.fit_log_odds)
        apply = lambda model, idx: point_score_matrix(model, [events[i] for i in idx])
        point_train = cross_fitted_scores(make_model, apply, train_idx, labels, seed * 100 + fold, len(classes) * 3)
        point_valid = apply(make_model(train_idx), valid_idx)
        point_train, point_valid = standardize(point_train, point_valid)
        x_train = hstack([matrix[train_idx], csr_matrix(eb_train), csr_matrix(point_train)], format="csr")
        x_valid = hstack([matrix[valid_idx], csr_matrix(eb_valid), csr_matrix(point_valid)], format="csr")
        model, warning_count = legacy.fit_lr(x_train, labels[train_idx], seed)
        warnings += warning_count
        probability[valid_idx] = legacy.normalize_proba(model.predict_proba(x_valid))
        folds.append({"fold": fold, "macro_f1": f1_score(labels[valid_idx], classes[probability[valid_idx].argmax(1)], average="macro"), "feature_count": x_train.shape[1]})
    return classes, baseline_probability, probability, baseline_folds, pd.DataFrame(folds), baseline_warn, warnings


def multieb_cv(seed: int, rank: int):
    legacy, base, enrichment_ref, cache, tokens, labels, classes, _ = context()
    _, baseline_probability, baseline_folds, baseline_warn = legacy.eb_cv((base, enrichment_ref, cache, tokens, labels, classes), seed)
    base_tokens = legacy_token_sets(tokens, len(labels))
    probability = np.zeros_like(baseline_probability)
    folds = []
    warnings = 0
    for fold, (train_idx, valid_idx) in enumerate(legacy.fixed_folds(labels, seed), 1):
        matrix, _, _ = eb_base_parts(legacy, base, cache, base_tokens, train_idx, valid_idx, labels, classes, seed, fold)
        builder = lambda idx: lowrank_weight_builder(base_tokens, idx, labels, classes, legacy.fit_log_odds, rank)
        applier = lambda model, idx: apply_weight_scores(base_tokens, idx, model, len(classes))
        score_train = cross_fitted_scores(builder, applier, train_idx, labels, seed * 100 + fold, len(classes))
        score_valid = applier(builder(train_idx), valid_idx)
        score_train, score_valid = standardize(score_train, score_valid)
        x_train = hstack([matrix[train_idx], csr_matrix(score_train)], format="csr")
        x_valid = hstack([matrix[valid_idx], csr_matrix(score_valid)], format="csr")
        model, warning_count = legacy.fit_lr(x_train, labels[train_idx], seed)
        warnings += warning_count
        probability[valid_idx] = legacy.normalize_proba(model.predict_proba(x_valid))
        folds.append({"fold": fold, "macro_f1": f1_score(labels[valid_idx], classes[probability[valid_idx].argmax(1)], average="macro"), "feature_count": x_train.shape[1]})
    return classes, baseline_probability, probability, baseline_folds, pd.DataFrame(folds), baseline_warn, warnings


def save_comparison(run_id: str, seed: int, labels, classes, variants: list[tuple[str, np.ndarray, pd.DataFrame, int]], baseline: tuple[np.ndarray, pd.DataFrame, int], axis: str):
    output = result_dir()
    base_probability, base_folds, base_warn = baseline
    rows = [{
        "variant": "P1+EB LR", "oof_macro_f1": f1_score(labels, classes[base_probability.argmax(1)], average="macro"),
        "feature_count": base_folds.feature_count.mean(), "convergence_warning_count": base_warn,
        "leakage_check": True, "nan_as_mutation_count": 0,
    }]
    probability_columns = {"true_class": labels, **{f"baseline_{c}": base_probability[:, i] for i, c in enumerate(classes)}}
    fold_frames = [base_folds.assign(variant="P1+EB LR")]
    for name, probability, folds, warnings in variants:
        rows.append({
            "variant": name, "oof_macro_f1": f1_score(labels, classes[probability.argmax(1)], average="macro"),
            "feature_count": folds.feature_count.mean(), "convergence_warning_count": warnings,
            "leakage_check": True, "nan_as_mutation_count": 0,
        })
        probability_columns.update({f"{name}_{c}": probability[:, i] for i, c in enumerate(classes)})
        fold_frames.append(folds.assign(variant=name))
    summary = pd.DataFrame(rows)
    summary["delta_vs_eb"] = summary.oof_macro_f1 - float(summary.loc[summary.variant.eq("P1+EB LR"), "oof_macro_f1"].iloc[0])
    summary.to_csv(output / f"{run_id}_seed{seed}_summary.csv", index=False)
    pd.DataFrame(probability_columns).to_csv(output / f"{run_id}_seed{seed}_oof_probabilities.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(output / f"{run_id}_seed{seed}_fold_metrics.csv", index=False)
    (output / f"{run_id}_seed{seed}_leakage_audit.json").write_text(json.dumps({
        "train_only": True, "test_read": False, "fold_train_supervised_statistics": True,
        "nan_as_mutation_count": 0, "axis": axis, "baseline": "P1+EB LR",
        "reference": str(exp002_common()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def run_audit(run_id: str, seed: int):
    legacy, base, enrichment_ref, cache, tokens, labels, classes, events = context()
    _, probability, folds, warnings = legacy.eb_cv((base, enrichment_ref, cache, tokens, labels, classes), seed)
    rank_summary, rank_rows = summarize_ranks(probability, labels, classes)
    coverage, class_coverage = parser_tables(events, labels)
    structure = support_tables(events)
    output = result_dir()
    rank_summary.assign(seed=seed, oof_macro_f1=f1_score(labels, classes[probability.argmax(1)], average="macro"), convergence_warning_count=warnings, leakage_check=True, nan_as_mutation_count=0).to_csv(output / f"{run_id}_seed{seed}_summary.csv", index=False)
    rank_rows.to_csv(output / f"{run_id}_seed{seed}_rank_rows.csv", index=False)
    coverage.to_csv(output / f"{run_id}_seed{seed}_parser_coverage.csv", index=False)
    class_coverage.to_csv(output / f"{run_id}_seed{seed}_parser_class_coverage.csv", index=False)
    structure.to_csv(output / f"{run_id}_seed{seed}_structure_support.csv", index=False)
    pd.DataFrame({"true_class": labels, **{f"eb_{c}": probability[:, i] for i, c in enumerate(classes)}}).to_csv(output / f"{run_id}_seed{seed}_oof_probabilities.csv", index=False)
    folds.to_csv(output / f"{run_id}_seed{seed}_fold_metrics.csv", index=False)
    (output / f"{run_id}_seed{seed}_leakage_audit.json").write_text(json.dumps({
        "train_only": True, "test_read": False, "fold_train_supervised_statistics": True,
        "nan_as_mutation_count": 0, "axis": "audit", "baseline": "P1+EB LR",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(rank_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=("audit", "point", "multieb"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    start = time.time()
    if args.axis == "audit":
        run_audit(args.run_id, args.seed)
    elif args.axis == "point":
        classes, base_prob, point_prob, base_folds, point_folds, base_warn, point_warn = point_cv(args.seed)
        _, _, _, _, _, labels, _, _ = context()
        save_comparison(args.run_id, args.seed, labels, classes, [("point_process_eb", point_prob, point_folds, point_warn)], (base_prob, base_folds, base_warn), args.axis)
    else:
        classes, base_prob, rank4, base_folds, folds4, base_warn, warn4 = multieb_cv(args.seed, 4)
        _, _, rank8, _, folds8, _, warn8 = multieb_cv(args.seed, 8)
        _, _, _, _, _, labels, _, _ = context()
        save_comparison(args.run_id, args.seed, labels, classes, [("multieb_rank4", rank4, folds4, warn4), ("multieb_rank8", rank8, folds8, warn8)], (base_prob, base_folds, base_warn), args.axis)
    print(f"runtime_seconds={time.time() - start:.2f}")


if __name__ == "__main__":
    main()
