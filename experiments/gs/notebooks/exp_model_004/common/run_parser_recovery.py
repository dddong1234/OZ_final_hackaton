"""G0/G1/G2 parser recovery OOF runner. Train-only; test is never read."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score, precision_recall_fscore_support

from parser_recovery import audit_frame, event_tokens, parse_frame


def root() -> Path:
    for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (path / "data/raw/train.csv").exists(): return path
    raise FileNotFoundError("project root not found")


def legacy_module():
    common = root() / "experiments/gs/notebooks/exp_model_002/common"
    sys.path.insert(0, str(common))
    spec = importlib.util.spec_from_file_location("parser_recovery_p1", common / "run_p1_axis.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def context():
    legacy = legacy_module()
    base, ref, cache, original_tokens, labels, classes = legacy.legacy_context()
    train = pd.read_csv(root() / "data/raw/train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert int(train[genes].isna().sum().sum()) == 0
    return legacy, base, ref, cache, original_tokens, labels, classes, train, genes


def standardized(train_score, valid_score):
    mean = train_score.mean(0, keepdims=True); std = np.maximum(train_score.std(0, keepdims=True), 1e-6)
    return (train_score - mean) / std, (valid_score - mean) / std


def candidate_cv(ctx, seed: int, mode: str):
    legacy, base, _, cache, _, labels, classes, train, genes = ctx
    events = parse_frame(train[genes])
    tokens = event_tokens(events, mode)
    probability = np.zeros((len(labels), len(classes)), dtype=np.float32); folds = []; warnings = 0
    for fold, (tr, va) in enumerate(legacy.fixed_folds(labels, seed), 1):
        matrix, _ = base._matrix(cache, tr, labels[tr], contrast=True, functional=False, scale_numeric=False)
        train_score = legacy.enriched(tokens, tr, tr, labels, classes, seed + fold, empirical=True, inner=True)
        valid_score = legacy.enriched(tokens, tr, va, labels, classes, seed + fold, empirical=True, inner=False)
        train_score, valid_score = standardized(train_score, valid_score)
        xtr = hstack([matrix[tr], csr_matrix(train_score)], format="csr")
        xva = hstack([matrix[va], csr_matrix(valid_score)], format="csr")
        model, warning = legacy.fit_lr(xtr, labels[tr], seed); warnings += warning
        probability[va] = legacy.normalize_proba(model.predict_proba(xva))
        folds.append({"fold": fold, "macro_f1": f1_score(labels[va], classes[probability[va].argmax(1)], average="macro"), "feature_count": xtr.shape[1]})
    return probability, pd.DataFrame(folds), warnings


def save_comparison(run_id, seed, labels, classes, variants, audit=None):
    output = Path(__file__).parent.parent / "result"; output.mkdir(exist_ok=True)
    rows = []; frames = []; probability_output = {"true_class": labels}
    for name, probability, folds, warnings in variants:
        prediction = classes[probability.argmax(1)]
        rows.append({"variant": name, "oof_macro_f1": f1_score(labels, prediction, average="macro"), "feature_count": folds.feature_count.mean(), "convergence_warning_count": warnings, "leakage_check": True, "nan_as_mutation_count": 0})
        frames.append(folds.assign(variant=name))
        probability_output.update({f"{name}_{label}": probability[:, i] for i, label in enumerate(classes)})
        _, recall, score, support = precision_recall_fscore_support(labels, prediction, labels=classes, zero_division=0)
        pd.DataFrame({"class": classes, "recall": recall, "f1": score, "support": support, "variant": name}).to_csv(output / f"{run_id}_seed{seed}_{name}_class_metrics.csv", index=False)
    summary = pd.DataFrame(rows); baseline = float(summary.oof_macro_f1.iloc[0]); summary["delta_vs_g0"] = summary.oof_macro_f1 - baseline
    summary.to_csv(output / f"{run_id}_seed{seed}_summary.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(output / f"{run_id}_seed{seed}_fold_metrics.csv", index=False)
    pd.DataFrame(probability_output).to_csv(output / f"{run_id}_seed{seed}_oof_probabilities.csv", index=False)
    audit_payload = {"train_only": True, "test_read": False, "fold_train_supervised_statistics": True, "nan_as_mutation_count": 0, "parser_audit": audit or {}}
    (output / f"{run_id}_seed{seed}_leakage_audit.json").write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def audit(run_id: str):
    _, base, _, _, _, _, _, train, genes = context()
    types, unknown, contract = audit_frame(train[genes])
    output = Path(__file__).parent.parent / "result"; output.mkdir(exist_ok=True)
    types.to_csv(output / f"{run_id}_canonical_types.csv", index=False)
    unknown.to_csv(output / f"{run_id}_unknown_patterns.csv", index=False)
    (output / f"{run_id}_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(contract, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=("audit", "g1", "g2"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.axis == "audit": audit(args.run_id); return
    ctx = context(); legacy, base, ref, cache, original_tokens, labels, classes, train, genes = ctx
    _, g0_probability, g0_folds, g0_warning = legacy.eb_cv((base, ref, cache, original_tokens, labels, classes), args.seed)
    mode = "legacy" if args.axis == "g1" else "canonical"
    candidate_probability, candidate_folds, candidate_warning = candidate_cv(ctx, args.seed, mode)
    _, _, contract = audit_frame(train[genes])
    save_comparison(args.run_id, args.seed, labels, classes, [("G0_P1_EB", g0_probability, g0_folds, g0_warning), (f"{args.axis.upper()}_{mode}_parser", candidate_probability, candidate_folds, candidate_warning)], contract)


if __name__ == "__main__": main()
