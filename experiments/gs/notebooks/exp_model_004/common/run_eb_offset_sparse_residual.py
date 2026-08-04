"""Seed 42 nested P1+EB offset sparse residual screen. train only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from eb_offset_residual import (
    HASH_DIMENSION,
    fit_offset_residual,
    hashed_event_matrix,
    offset_audit,
    offset_probability,
)
from selective_eb_gate import SELECTIVE_MARGIN, selective_probability

SEED = 42
EPOCHS = 40
LEARNING_RATE = 0.05
L2 = 0.001
BATCH_SIZE = 256


def project_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "data/raw/train.csv").exists():
            return candidate
    raise FileNotFoundError("project root를 찾지 못했습니다.")


def p1_module():
    common = project_root() / "experiments/gs/notebooks/exp_model_002/common"
    sys.path.insert(0, str(common))
    spec = importlib.util.spec_from_file_location("offset_residual_p1", common / "run_p1_axis.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def token_sets_from_frame(tokens: pd.DataFrame, row_count: int) -> list[set[str]]:
    output = [set() for _ in range(row_count)]
    for row, token in tokens.itertuples(index=False):
        output[int(row)].add(str(token))
    return output


def sparse_residual_features(cache, token_sets, index: np.ndarray) -> csr_matrix:
    raw = cache.mutation[index]
    hashed = hashed_event_matrix(token_sets, index, HASH_DIMENSION)
    return hstack([raw, hashed], format="csr", dtype=np.float32)


def eb_probability(module, context, token_sets, fit_index, out_index, seed, fold_code):
    base, _, cache, _, y, classes = context
    matrix, _ = base._matrix(cache, fit_index, y[fit_index], contrast=True, functional=False, scale_numeric=False)
    inner_score = module.enriched(token_sets, fit_index, fit_index, y, classes, seed + fold_code, empirical=True, inner=True)
    out_score = module.enriched(token_sets, fit_index, out_index, y, classes, seed + fold_code, empirical=True)
    mean = inner_score.mean(axis=0, keepdims=True)
    std = np.maximum(inner_score.std(axis=0, keepdims=True), 1e-6)
    x_train = hstack([matrix[fit_index], csr_matrix((inner_score - mean) / std)], format="csr")
    x_out = hstack([matrix[out_index], csr_matrix((out_score - mean) / std)], format="csr")
    model, warning_count = module.fit_lr(x_train, y[fit_index], seed)
    return module.normalize_proba(model.predict_proba(x_out)), warning_count, x_train.shape[1]


def p1_non_eb_probability(module, context, fit_index, out_index, seed, fold_code):
    _, _, _, _, y, _ = context
    x_train, x_out, _, _ = module.p1_parts(context, fit_index, out_index, seed, fold_code)
    model, warning_count = module.fit_lr(x_train, y[fit_index], seed)
    return module.normalize_proba(model.predict_proba(x_out)), warning_count


def class_rows(y, classes, probabilities, seed):
    rows = []
    for label in classes:
        row = {"seed": seed, "class": label, "support": int((y == label).sum())}
        for variant, probability in probabilities.items():
            row[variant] = f1_score(y, classes[probability.argmax(1)], labels=[label], average="macro", zero_division=0)
        row["delta_residual_vs_gate"] = row["eb_offset_residual"] - row["selective_EB_gate"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-id", default="exp-eb-offset-sparse-residual-01")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("이 screen은 사전 고정 seed 42만 허용합니다.")

    started = time.time()
    module = p1_module()
    context = module.legacy_context()
    base, _, cache, tokens, y, classes = context
    train = pd.read_csv(project_root() / "data/raw/train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN 계약 위반")
    token_sets = token_sets_from_frame(tokens, len(y))
    folds = module.fixed_folds(y, args.seed)
    n_rows, n_classes = len(y), len(classes)
    peb_all = np.zeros((n_rows, n_classes), dtype=np.float32)
    p0_all = np.zeros_like(peb_all)
    residual_all = np.zeros_like(peb_all)
    warnings_total = 0
    base_feature_count = 0
    fold_rows, audit_rows, loss_rows = [], [], []

    for outer_fold, (outer_train, outer_valid) in enumerate(folds, 1):
        inner_eb = np.zeros((len(outer_train), n_classes), dtype=np.float32)
        inner_seen = np.zeros(len(outer_train), dtype=bool)
        splitter = StratifiedKFold(5, shuffle=True, random_state=args.seed * 1000 + outer_fold)
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(splitter.split(outer_train, y[outer_train]), 1):
            fit_index = outer_train[inner_train_local]
            out_index = outer_train[inner_valid_local]
            probability, warning_count, _ = eb_probability(
                module, context, token_sets, fit_index, out_index, args.seed, outer_fold * 10 + inner_fold
            )
            inner_eb[inner_valid_local] = probability
            inner_seen[inner_valid_local] = True
            warnings_total += warning_count
        audit = offset_audit(outer_train, outer_train[inner_seen], outer_valid)
        if not audit["offset_train_rows_are_inner_oof"] or audit["outer_validation_used_for_residual_fit"]:
            raise AssertionError("residual offset은 outer-train inner OOF만 사용해야 합니다.")

        x_train = sparse_residual_features(cache, token_sets, outer_train)
        x_valid = sparse_residual_features(cache, token_sets, outer_valid)
        y_train_index = np.searchsorted(classes, y[outer_train])
        counts = np.bincount(y_train_index, minlength=n_classes).astype(np.float64)
        class_weight = len(outer_train) / np.maximum(n_classes * counts, 1.0)
        weight, bias, history = fit_offset_residual(
            x_train, y_train_index, np.log(np.clip(inner_eb, 1e-7, 1.0)), class_weight,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, l2=L2, batch_size=BATCH_SIZE, seed=args.seed + outer_fold,
        )
        peb, warning_eb, base_feature_count = eb_probability(
            module, context, token_sets, outer_train, outer_valid, args.seed, outer_fold
        )
        p0, warning_p0 = p1_non_eb_probability(module, context, outer_train, outer_valid, args.seed, outer_fold)
        warnings_total += warning_eb + warning_p0
        residual = offset_probability(np.log(np.clip(peb, 1e-7, 1.0)), weight, bias, x_valid)
        peb_all[outer_valid] = peb
        p0_all[outer_valid] = p0
        residual_all[outer_valid] = residual
        for epoch, loss in enumerate(history, 1):
            loss_rows.append({"seed": args.seed, "fold": outer_fold, "epoch": epoch, "weighted_loss": loss})
        gate, _ = selective_probability(p0, peb)
        for variant, probability in (("P1_EB", peb), ("selective_EB_gate", gate), ("eb_offset_residual", residual)):
            fold_rows.append({"seed": args.seed, "fold": outer_fold, "variant": variant,
                              "macro_f1": f1_score(y[outer_valid], classes[probability.argmax(1)], average="macro"),
                              "feature_count": x_train.shape[1] if variant == "eb_offset_residual" else base_feature_count})
        audit_rows.append({"seed": args.seed, "fold": outer_fold, **audit,
                           "inner_oof_rows": len(outer_train), "outer_validation_rows": len(outer_valid),
                           "residual_feature_count": x_train.shape[1]})

    gate_all, gate_mask = selective_probability(p0_all, peb_all)
    variants = {"P1_EB": peb_all, "selective_EB_gate": gate_all, "eb_offset_residual": residual_all}
    rows = []
    for variant, probability in variants.items():
        rows.append({"variant": variant, "oof_macro_f1": f1_score(y, classes[probability.argmax(1)], average="macro"),
                     "feature_count": x_train.shape[1] if variant == "eb_offset_residual" else base_feature_count,
                     "convergence_warning_count": warnings_total if variant == "eb_offset_residual" else 0,
                     "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": time.time() - started})
    summary = pd.DataFrame(rows)
    gate_score = float(summary.loc[summary.variant.eq("selective_EB_gate"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_gate"] = summary.oof_macro_f1 - gate_score
    margin = np.partition(peb_all, kth=-2, axis=1)[:, -1] - np.partition(peb_all, kth=-2, axis=1)[:, -2]
    low = margin < SELECTIVE_MARGIN
    low_rows = [{"variant": variant, "group": "low_margin_<0.05", "support": int(low.sum()),
                 "macro_f1": f1_score(y[low], classes[probability[low].argmax(1)], average="macro", zero_division=0)}
                for variant, probability in variants.items()]
    out = Path(__file__).parent.parent / "result"
    out.mkdir(exist_ok=True)
    summary.to_csv(out / f"{args.run_id}_seed{args.seed}_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / f"{args.run_id}_seed{args.seed}_fold_metrics.csv", index=False)
    pd.DataFrame(class_rows(y, classes, variants, args.seed)).to_csv(out / f"{args.run_id}_seed{args.seed}_class_metrics.csv", index=False)
    pd.DataFrame(low_rows).to_csv(out / f"{args.run_id}_seed{args.seed}_low_margin_metrics.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(out / f"{args.run_id}_seed{args.seed}_epoch_loss.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / f"{args.run_id}_seed{args.seed}_offset_audit.csv", index=False)
    pd.DataFrame({"true_class": y, "eb_margin": margin, "gate_uses_non_eb": gate_mask,
                  **{f"p1_eb__{label}": peb_all[:, index] for index, label in enumerate(classes)},
                  **{f"gate__{label}": gate_all[:, index] for index, label in enumerate(classes)},
                  **{f"residual__{label}": residual_all[:, index] for index, label in enumerate(classes)}}).to_csv(out / f"{args.run_id}_seed{args.seed}_oof_probabilities.csv", index=False)
    audit = {"test_read": False, "submission_created": False, "seed": args.seed, "outer_folds": 5, "inner_folds": 5,
             "offset_train_inner_oof_only": True, "offset_zero_initialized": True, "hash_dimension": HASH_DIMENSION,
             "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "l2": L2, "batch_size": BATCH_SIZE,
             "threshold_retuned": False, "selective_gate_threshold": SELECTIVE_MARGIN,
             "leakage_check": True, "nan_as_mutation_count": 0}
    (out / f"{args.run_id}_seed{args.seed}_leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
