"""Nested inner-OOF all-class candidate evidence ranker screen (train only)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from candidate_evidence_ranker import (
    build_ranker_audit,
    candidate_matrix,
    candidate_scores_to_probability,
    topk_metrics,
)
from selective_eb_gate import SELECTIVE_MARGIN, selective_probability

SEED = 42
RARE_SUPPORT_CUTOFF = 10


def project_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "data/raw/train.csv").exists():
            return candidate
    raise FileNotFoundError("project root를 찾지 못했습니다.")


def p1_module():
    common = project_root() / "experiments/gs/notebooks/exp_model_002/common"
    sys.path.insert(0, str(common))
    spec = importlib.util.spec_from_file_location("candidate_ranker_p1", common / "run_p1_axis.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def make_token_sets(tokens: pd.DataFrame, row_count: int) -> list[set[str]]:
    output = [set() for _ in range(row_count)]
    for row, token in tokens.itertuples(index=False):
        output[int(row)].add(str(token))
    return output


def evidence_and_rare_split(token_sets, fit_index, out_index, y, classes):
    """fit partition support만으로 EB evidence와 rare-token evidence를 분리한다."""
    from p1_core import apply_log_odds, fit_log_odds

    support: dict[str, int] = {}
    for row in fit_index:
        for token in token_sets[int(row)]:
            support[token] = support.get(token, 0) + 1
    weights = fit_log_odds(
        [token_sets[int(row)] for row in fit_index], y[fit_index], classes, empirical_bayes=True
    )
    rare_weights = {token: weight for token, weight in weights.items() if support[token] < RARE_SUPPORT_CUTOFF}
    full = apply_log_odds([token_sets[int(row)] for row in out_index], weights, classes)
    rare = apply_log_odds([token_sets[int(row)] for row in out_index], rare_weights, classes)
    return full, rare


def eb_design_matrix(module, context, token_sets, fit_index, out_index, seed, fold_code):
    """P1+EB의 matrix 조립을 run_p1_axis.eb_cv와 동일하게 재현한다."""
    base, _, cache, _, y, classes = context
    matrix, _ = base._matrix(cache, fit_index, y[fit_index], contrast=True, functional=False, scale_numeric=False)
    inner_score = module.enriched(token_sets, fit_index, fit_index, y, classes, seed + fold_code, empirical=True, inner=True)
    outer_score = module.enriched(token_sets, fit_index, out_index, y, classes, seed + fold_code, empirical=True)
    evidence, rare_evidence = evidence_and_rare_split(token_sets, fit_index, out_index, y, classes)
    if not np.allclose(outer_score, evidence, atol=1e-6):
        raise AssertionError("EB evidence가 P1+EB 기준선 score와 일치하지 않습니다.")
    mean = inner_score.mean(axis=0, keepdims=True)
    std = np.maximum(inner_score.std(axis=0, keepdims=True), 1e-6)
    x_train = hstack([matrix[fit_index], csr_matrix((inner_score - mean) / std)], format="csr")
    x_out = hstack([matrix[out_index], csr_matrix((outer_score - mean) / std)], format="csr")
    return x_train, x_out, evidence, rare_evidence


def fit_base_pair(module, context, token_sets, fit_index, out_index, seed, fold_code):
    """한 fit partition에서 P1 non-EB/P1+EB와 candidate evidence를 얻는다."""
    _, _, _, _, y, classes = context
    x0_train, x0_out, _, _ = module.p1_parts(context, fit_index, out_index, seed, fold_code)
    model0, warning0 = module.fit_lr(x0_train, y[fit_index], seed)
    p_non_eb = module.normalize_proba(model0.predict_proba(x0_out))
    xeb_train, xeb_out, evidence, rare_evidence = eb_design_matrix(
        module, context, token_sets, fit_index, out_index, seed, fold_code
    )
    model_eb, warning_eb = module.fit_lr(xeb_train, y[fit_index], seed)
    p_eb = module.normalize_proba(model_eb.predict_proba(xeb_out))
    return p_non_eb, p_eb, evidence, rare_evidence, warning0 + warning_eb, xeb_train.shape[1]


def fit_ranker(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    model = LogisticRegression(
        solver="lbfgs", C=0.07, max_iter=2000, class_weight="balanced", random_state=seed
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    warning_count = sum(isinstance(item.message, ConvergenceWarning) for item in captured)
    return scaler, model, warning_count


def class_metrics(y, classes, probability_by_variant, seed):
    rows = []
    for label in classes:
        row = {"seed": seed, "class": label, "support": int((y == label).sum())}
        for name, probability in probability_by_variant.items():
            row[name] = f1_score(y, classes[probability.argmax(1)], labels=[label], average="macro", zero_division=0)
        row["delta_ranker_vs_gate"] = row["all_class_ranker"] - row["selective_EB_gate"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-id", default="exp-all-class-evidence-ranker-01")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("이 1차 screen은 사전 고정된 seed 42만 허용합니다.")

    start = time.time()
    module = p1_module()
    context = module.legacy_context()
    base, _, cache, tokens, y, classes = context
    train = pd.read_csv(project_root() / "data/raw/train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN 계약 위반")
    token_sets = make_token_sets(tokens, len(y))
    burden = np.asarray(cache.burden)[:, 0].astype(np.float64)
    folds = module.fixed_folds(y, args.seed)
    n_samples, n_classes = len(y), len(classes)
    p0_all = np.zeros((n_samples, n_classes), dtype=np.float32)
    peb_all = np.zeros_like(p0_all)
    ranker_all = np.zeros_like(p0_all)
    fold_rows, ranker_audit_rows = [], []
    warning_total = 0
    ranker_feature_count = None

    for outer_fold, (outer_train, outer_valid) in enumerate(folds, 1):
        inner_p0 = np.zeros((len(outer_train), n_classes), dtype=np.float32)
        inner_peb = np.zeros_like(inner_p0)
        inner_evidence = np.zeros_like(inner_p0)
        inner_rare = np.zeros_like(inner_p0)
        inner_seen = np.zeros(len(outer_train), dtype=bool)
        inner_splitter = StratifiedKFold(5, shuffle=True, random_state=args.seed * 1000 + outer_fold)
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(inner_splitter.split(outer_train, y[outer_train]), 1):
            fit_index = outer_train[inner_train_local]
            out_index = outer_train[inner_valid_local]
            p0, peb, evidence, rare, warning_count, _ = fit_base_pair(
                module, context, token_sets, fit_index, out_index, args.seed, outer_fold * 10 + inner_fold
            )
            inner_p0[inner_valid_local] = p0
            inner_peb[inner_valid_local] = peb
            inner_evidence[inner_valid_local] = evidence
            inner_rare[inner_valid_local] = rare
            inner_seen[inner_valid_local] = True
            warning_total += warning_count

        audit = build_ranker_audit(outer_train, outer_train[inner_seen], outer_valid)
        if not audit["ranker_training_rows_are_inner_oof"] or audit["outer_validation_used_for_ranker_fit"]:
            raise AssertionError("ranker는 outer train 내부 inner OOF만으로 학습해야 합니다.")
        x_inner, patient_inner, candidate_inner = candidate_matrix(
            inner_p0, inner_peb, inner_evidence, burden[outer_train], n_classes, inner_rare
        )
        y_inner = (candidate_inner == np.searchsorted(classes, y[outer_train][patient_inner])).astype(int)
        scaler, ranker, warning_ranker = fit_ranker(x_inner, y_inner, args.seed)
        warning_total += warning_ranker

        p0, peb, evidence, rare, warning_count, base_feature_count = fit_base_pair(
            module, context, token_sets, outer_train, outer_valid, args.seed, outer_fold
        )
        warning_total += warning_count
        x_valid, patient_valid, _ = candidate_matrix(p0, peb, evidence, burden[outer_valid], n_classes, rare)
        if not np.array_equal(patient_valid, np.repeat(np.arange(len(outer_valid)), n_classes)):
            raise AssertionError("validation candidate 행 정렬 계약 위반")
        ranker_probability = candidate_scores_to_probability(
            ranker.decision_function(scaler.transform(x_valid)), len(outer_valid), n_classes
        )
        p0_all[outer_valid] = p0
        peb_all[outer_valid] = peb
        ranker_all[outer_valid] = ranker_probability
        ranker_feature_count = x_inner.shape[1]
        gate, _ = selective_probability(p0, peb)
        for variant, probability in (("P1_EB", peb), ("selective_EB_gate", gate), ("all_class_ranker", ranker_probability)):
            fold_rows.append({"seed": args.seed, "fold": outer_fold, "variant": variant,
                              "macro_f1": f1_score(y[outer_valid], classes[probability.argmax(1)], average="macro"),
                              "feature_count": ranker_feature_count if variant == "all_class_ranker" else base_feature_count})
        ranker_audit_rows.append({"seed": args.seed, "fold": outer_fold, **audit,
                                  "inner_oof_rows": len(outer_train), "outer_validation_rows": len(outer_valid),
                                  "ranker_feature_count": ranker_feature_count})

    gate_all, gate_mask = selective_probability(p0_all, peb_all)
    variants = {"P1_EB": peb_all, "selective_EB_gate": gate_all, "all_class_ranker": ranker_all}
    metric_rows = []
    for name, probability in variants.items():
        metrics = topk_metrics(y, probability, classes)
        metric_rows.append({"variant": name, "oof_macro_f1": f1_score(y, classes[probability.argmax(1)], average="macro"),
                            "feature_count": ranker_feature_count if name == "all_class_ranker" else base_feature_count,
                            "convergence_warning_count": warning_total if name == "all_class_ranker" else 0,
                            "leakage_check": True, "nan_as_mutation_count": 0,
                            "runtime_seconds": time.time() - start, **metrics})
    summary = pd.DataFrame(metric_rows)
    gate_score = float(summary.loc[summary.variant.eq("selective_EB_gate"), "oof_macro_f1"].iloc[0])
    summary["delta_vs_gate"] = summary.oof_macro_f1 - gate_score

    margin = np.partition(peb_all, kth=-2, axis=1)[:, -1] - np.partition(peb_all, kth=-2, axis=1)[:, -2]
    low_margin = margin < SELECTIVE_MARGIN
    low_rows = []
    for name, probability in variants.items():
        low_rows.append({"variant": name, "group": "low_margin_<0.05", "support": int(low_margin.sum()),
                         "macro_f1": f1_score(y[low_margin], classes[probability[low_margin].argmax(1)], average="macro", zero_division=0)})
    class_rows = class_metrics(y, classes, variants, args.seed)
    out = Path(__file__).parent.parent / "result"
    out.mkdir(exist_ok=True)
    summary.to_csv(out / f"{args.run_id}_seed{args.seed}_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / f"{args.run_id}_seed{args.seed}_fold_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / f"{args.run_id}_seed{args.seed}_class_metrics.csv", index=False)
    pd.DataFrame(low_rows).to_csv(out / f"{args.run_id}_seed{args.seed}_low_margin_metrics.csv", index=False)
    pd.DataFrame(ranker_audit_rows).to_csv(out / f"{args.run_id}_seed{args.seed}_ranker_audit.csv", index=False)
    pd.DataFrame({"true_class": y, "eb_margin": margin, "gate_uses_non_eb": gate_mask,
                  **{f"p1_eb__{label}": peb_all[:, index] for index, label in enumerate(classes)},
                  **{f"gate__{label}": gate_all[:, index] for index, label in enumerate(classes)},
                  **{f"ranker__{label}": ranker_all[:, index] for index, label in enumerate(classes)}}).to_csv(
        out / f"{args.run_id}_seed{args.seed}_oof_probabilities.csv", index=False
    )
    audit = {"test_read": False, "submission_created": False, "seed": args.seed,
             "outer_folds": 5, "inner_folds": 5, "ranker_training_inner_oof_only": True,
             "ranker_feature_fit_partition": "outer_train_inner_oof", "threshold_retuned": False,
             "selective_gate_threshold": SELECTIVE_MARGIN, "rare_support_cutoff": RARE_SUPPORT_CUTOFF,
             "leakage_check": True, "nan_as_mutation_count": 0}
    (out / f"{args.run_id}_seed{args.seed}_leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
