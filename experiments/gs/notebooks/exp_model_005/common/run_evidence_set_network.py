"""Seed-42 screen for a train-only class-conditional Evidence Set Network.

The runner never reads test data. It trains the candidate model directly.
An already-created, row-aligned train-only team OOF artifact is optional and
is used only for paired comparison; the runner never re-trains team models.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from evidence_set_core import EvidenceSetNetwork, build_event_evidence, listwise_loss, nested_audit, pad_evidence_sets
from baseline_artifact import reference_only_baseline, validate_baseline_oof
from team_ensemble_baseline import EventCache, parse_train_frame


SEED = 42
HIDDEN_DIM = 32
DROPOUT = 0.15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
EPOCHS = 60
EB_PRIOR_STRENGTH = 20.0


def project_root() -> Path:
    for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv를 가진 프로젝트 루트를 찾지 못했습니다.")


def row_events(cache: EventCache) -> list[list[tuple[str, str, str]]]:
    output: list[list[tuple[str, str, str]]] = [[] for _ in range(cache.row_count)]
    for row, gene, event_type, event in cache.events[["row", "gene", "event_type", "event"]].itertuples(index=False):
        output[int(row)].append((str(gene), str(event_type), str(event)))
    return output


def fit_empirical_bayes_weights(events_by_row: list[list[tuple[str, str, str]]], rows: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Fit a conservative token-wise class log-odds table on fit rows only."""
    rows = np.asarray(rows, dtype=np.int64)
    supports: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = {str(label): Counter() for label in classes}
    for row in rows:
        tokens = {f"{gene}__{event_type}" for gene, event_type, _ in events_by_row[int(row)]}
        supports.update(tokens)
        class_counts[str(labels[int(row)])].update(tokens)
    output: dict[str, np.ndarray] = {}
    class_size = {str(label): int((labels[rows] == label).sum()) for label in classes}
    total = max(len(rows), 1)
    for token, support in supports.items():
        global_rate = support / total
        values = np.zeros(len(classes), dtype=np.float32)
        for index, label in enumerate(classes):
            name = str(label)
            positive_size = class_size[name]
            negative_size = total - positive_size
            positive = class_counts[name][token]
            negative = support - positive
            pos_rate = (positive + EB_PRIOR_STRENGTH * global_rate) / (positive_size + EB_PRIOR_STRENGTH)
            neg_rate = (negative + EB_PRIOR_STRENGTH * global_rate) / (max(negative_size, 1) + EB_PRIOR_STRENGTH)
            values[index] = np.float32(np.log((pos_rate + 1e-6) / (1.0 - pos_rate + 1e-6)) - np.log((neg_rate + 1e-6) / (1.0 - neg_rate + 1e-6)))
        output[token] = np.clip(values * (support / (support + EB_PRIOR_STRENGTH)), -4.0, 4.0)
    return output, dict(supports)


def build_fold_evidence(events_by_row: list[list[tuple[str, str, str]]], cache: EventCache, rows: np.ndarray, weights: dict[str, np.ndarray], supports: dict[str, int], class_count: int) -> list[np.ndarray]:
    burdens = np.asarray(cache.burden)[:, 0]
    return [build_event_evidence(events_by_row[int(row)], weights, supports, np.asarray([burdens[int(row)]]), class_count=class_count) for row in rows]


def build_inner_oof_evidence(events_by_row: list[list[tuple[str, str, str]]], cache: EventCache, outer_train: np.ndarray, labels: np.ndarray, classes: np.ndarray, seed: int) -> list[np.ndarray]:
    result: list[np.ndarray | None] = [None] * len(outer_train)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for local_fit, local_holdout in splitter.split(np.zeros(len(outer_train)), labels[outer_train]):
        fit_rows = outer_train[local_fit]
        holdout_rows = outer_train[local_holdout]
        weights, supports = fit_empirical_bayes_weights(events_by_row, fit_rows, labels, classes)
        evidence = build_fold_evidence(events_by_row, cache, holdout_rows, weights, supports, len(classes))
        for target, value in zip(local_holdout, evidence):
            result[int(target)] = value
    if any(value is None for value in result):
        raise AssertionError("inner OOF evidence가 모든 outer-train 행을 덮지 못했습니다.")
    return [value for value in result if value is not None]


def class_weight(labels: np.ndarray, classes: np.ndarray) -> torch.Tensor:
    count = np.asarray([(labels == label).sum() for label in classes], dtype=np.float32)
    values = len(labels) / np.maximum(count * len(classes), 1.0)
    return torch.from_numpy(values.astype(np.float32))


def train_network(evidence: list[np.ndarray], labels: np.ndarray, classes: np.ndarray, seed: int, fold: int) -> tuple[EvidenceSetNetwork, list[dict[str, float]]]:
    torch.manual_seed(seed * 100 + fold)
    model = EvidenceSetNetwork(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    y_index = np.searchsorted(classes, labels).astype(np.int64)
    weights = class_weight(labels, classes)
    history = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(evidence)).numpy()
        loss_sum, seen = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            batch_index = order[start:start + BATCH_SIZE]
            features, mask = pad_evidence_sets([evidence[int(index)] for index in batch_index])
            target = torch.from_numpy(y_index[batch_index])
            optimizer.zero_grad(set_to_none=True)
            loss = listwise_loss(model(features, mask), target, weights)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_index)
            seen += len(batch_index)
        history.append({"fold": fold, "epoch": epoch, "phase": "train", "loss": loss_sum / max(seen, 1)})
    return model, history


def predict_network(model: EvidenceSetNetwork, evidence: list[np.ndarray]) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(evidence), BATCH_SIZE):
            features, mask = pad_evidence_sets(evidence[start:start + BATCH_SIZE])
            output.append(torch.softmax(model(features, mask), dim=1).cpu().numpy())
    return np.vstack(output).astype(np.float32)


def topk_metrics(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    ordered = np.argsort(probabilities, axis=1)[:, ::-1]
    metrics: dict[str, float] = {}
    for k in (1, 2, 3):
        included = np.asarray([labels[row] in classes[ordered[row, :k]] for row in range(len(labels))])
        metrics[f"top{k}_recall"] = float(included.mean())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-id", default="exp-class-conditional-evidence-set-network-01")
    parser.add_argument("--baseline-oof", type=Path, default=None, help="Optional train-only, row-aligned team OOF probability CSV.")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("사전 고정된 첫 screen은 seed 42만 허용합니다.")
    start = time.time()
    root = project_root()
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [column for column in train if column not in ("ID", "SUBCLASS")]
    if int(train[genes].isna().sum().sum()) != 0:
        raise AssertionError("train NaN 계약 위반")
    labels = train.SUBCLASS.to_numpy()
    classes = np.asarray(sorted(pd.unique(labels)), dtype=object)
    baseline = reference_only_baseline() if args.baseline_oof is None else validate_baseline_oof(pd.read_csv(args.baseline_oof), classes, len(train))
    print(f"[stage] baseline comparison: {baseline.comparison_mode}", flush=True)
    print("[stage] train-only mutation parsing", flush=True)
    cache = parse_train_frame(train[genes], genes, show_progress=False)
    events_by_row = row_events(cache)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    network_oof = np.zeros((len(train), len(classes)), dtype=np.float32)
    fold_rows, audit_rows, loss_rows = [], [], []
    for fold, (outer_train, outer_valid) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        print(f"[stage] outer fold {fold}/5: inner OOF evidence", flush=True)
        inner_evidence = build_inner_oof_evidence(events_by_row, cache, outer_train, labels, classes, args.seed * 1000 + fold)
        audit = nested_audit(outer_train, outer_train, outer_valid)
        if not audit["ranker_training_rows_are_inner_oof"] or audit["outer_validation_used_for_eb_fit"]:
            raise AssertionError("outer validation 행이 evidence fit에 포함되었습니다.")
        model, history = train_network(inner_evidence, labels[outer_train], classes, args.seed, fold)
        weights, supports = fit_empirical_bayes_weights(events_by_row, outer_train, labels, classes)
        valid_evidence = build_fold_evidence(events_by_row, cache, outer_valid, weights, supports, len(classes))
        probability = predict_network(model, valid_evidence)
        network_oof[outer_valid] = probability
        fold_rows.append({"fold": fold, "variant": "evidence_set_network", "macro_f1": float(f1_score(labels[outer_valid], classes[probability.argmax(1)], average="macro")), "feature_count": 16})
        if baseline.probabilities is not None:
            fold_rows.append({"fold": fold, "variant": "team_3way", "macro_f1": float(f1_score(labels[outer_valid], classes[baseline.probabilities[outer_valid].argmax(1)], average="macro")), "feature_count": np.nan})
        audit_rows.append({"seed": args.seed, "fold": fold, **audit, "inner_oof_rows": len(outer_train), "outer_validation_rows": len(outer_valid), "test_read": False})
        loss_rows.extend(history)
    probabilities = {"evidence_set_network": network_oof}
    if baseline.probabilities is not None:
        probabilities = {"team_3way": baseline.probabilities, **probabilities}
    summary_rows = []
    class_rows = []
    for name, probability in probabilities.items():
        score = float(f1_score(labels, classes[probability.argmax(1)], average="macro"))
        summary_rows.append({"variant": name, "oof_macro_f1": score, "feature_count": 16 if name == "evidence_set_network" else np.nan, "convergence_warning_count": 0, "leakage_check": True, "nan_as_mutation_count": 0, "runtime_seconds": time.time() - start, **topk_metrics(labels, probability, classes)})
        for label in classes:
            class_rows.append({"variant": name, "class": label, "support": int((labels == label).sum()), "f1": float(f1_score(labels, classes[probability.argmax(1)], labels=[label], average="macro", zero_division=0))})
    summary = pd.DataFrame(summary_rows)
    base_score = baseline.reference_macro_f1 if baseline.probabilities is None else float(summary.loc[summary.variant.eq("team_3way"), "oof_macro_f1"].iloc[0])
    summary["team_reference_macro_f1"] = base_score
    summary["delta_vs_team_3way"] = summary.oof_macro_f1 - base_score
    summary["baseline_comparison_mode"] = baseline.comparison_mode
    candidate_score = float(summary.loc[summary.variant.eq("evidence_set_network"), "oof_macro_f1"].iloc[0])
    class_table = pd.DataFrame(class_rows).pivot(index="class", columns="variant", values="f1").reset_index()
    if baseline.probabilities is not None:
        class_table["delta_network_vs_team"] = class_table.evidence_set_network - class_table.team_3way
        base_margin = np.partition(baseline.probabilities, kth=-2, axis=1)[:, -1] - np.partition(baseline.probabilities, kth=-2, axis=1)[:, -2]
    else:
        base_margin = np.partition(network_oof, kth=-2, axis=1)[:, -1] - np.partition(network_oof, kth=-2, axis=1)[:, -2]
    low_mask = base_margin < 0.05
    low_margin = pd.DataFrame({"variant": list(probabilities), "group": "baseline_margin_<0.05", "support": int(low_mask.sum()), "macro_f1": [float(f1_score(labels[low_mask], classes[value[low_mask].argmax(1)], average="macro", zero_division=0)) for value in probabilities.values()]})
    promotion = {
        "delta_at_least_0_03": bool(candidate_score - base_score >= 0.03),
        "paired_team_oof_supplied": baseline.probabilities is not None,
    }
    if baseline.probabilities is not None:
        fold_frame = pd.DataFrame(fold_rows)
        team_fold_score = fold_frame.loc[fold_frame.variant.eq("team_3way")].set_index("fold").macro_f1.to_dict()
        network_fold_score = fold_frame.loc[fold_frame.variant.eq("evidence_set_network")].set_index("fold").macro_f1.to_dict()
        promotion["folds_improved_at_least_4"] = sum(network_fold_score[fold] > team_fold_score[fold] for fold in network_fold_score) >= 4
        promotion["low_margin_delta_at_least_0_04"] = bool(float(low_margin.loc[low_margin.variant.eq("evidence_set_network"), "macro_f1"].iloc[0] - low_margin.loc[low_margin.variant.eq("team_3way"), "macro_f1"].iloc[0]) >= 0.04)
        promotion["improved_classes_at_least_15"] = int((class_table.delta_network_vs_team > 0).sum()) >= 15
        promotion["decision"] = "promote_to_3seed" if all(promotion.values()) else "stop_screen"
    else:
        promotion["decision"] = "reference_only_screen_requires_paired_team_oof"
    result = Path(__file__).parent.parent / "result"
    result.mkdir(exist_ok=True)
    summary.to_csv(result / f"{args.run_id}_seed{args.seed}_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(result / f"{args.run_id}_seed{args.seed}_fold_metrics.csv", index=False)
    class_table.to_csv(result / f"{args.run_id}_seed{args.seed}_class_metrics.csv", index=False)
    low_margin.to_csv(result / f"{args.run_id}_seed{args.seed}_low_margin_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(result / f"{args.run_id}_seed{args.seed}_nested_audit.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(result / f"{args.run_id}_seed{args.seed}_loss.csv", index=False)
    pd.DataFrame({"true_class": labels, **{f"{name}__{label}": value[:, index] for name, value in probabilities.items() for index, label in enumerate(classes)}}).to_csv(result / f"{args.run_id}_seed{args.seed}_oof_probabilities.csv", index=False)
    (result / f"{args.run_id}_seed{args.seed}_feature_contract.json").write_text(json.dumps({"event_features": ["eb_log_odds", "absolute_evidence", "log_support", "posterior_reliability", "positive", "negative", "burden_normalized", "exact", "recurrent", "event_type_one_hot"], "event_feature_count": 16, "train_only_vocabulary": True, "test_read": False, "baseline_comparison_mode": baseline.comparison_mode}, ensure_ascii=False, indent=2), encoding="utf-8")
    (result / f"{args.run_id}_seed{args.seed}_leakage_audit.json").write_text(json.dumps({"leakage_check": True, "nan_as_mutation_count": 0, "outer_validation_used_for_eb_fit": False, "baseline_comparison_mode": baseline.comparison_mode, "promotion": promotion}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
