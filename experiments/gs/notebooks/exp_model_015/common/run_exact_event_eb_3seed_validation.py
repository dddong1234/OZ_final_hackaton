"""Fixed 3-seed validation for the already screened exact-event EB candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_exact_event_eb_screen import RESULT_DIR, run

SEEDS = (42, 777, 2024)


def aggregate(seed_summaries: pd.DataFrame, folds: pd.DataFrame, classes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    baseline = seed_summaries.loc[seed_summaries.variant.eq("H0_selective_EB"), ["seed", "oof_macro_f1"]].rename(columns={"oof_macro_f1": "h0_macro_f1"})
    candidate = seed_summaries.loc[seed_summaries.variant.eq("exact_event_EB")].merge(baseline, on="seed", validate="one_to_one")
    candidate["paired_delta"] = candidate.oof_macro_f1 - candidate.h0_macro_f1
    aggregate_table = seed_summaries.groupby("variant", as_index=False).agg(
        seed_count=("seed", "nunique"),
        oof_macro_f1_mean=("oof_macro_f1", "mean"),
        oof_macro_f1_std=("oof_macro_f1", "std"),
        oof_accuracy_mean=("oof_accuracy", "mean"),
        feature_count_mean=("feature_count_mean", "mean"),
        convergence_warning_count=("convergence_warning_count", "sum"),
        leakage_check=("leakage_check", "all"),
        nan_as_mutation_count=("nan_as_mutation_count", "max"),
    )
    aggregate_table = aggregate_table.merge(
        candidate.paired_delta.agg(delta_vs_h0_mean="mean", delta_vs_h0_std="std", delta_vs_h0_min="min").to_frame().T,
        how="cross",
    )
    pivot = folds.pivot(index=["seed", "fold"], columns="variant", values="macro_f1")
    class_pivot = classes.pivot(index=["seed", "class"], columns="variant", values="f1")
    class_delta = class_pivot["exact_event_EB"] - class_pivot["H0_selective_EB"]
    decision = {
        "seeds": list(SEEDS),
        "new_feature_or_parameter_search": False,
        "all_seed_delta_positive": bool((candidate.paired_delta > 0).all()),
        "mean_delta": float(candidate.paired_delta.mean()),
        "minimum_delta": float(candidate.paired_delta.min()),
        "positive_fold_count": int((pivot["exact_event_EB"] > pivot["H0_selective_EB"]).sum()),
        "minimum_class_mean_delta": float(class_delta.groupby("class").mean().min()),
        "class_collapse_absent": bool(class_delta.groupby("class").mean().min() >= -0.05),
        "leakage_check": bool(seed_summaries.leakage_check.all()),
        "nan_as_mutation_count": int(seed_summaries.nan_as_mutation_count.max()),
    }
    decision["accepted_3seed"] = bool(
        decision["all_seed_delta_positive"]
        and decision["mean_delta"] >= 0.010
        and decision["minimum_delta"] >= 0.005
        and decision["positive_fold_count"] >= 11
        and decision["class_collapse_absent"]
        and decision["leakage_check"]
        and decision["nan_as_mutation_count"] == 0
    )
    return aggregate_table, decision


def run_validation(run_id: str, reuse_seed42: bool) -> None:
    for seed in SEEDS:
        summary_path = RESULT_DIR / f"{run_id}_seed{seed}_summary.csv"
        if reuse_seed42 and seed == 42 and summary_path.exists():
            print(f"[exact-event EB 3seed] reusing validated seed42 result: {summary_path.name}", flush=True)
            continue
        print(f"[exact-event EB 3seed] running seed {seed}", flush=True)
        run(run_id, seed)
    summaries = pd.concat([pd.read_csv(RESULT_DIR / f"{run_id}_seed{seed}_summary.csv") for seed in SEEDS], ignore_index=True)
    folds = pd.concat([pd.read_csv(RESULT_DIR / f"{run_id}_seed{seed}_fold_metrics.csv") for seed in SEEDS], ignore_index=True)
    classes = pd.concat([pd.read_csv(RESULT_DIR / f"{run_id}_seed{seed}_class_metrics.csv") for seed in SEEDS], ignore_index=True)
    aggregate_table, decision = aggregate(summaries, folds, classes)
    summaries.to_csv(RESULT_DIR / f"{run_id}_3seed_summary.csv", index=False)
    aggregate_table.to_csv(RESULT_DIR / f"{run_id}_3seed_aggregate.csv", index=False)
    folds.to_csv(RESULT_DIR / f"{run_id}_3seed_fold_metrics.csv", index=False)
    classes.to_csv(RESULT_DIR / f"{run_id}_3seed_class_metrics.csv", index=False)
    (RESULT_DIR / f"{run_id}_3seed_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


def smoke() -> None:
    rows = []
    for seed, h0, candidate in ((42, .50, .52), (777, .49, .51), (2024, .51, .53)):
        for variant, value in (("H0_selective_EB", h0), ("exact_event_EB", candidate)):
            rows.append({"seed": seed, "variant": variant, "oof_macro_f1": value, "oof_accuracy": value, "feature_count_mean": 10, "convergence_warning_count": 0, "leakage_check": True, "nan_as_mutation_count": 0})
    summary = pd.DataFrame(rows)
    folds = pd.DataFrame([{"seed": seed, "fold": fold, "variant": variant, "macro_f1": score} for seed, base in ((42, .50), (777, .49), (2024, .51)) for fold in range(1, 6) for variant, score in (("H0_selective_EB", base), ("exact_event_EB", base + .02))])
    classes = pd.DataFrame([{"seed": seed, "class": label, "variant": variant, "f1": value} for seed in SEEDS for label in ("A", "B") for variant, value in (("H0_selective_EB", .5), ("exact_event_EB", .52))])
    _, decision = aggregate(summary, folds, classes)
    assert decision["accepted_3seed"]
    print(json.dumps({"smoke": "ok", "test_read": False, "parameter_search": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-exact-event-eb-01")
    parser.add_argument("--no-reuse-seed42", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else run_validation(args.run_id, reuse_seed42=not args.no_reuse_seed42)
