"""Run exp_004 combinations with the shared preprocessing benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.preprocessing_benchmark import (
    BenchmarkModel,
    run_preprocessing_benchmark,
)
from experiments.SDH.exp_004_feature_combinations.preprocessing import (
    make_combination_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run(
    train_path: Path = DEFAULT_TRAIN_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    selected_cases: list[str] | None = None,
    confirmation: bool = False,
    model: BenchmarkModel = "logistic",
) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    candidates = make_combination_candidates()
    case_names = selected_cases or list(candidates)
    unknown = sorted(set(case_names) - set(candidates))
    if unknown:
        raise ValueError(f"알 수 없는 case: {unknown}")

    suffix_parts = []
    if model != "logistic":
        suffix_parts.append(model)
    if confirmation:
        suffix_parts.append("confirmation")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

    results_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for case_name in case_names:
        print(f"\n===== {case_name} =====")
        result = run_preprocessing_benchmark(
            train,
            candidates[case_name],
            experiment_id=f"exp_004_{case_name}",
            preprocessing_name=case_name,
            model=model,
            confirmation=confirmation,
        )
        result.save_metrics(
            results_dir / f"metrics_{case_name}{suffix}.json"
        )
        summaries.append(result.summary)

    leaderboard = (
        pd.DataFrame(summaries)
        .sort_values("oof_f1_macro_mean", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.to_csv(
        results_dir / f"leaderboard{suffix}.csv",
        index=False,
    )
    return leaderboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument(
        "--model",
        choices=("logistic", "lightgbm"),
        default="logistic",
    )
    parser.add_argument("--confirmation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    leaderboard = run(
        train_path=arguments.train,
        results_dir=arguments.results_dir,
        selected_cases=arguments.cases,
        model=arguments.model,
        confirmation=arguments.confirmation,
    )
    print(
        leaderboard[
            [
                "preprocessing",
                "oof_f1_macro_mean",
                "oof_accuracy_mean",
                "elapsed_seconds",
            ]
        ].to_string(index=False)
    )
