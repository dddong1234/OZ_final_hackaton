"""Run exp_006 with the unchanged shared preprocessing benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.preprocessing_benchmark import run_preprocessing_benchmark
from experiments.SDH.exp_006_burden3_ablation.preprocessing import (
    make_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run(
    train_path: Path = DEFAULT_TRAIN_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    confirmation: bool = False,
) -> pd.DataFrame:
    """Compare burden2 and burden3 without changing model parameters."""

    train = pd.read_csv(train_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_confirmation" if confirmation else ""
    summaries = []

    for case_name, preprocessor in make_candidates().items():
        print(f"\n===== {case_name} =====")
        result = run_preprocessing_benchmark(
            train,
            preprocessor,
            experiment_id=f"exp_006_{case_name}",
            preprocessing_name=case_name,
            model="logistic",
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
    parser.add_argument("--confirmation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    table = run(
        train_path=arguments.train,
        results_dir=arguments.results_dir,
        confirmation=arguments.confirmation,
    )
    print(
        table[
            [
                "preprocessing",
                "oof_f1_macro_mean",
                "oof_accuracy_mean",
                "elapsed_seconds",
            ]
        ].to_string(index=False)
    )
