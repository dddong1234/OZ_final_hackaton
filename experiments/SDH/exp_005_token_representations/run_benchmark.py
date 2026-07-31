"""Run exp_005 token representations with the fixed shared benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.preprocessing_benchmark import run_preprocessing_benchmark
from experiments.SDH.exp_005_token_representations.preprocessing import (
    make_token_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run(
    train_path: Path = DEFAULT_TRAIN_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    selected_cases: list[str] | None = None,
    confirmation: bool = False,
) -> pd.DataFrame:
    """Run preprocessing candidates without changing benchmark model settings."""

    train = pd.read_csv(train_path)
    candidates = make_token_candidates()
    case_names = selected_cases or list(candidates)
    unknown = sorted(set(case_names) - set(candidates))
    if unknown:
        raise ValueError(f"알 수 없는 case: {unknown}")

    suffix = "_confirmation" if confirmation else ""
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for case_name in case_names:
        print(f"\n===== {case_name} =====")
        result = run_preprocessing_benchmark(
            train,
            candidates[case_name],
            experiment_id=f"exp_005_{case_name}",
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
