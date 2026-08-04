"""고정 Empirical-Bayes 후보의 3-seed 확정 검증 집계 도구."""
from __future__ import annotations

import pandas as pd


P1_VARIANT = "P1 multinomial LR"
EB_VARIANT = "eb"


def summarize_three_seed(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """seed별 P1/EB 결과를 paired delta 중심으로 요약한다.

    이 함수는 이미 실행된 OOF 결과만 집계하며, feature selection·통계 학습·test 접근을 하지 않는다.
    """
    required = {
        "seed", "variant", "oof_macro_f1", "feature_count", "convergence_warning_count",
        "leakage_check", "nan_as_mutation_count",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"필수 결과 컬럼이 없습니다: {sorted(missing)}")

    subset = rows.loc[rows.variant.isin([P1_VARIANT, EB_VARIANT])].copy()
    counts = subset.groupby("seed").variant.nunique()
    if not counts.eq(2).all():
        raise ValueError("각 seed에는 P1과 eb 결과가 정확히 하나씩 필요합니다.")

    pivot = subset.pivot(index="seed", columns="variant", values="oof_macro_f1")
    paired = pivot.reset_index()
    paired["paired_delta_vs_p1"] = paired[EB_VARIANT] - paired[P1_VARIANT]
    per_seed = (
        subset.merge(paired[["seed", "paired_delta_vs_p1"]], on="seed", how="left")
        .sort_values(["seed", "variant"])
        .reset_index(drop=True)
    )

    aggregate = (
        per_seed.groupby("variant", as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            oof_macro_f1_mean=("oof_macro_f1", "mean"),
            oof_macro_f1_std=("oof_macro_f1", "std"),
            feature_count_mean=("feature_count", "mean"),
            convergence_warning_count_sum=("convergence_warning_count", "sum"),
            leakage_check_all=("leakage_check", "all"),
            nan_as_mutation_count_max=("nan_as_mutation_count", "max"),
            paired_delta_mean=("paired_delta_vs_p1", "mean"),
            paired_delta_std=("paired_delta_vs_p1", "std"),
            paired_delta_all_positive=("paired_delta_vs_p1", lambda x: bool((x > 0).all())),
        )
        .sort_values("variant")
        .reset_index(drop=True)
    )
    return per_seed, aggregate
