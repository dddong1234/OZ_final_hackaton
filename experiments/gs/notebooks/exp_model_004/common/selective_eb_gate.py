"""고정 선택 규칙으로 P1 non-EB와 P1+EB 확률을 결합한다.

이 모듈은 학습하거나 threshold를 탐색하지 않는다. ``SELECTIVE_MARGIN``은
기존 42/777/2024 취약구간 감사 후 고정됐으며, 이 실험에서는 새 seed에만
적용해 검증한다.
"""
from __future__ import annotations

import numpy as np

SELECTIVE_MARGIN = 0.05
VALIDATION_SEEDS = (31415, 52, 62)


def selective_probability(
    p1_non_eb: np.ndarray, p1_empirical_bayes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """EB margin이 0.05 미만인 행만 P1 non-EB 확률로 되돌린다.

    Returns
    -------
    probability:
        행별로 선택된 26-class 확률.
    use_non_eb:
        P1 non-EB를 선택한 행을 나타내는 boolean mask.
    """
    p1_non_eb = np.asarray(p1_non_eb, dtype=np.float64)
    p1_empirical_bayes = np.asarray(p1_empirical_bayes, dtype=np.float64)
    if p1_non_eb.ndim != 2 or p1_non_eb.shape != p1_empirical_bayes.shape:
        raise ValueError("두 확률 행렬은 동일한 (n_samples, n_classes) 형태여야 합니다.")
    if p1_non_eb.shape[1] < 2:
        raise ValueError("선택 gate는 최소 두 클래스 확률이 필요합니다.")

    top_two = np.partition(p1_empirical_bayes, kth=-2, axis=1)[:, -2:]
    eb_margin = top_two[:, 1] - top_two[:, 0]
    use_non_eb = eb_margin < SELECTIVE_MARGIN
    selected = np.where(use_non_eb[:, None], p1_non_eb, p1_empirical_bayes)
    if not np.allclose(selected.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("입력 확률은 각 행에서 1로 정규화되어야 합니다.")
    return selected, use_non_eb
