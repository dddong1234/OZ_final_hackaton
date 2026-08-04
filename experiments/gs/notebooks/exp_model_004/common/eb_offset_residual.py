"""P1+EB 확률을 고정 offset으로 보정하는 희소 선형 softmax residual."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.sparse import csr_matrix

HASH_DIMENSION = 16_384


def hashed_event_matrix(
    token_sets: list[set[str]], rows: np.ndarray, dimension: int = HASH_DIMENSION
) -> csr_matrix:
    """학습 없이 고정 blake2b hash로 gene×event-type binary matrix를 만든다."""
    rows = np.asarray(rows, dtype=int)
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    for output_row, source_row in enumerate(rows):
        for token in token_sets[int(source_row)]:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            column = int.from_bytes(digest, byteorder="little", signed=False) % dimension
            matrix_rows.append(output_row)
            matrix_columns.append(column)
    data = np.ones(len(matrix_rows), dtype=np.float32)
    return csr_matrix((data, (matrix_rows, matrix_columns)), shape=(len(rows), dimension), dtype=np.float32)


def offset_probability(
    offset_log_probability: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    features: csr_matrix,
) -> np.ndarray:
    """고정 offset과 희소 residual을 합쳐 row-wise softmax 확률을 반환한다."""
    offset_log_probability = np.asarray(offset_log_probability, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if offset_log_probability.ndim != 2 or weight.shape != (features.shape[1], offset_log_probability.shape[1]):
        raise ValueError("offset/weight/features shape가 일치하지 않습니다.")
    if bias.shape != (offset_log_probability.shape[1],):
        raise ValueError("bias class dimension이 일치하지 않습니다.")
    logits = offset_log_probability + np.asarray(features @ weight) + bias
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("residual softmax 확률 계약 위반")
    return probability.astype(np.float32)


def fit_offset_residual(
    features: csr_matrix,
    y_index: np.ndarray,
    offset_log_probability: np.ndarray,
    class_weight: np.ndarray,
    *,
    epochs: int = 40,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """class-balanced cross-entropy로 zero-start linear residual을 학습한다."""
    y_index = np.asarray(y_index, dtype=int)
    offset_log_probability = np.asarray(offset_log_probability, dtype=np.float64)
    class_weight = np.asarray(class_weight, dtype=np.float64)
    n_rows, n_features = features.shape
    n_classes = offset_log_probability.shape[1]
    if len(y_index) != n_rows or offset_log_probability.shape[0] != n_rows:
        raise ValueError("features/y/offset row count가 일치하지 않습니다.")
    if class_weight.shape != (n_classes,) or y_index.min() < 0 or y_index.max() >= n_classes:
        raise ValueError("class weight 또는 label index 계약 위반")

    rng = np.random.default_rng(seed)
    weight = np.zeros((n_features, n_classes), dtype=np.float64)
    bias = np.zeros(n_classes, dtype=np.float64)
    one_hot = np.eye(n_classes, dtype=np.float64)[y_index]
    sample_weight = class_weight[y_index]
    history: list[float] = []
    for _ in range(epochs):
        for batch in (rng.permutation(n_rows)[start : start + batch_size] for start in range(0, n_rows, batch_size)):
            probability = offset_probability(offset_log_probability[batch], weight, bias, features[batch]).astype(np.float64)
            scale = sample_weight[batch, None]
            difference = (probability - one_hot[batch]) * scale
            normalizer = float(scale.sum())
            gradient_weight = np.asarray(features[batch].T @ difference) / normalizer + l2 * weight
            gradient_bias = difference.sum(axis=0) / normalizer
            weight -= learning_rate * gradient_weight
            bias -= learning_rate * gradient_bias
        whole_probability = offset_probability(offset_log_probability, weight, bias, features).astype(np.float64)
        nll = -np.log(np.clip(whole_probability[np.arange(n_rows), y_index], 1e-12, 1.0))
        history.append(float(np.average(nll, weights=sample_weight) + 0.5 * l2 * np.square(weight).sum()))
    return weight.astype(np.float32), bias.astype(np.float32), history


def offset_audit(
    outer_train: np.ndarray, inner_offset_rows: np.ndarray, outer_valid: np.ndarray
) -> dict[str, bool]:
    outer_train_set = set(np.asarray(outer_train, dtype=int).tolist())
    inner_set = set(np.asarray(inner_offset_rows, dtype=int).tolist())
    outer_valid_set = set(np.asarray(outer_valid, dtype=int).tolist())
    return {
        "offset_train_rows_are_inner_oof": inner_set == outer_train_set,
        "outer_validation_used_for_residual_fit": bool(inner_set & outer_valid_set),
    }
