"""환자×암종 후보 행을 만들고 후보 점수를 26-class 확률로 복원한다."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(probability / (1.0 - probability))


def candidate_matrix(
    p_non_eb: np.ndarray,
    p_eb: np.ndarray,
    eb_evidence: np.ndarray,
    burden: np.ndarray,
    class_count: int,
    rare_eb_evidence: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """모든 환자에 대해 class_count개의 후보 행을 만든다.

    ``eb_evidence``는 해당 fit partition의 train으로 계산된 클래스별 EB
    evidence여야 한다. 이 함수는 통계량을 fit하지 않는다.
    """
    p_non_eb = np.asarray(p_non_eb, dtype=np.float64)
    p_eb = np.asarray(p_eb, dtype=np.float64)
    eb_evidence = np.asarray(eb_evidence, dtype=np.float64)
    burden = np.asarray(burden, dtype=np.float64).reshape(-1)
    expected = (len(burden), class_count)
    if p_non_eb.shape != expected or p_eb.shape != expected or eb_evidence.shape != expected:
        raise ValueError("확률과 EB evidence는 (n_samples, class_count)여야 합니다.")
    if rare_eb_evidence is None:
        rare_eb_evidence = np.zeros_like(eb_evidence)
    rare_eb_evidence = np.asarray(rare_eb_evidence, dtype=np.float64)
    if rare_eb_evidence.shape != expected:
        raise ValueError("rare EB evidence는 (n_samples, class_count)여야 합니다.")
    if not np.isfinite(p_non_eb).all() or not np.isfinite(p_eb).all() or not np.isfinite(eb_evidence).all():
        raise ValueError("candidate 입력에는 NaN/inf가 있을 수 없습니다.")

    n_samples = len(burden)
    patient_index = np.repeat(np.arange(n_samples), class_count)
    candidate_index = np.tile(np.arange(class_count), n_samples)
    candidate_one_hot = np.eye(class_count, dtype=np.float64)[candidate_index]

    # (n_samples, class_count): candidate를 제외한 최대 EB probability
    best_competitor = np.empty_like(p_eb)
    for candidate in range(class_count):
        values = p_eb.copy()
        values[:, candidate] = -np.inf
        best_competitor[:, candidate] = values.max(axis=1)

    positive_evidence = np.maximum(eb_evidence, 0.0)
    negative_evidence = np.minimum(eb_evidence, 0.0)
    row_positive_top3 = np.sort(positive_evidence, axis=1)[:, -min(3, class_count):].sum(axis=1)
    row_positive_max = positive_evidence.max(axis=1)
    row_negative_sum = negative_evidence.sum(axis=1)
    row_negative_min = negative_evidence.min(axis=1)
    positive_count = (eb_evidence > 0.0).sum(axis=1)
    negative_count = (eb_evidence < 0.0).sum(axis=1)
    p0_top = p_non_eb.argmax(axis=1)
    p1_top = p_eb.argmax(axis=1)
    log_burden = np.log1p(np.maximum(burden, 0.0))

    def flat(values: np.ndarray) -> np.ndarray:
        return values.reshape(-1, 1)

    features = np.hstack([
        flat(p_non_eb),
        flat(p_eb),
        flat(_logit(p_non_eb)),
        flat(_logit(p_eb)),
        flat(p_eb - best_competitor),
        flat(eb_evidence),
        flat(eb_evidence / np.maximum(log_burden[:, None], 1.0)),
        flat(rare_eb_evidence),
        flat(eb_evidence - rare_eb_evidence),
        np.repeat(row_positive_max, class_count)[:, None],
        np.repeat(row_positive_top3, class_count)[:, None],
        np.repeat(row_negative_sum, class_count)[:, None],
        np.repeat(row_negative_min, class_count)[:, None],
        np.repeat(positive_count, class_count)[:, None],
        np.repeat(negative_count, class_count)[:, None],
        np.repeat(log_burden, class_count)[:, None],
        (candidate_index == np.repeat(p0_top, class_count))[:, None].astype(np.float64),
        (candidate_index == np.repeat(p1_top, class_count))[:, None].astype(np.float64),
        np.repeat(p0_top == p1_top, class_count)[:, None].astype(np.float64),
        candidate_one_hot,
    ])
    if not np.isfinite(features).all():
        raise ValueError("candidate feature 생성 후 NaN/inf가 남았습니다.")
    return features.astype(np.float32), patient_index, candidate_index


def candidate_scores_to_probability(
    positive_score: np.ndarray, n_samples: int, n_classes: int
) -> np.ndarray:
    """후보별 logit을 환자별 softmax 확률로 바꾼다."""
    logits = np.asarray(positive_score, dtype=np.float64)
    if logits.size != n_samples * n_classes:
        raise ValueError("후보 점수 수가 n_samples × n_classes와 일치하지 않습니다.")
    logits = logits.reshape(n_samples, n_classes)
    logits = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("후보 확률 softmax 정규화에 실패했습니다.")
    return probability.astype(np.float32)


def build_ranker_audit(
    outer_train: np.ndarray, inner_prediction_rows: np.ndarray, outer_valid: np.ndarray
) -> dict[str, bool]:
    """ranker 학습용 후보 피처가 inner OOF 행에서만 왔는지 확인한다."""
    outer_train_set = set(np.asarray(outer_train, dtype=int).tolist())
    inner_set = set(np.asarray(inner_prediction_rows, dtype=int).tolist())
    valid_set = set(np.asarray(outer_valid, dtype=int).tolist())
    return {
        "ranker_training_rows_are_inner_oof": inner_set == outer_train_set,
        "outer_validation_used_for_ranker_fit": bool(inner_set & valid_set),
    }


def topk_metrics(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    """후보군 포함률과 oracle Macro F1을 train OOF에서만 계산한다."""
    y = np.asarray(y)
    probability = np.asarray(probability)
    classes = np.asarray(classes)
    if probability.shape != (len(y), len(classes)):
        raise ValueError("probability shape가 y/classes와 일치하지 않습니다.")
    order = np.argsort(probability, axis=1)[:, ::-1]
    true_index = np.searchsorted(classes, y)
    metrics: dict[str, float] = {}
    for k in (1, 2, 3):
        included = (order[:, :k] == true_index[:, None]).any(axis=1)
        oracle_index = np.where(included, true_index, order[:, 0])
        metrics[f"top{k}_recall"] = float(included.mean())
        metrics[f"oracle_macro_f1_at{k}"] = float(f1_score(y, classes[oracle_index], average="macro"))
    return metrics
