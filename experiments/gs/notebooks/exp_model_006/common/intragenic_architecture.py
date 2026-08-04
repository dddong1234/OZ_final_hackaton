"""Train-only gene-local multi-event architecture Empirical-Bayes features."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from h0_faithful_pipeline import SUB_RE, TRUNCATING, classify_event, normalise_cell


PRIOR_STRENGTH = 20.0
SUPPORT_SHRINKAGE = 20.0


def architecture_token_sets(frame: pd.DataFrame, genes: list[str]) -> list[set[str]]:
    """Return row-local, gene-specific architecture tokens without labels."""
    output = [set() for _ in range(len(frame))]
    for gene in genes:
        for row, value in enumerate(frame[gene].array):
            events = normalise_cell(value)
            if len(events) < 2:
                continue
            event_types = [classify_event(event) for event in events]
            types = set(event_types)
            positions = []
            for event in events:
                match = SUB_RE.fullmatch(event)
                if match:
                    positions.append(int(match.group(2)))
            tokens = output[row]
            tokens.add(f"{gene}__EVENT_COUNT_2PLUS")
            if len(events) >= 3:
                tokens.add(f"{gene}__EVENT_COUNT_3PLUS")
            if event_types.count("MISSENSE") >= 2:
                tokens.add(f"{gene}__MULTI_MISSENSE")
            if "MISSENSE" in types and bool(types & TRUNCATING):
                tokens.add(f"{gene}__MISSENSE_PLUS_TRUNCATING")
            if len(types) >= 2:
                tokens.add(f"{gene}__MULTI_FUNCTIONAL_TYPE")
            if len(positions) >= 2 and len(set(positions)) < len(positions):
                tokens.add(f"{gene}__SAME_POSITION_MULTI_EVENT")
    return output


def _fit_weights(token_sets: list[set[str]], labels: np.ndarray, classes: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    vocabulary = sorted(set().union(*token_sets)) if token_sets else []
    if not vocabulary:
        return {}, 0
    class_count = len(classes)
    class_index = {str(label): index for index, label in enumerate(classes)}
    support = {token: 0 for token in vocabulary}
    positive = {token: np.zeros(class_count, dtype=np.float64) for token in vocabulary}
    for tokens, label in zip(token_sets, labels):
        index = class_index[str(label)]
        for token in tokens:
            support[token] += 1
            positive[token][index] += 1.0
    class_support = np.array([(labels == label).sum() for label in classes], dtype=np.float64)
    total = float(len(labels))
    weights: dict[str, np.ndarray] = {}
    for token in vocabulary:
        count = float(support[token])
        global_rate = (count + 1.0) / (total + 2.0)
        pos_rate = (positive[token] + PRIOR_STRENGTH * global_rate) / (class_support + PRIOR_STRENGTH)
        neg_count = count - positive[token]
        neg_support = total - class_support
        neg_rate = (neg_count + PRIOR_STRENGTH * global_rate) / (neg_support + PRIOR_STRENGTH)
        raw = np.log(np.clip(pos_rate, 1e-6, 1 - 1e-6) / np.clip(1 - pos_rate, 1e-6, 1)) - np.log(np.clip(neg_rate, 1e-6, 1 - 1e-6) / np.clip(1 - neg_rate, 1e-6, 1))
        weights[token] = (raw * (count / (count + SUPPORT_SHRINKAGE))).astype(np.float32)
    return weights, len(vocabulary)


def _apply_weights(token_sets: list[set[str]], weights: dict[str, np.ndarray], class_count: int) -> np.ndarray:
    output = np.zeros((len(token_sets), class_count), dtype=np.float32)
    for row, tokens in enumerate(token_sets):
        active = [weights[token] for token in tokens if token in weights]
        if active:
            output[row] = np.sum(active, axis=0) / math.sqrt(len(active))
    return output


def cross_fitted_architecture_scores(
    token_sets: list[set[str]], labels: np.ndarray, classes: np.ndarray, fit_index: np.ndarray, apply_index: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Fit every supervised architecture statistic inside the supplied fit rows."""
    fit_index = np.asarray(fit_index, dtype=np.int64)
    apply_index = np.asarray(apply_index, dtype=np.int64)
    y_fit = labels[fit_index]
    fit_sets = [token_sets[index] for index in fit_index]
    scores = np.zeros((len(fit_index), len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for inner_fit, inner_valid in splitter.split(np.zeros(len(y_fit)), y_fit):
        weights, _ = _fit_weights([fit_sets[index] for index in inner_fit], y_fit[inner_fit], classes)
        scores[inner_valid] = _apply_weights([fit_sets[index] for index in inner_valid], weights, len(classes))
    weights, vocabulary_size = _fit_weights(fit_sets, y_fit, classes)
    apply_scores = _apply_weights([token_sets[index] for index in apply_index], weights, len(classes))
    keep = scores.min(axis=0) != scores.max(axis=0)
    scores, apply_scores = scores[:, keep], apply_scores[:, keep]
    mean, std = scores.mean(axis=0), scores.std(axis=0)
    std[std < 1e-6] = 1.0
    names = [f"IA_EB__{label}" for label, enabled in zip(classes, keep) if enabled]
    return ((scores - mean) / std).astype(np.float32), ((apply_scores - mean) / std).astype(np.float32), names, vocabulary_size
