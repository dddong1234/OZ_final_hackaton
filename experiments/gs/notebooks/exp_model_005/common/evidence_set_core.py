"""Candidate-wise evidence tensors and a fixed shared listwise network."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EVENT_TYPES = ("MISSENSE", "SYNONYMOUS", "NONSENSE", "FRAMESHIFT", "SPLICE", "INFRAME_INDEL", "OTHER")
EVENT_TYPE_INDEX = {event_type: index for index, event_type in enumerate(EVENT_TYPES)}
EVIDENCE_DIM = 16


def build_event_evidence(
    events: list[tuple[str, str, str]],
    weights: dict[str, np.ndarray],
    support: dict[str, int],
    burden: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    """Return [candidate, event, feature] train-fitted EB evidence.

    ``weights`` and ``support`` must originate from the fitting partition. A
    missing token retains an all-zero EB contribution but still preserves its
    canonical event type and row-local burden information.
    """
    output = np.zeros((class_count, max(len(events), 1), EVIDENCE_DIM), dtype=np.float32)
    normalized_burden = float(math.sqrt(max(float(np.asarray(burden).ravel()[0]), 1.0)))
    for event_index, (gene, event_type, allele) in enumerate(events):
        token = f"{gene}__{event_type}"
        token_weight = np.asarray(weights.get(token, np.zeros(class_count, dtype=np.float32)), dtype=np.float32)
        token_support = float(support.get(token, 0))
        type_index = EVENT_TYPE_INDEX.get(event_type, EVENT_TYPE_INDEX["OTHER"])
        for class_index in range(class_count):
            contribution = float(token_weight[class_index])
            output[class_index, event_index, 0] = contribution
            output[class_index, event_index, 1] = abs(contribution)
            output[class_index, event_index, 2] = math.log1p(token_support)
            output[class_index, event_index, 3] = token_support / (token_support + 20.0)
            output[class_index, event_index, 4] = float(contribution > 0)
            output[class_index, event_index, 5] = float(contribution < 0)
            output[class_index, event_index, 6] = contribution / normalized_burden
            output[class_index, event_index, 7] = float(bool(allele))
            output[class_index, event_index, 8] = float(token_support >= 5)
            output[class_index, event_index, 9 + type_index] = 1.0
    return output


def pad_evidence_sets(evidence_sets: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad sample tensors to [batch, candidate, event, feature] and return mask."""
    if not evidence_sets:
        raise ValueError("at least one sample is required")
    candidate_count = evidence_sets[0].shape[0]
    feature_count = evidence_sets[0].shape[2]
    max_events = max(item.shape[1] for item in evidence_sets)
    features = np.zeros((len(evidence_sets), candidate_count, max_events, feature_count), dtype=np.float32)
    mask = np.zeros((len(evidence_sets), candidate_count, max_events), dtype=bool)
    for index, evidence in enumerate(evidence_sets):
        if evidence.shape[0] != candidate_count or evidence.shape[2] != feature_count:
            raise ValueError("candidate/feature dimensions must agree")
        length = evidence.shape[1]
        features[index, :, :length] = evidence
        mask[index, :, :length] = True
    return torch.from_numpy(features), torch.from_numpy(mask)


class EvidenceSetNetwork(nn.Module):
    """One event encoder shared by all patients and all candidate classes."""

    def __init__(self, input_dim: int = EVIDENCE_DIM, hidden_dim: int = 32, dropout: float = 0.15):
        super().__init__()
        self.event_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.score = nn.Linear(hidden_dim * 3, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.event_mlp(features)
        expanded_mask = mask.unsqueeze(-1)
        count = expanded_mask.sum(dim=2).clamp_min(1)
        summed = (encoded * expanded_mask).sum(dim=2)
        mean = summed / count
        maximum = encoded.masked_fill(~expanded_mask, -torch.inf).amax(dim=2)
        maximum = torch.nan_to_num(maximum, nan=0.0, neginf=0.0, posinf=0.0)
        return self.score(torch.cat([mean, maximum, summed], dim=-1)).squeeze(-1)


def listwise_loss(logits: torch.Tensor, target_index: torch.Tensor, class_weight: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target_index, weight=class_weight.to(logits.device))


def nested_audit(outer_train: np.ndarray, inner_oof_rows: np.ndarray, outer_validation: np.ndarray) -> dict[str, bool]:
    """Record the core no-leakage relationship for one outer fold."""
    outer_train_set = set(np.asarray(outer_train, dtype=np.int64).tolist())
    inner_set = set(np.asarray(inner_oof_rows, dtype=np.int64).tolist())
    validation_set = set(np.asarray(outer_validation, dtype=np.int64).tolist())
    return {
        "ranker_training_rows_are_inner_oof": inner_set == outer_train_set,
        "outer_validation_used_for_eb_fit": bool(outer_train_set & validation_set or inner_set & validation_set),
    }
