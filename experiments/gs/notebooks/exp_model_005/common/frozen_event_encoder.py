"""Frozen encoder helpers. No tokenizer fitting or external annotation."""
from __future__ import annotations

import numpy as np


# Microsoft renamed this public checkpoint from PubMedBERT to BiomedBERT.
# Pin the published commit so a future `main` update cannot change the experiment.
MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MODEL_REVISION = "2839b4fc440a3c41dc2b716fb14d530c33c8c1ff"
MAX_LENGTH = 32
BATCH_SIZE = 32


def event_sentence(gene: str, event_type: str, ref: str | None, position: int | None, alt: str | None) -> str:
    return f"gene {gene} type {event_type} ref {ref or 'NONE'} position {position if position is not None else 'NONE'} alt {alt or 'NONE'}"


def pool_event_embeddings(event_embeddings: np.ndarray, embedding_dim: int) -> np.ndarray:
    if event_embeddings.size == 0:
        return np.zeros(embedding_dim * 2 + 1, dtype=np.float32)
    return np.concatenate([event_embeddings.mean(0), event_embeddings.max(0), [np.log1p(len(event_embeddings))]]).astype(np.float32)
