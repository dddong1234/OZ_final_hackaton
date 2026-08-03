"""exp_model_003: P1+EB 기준선 위의 train-only 신규 축 공통 구현."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import importlib.util
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


_MISSENSE = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")
_POSITION = re.compile(r"(\d+)")
_MULTI = re.compile(r"\s*[;,|]\s*")


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "data/raw/train.csv").exists():
            return parent
    raise FileNotFoundError("data/raw/train.csv를 찾지 못했습니다.")


def exp002_common() -> Path:
    path = project_root() / "experiments/gs/notebooks/exp_model_002/common"
    if not path.exists():
        raise FileNotFoundError(f"P1+EB 기준선 common 경로가 없습니다: {path}")
    return path


def legacy_runner_module():
    """프로젝트 내부의 검증된 P1+EB 기준선만 호환 reference로 읽는다."""
    module_path = exp002_common() / "run_p1_axis.py"
    name = "exp_model_003_reference_runner"
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(exp002_common()))
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"reference runner를 불러올 수 없습니다: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class EventDetail:
    gene: str
    raw: str | None
    event_type: str
    position: int | None
    ref: str | None
    alt: str | None


def parse_event_detail(gene: str, value: object) -> EventDetail:
    """WT/빈 문자열/NaN을 NONE으로 유지하는 결정론적 parser."""
    if not isinstance(value, str):
        return EventDetail(gene, None, "NONE", None, None, None)
    raw = value.strip().upper()
    if raw in {"", "WT", "NAN"}:
        return EventDetail(gene, None, "NONE", None, None, None)
    missense = _MISSENSE.match(raw)
    position_match = _POSITION.search(raw)
    position = int(position_match.group(1)) if position_match else None
    if "FS" in raw:
        event_type = "FRAMESHIFT"
    elif "DELINS" in raw:
        event_type = "DELINS"
    elif "INS" in raw:
        event_type = "INFRAME_INS"
    elif "DEL" in raw:
        event_type = "INFRAME_DEL"
    elif "SPLICE" in raw:
        event_type = "SPLICE"
    elif "TER" in raw or "*" in raw or "X" in raw:
        event_type = "NONSENSE"
    elif missense:
        event_type = "MISSENSE"
    else:
        event_type = "OTHER"
    return EventDetail(
        gene=gene,
        raw=raw,
        event_type=event_type,
        position=position,
        ref=missense.group(1) if missense else None,
        alt=missense.group(3) if missense else None,
    )


def split_cell_events(gene: str, value: object) -> list[EventDetail]:
    if not isinstance(value, str):
        return []
    pieces = [piece for piece in _MULTI.split(value) if piece.strip()]
    events = [parse_event_detail(gene, piece) for piece in pieces]
    return [event for event in events if event.event_type != "NONE"]


def build_event_rows(frame: pd.DataFrame) -> list[list[EventDetail]]:
    genes = list(frame.columns)
    all_rows: list[list[EventDetail]] = []
    for row in frame.itertuples(index=False, name=None):
        events: list[EventDetail] = []
        for gene, value in zip(genes, row):
            events.extend(split_cell_events(gene, value))
        all_rows.append(events)
    return all_rows


def aggregate_event_evidence(event_scores: np.ndarray, n_classes: int) -> np.ndarray:
    """이벤트별 class evidence를 sum/max/top2 summary로 3*C 벡터화한다."""
    if event_scores.size == 0:
        return np.zeros(n_classes * 3, dtype=np.float32)
    if event_scores.ndim != 2 or event_scores.shape[1] != n_classes:
        raise ValueError("event_scores shape는 (n_events, n_classes)여야 합니다.")
    summed = event_scores.sum(axis=0)
    maximum = event_scores.max(axis=0)
    ordered = np.sort(event_scores, axis=0)
    top2 = ordered[-1] + (ordered[-2] if len(ordered) > 1 else 0.0)
    return np.concatenate([summed, maximum, top2]).astype(np.float32)


def summarize_ranks(probability: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF 확률만으로 true-rank, Recall@k, oracle@k를 계산한다."""
    if probability.shape != (len(labels), len(classes)):
        raise ValueError("probability와 labels/classes shape가 일치하지 않습니다.")
    index = {label: i for i, label in enumerate(classes)}
    true_idx = np.asarray([index[label] for label in labels])
    order = np.argsort(-probability, axis=1)
    rank = np.empty(len(labels), dtype=int)
    for row, target in enumerate(true_idx):
        rank[row] = int(np.where(order[row] == target)[0][0]) + 1
    top = np.sort(probability, axis=1)[:, -2:]
    rows = pd.DataFrame({
        "true_class": labels,
        "true_rank": rank,
        "top1_class": classes[order[:, 0]],
        "top1_probability": probability[np.arange(len(labels)), order[:, 0]],
        "margin": top[:, 1] - top[:, 0],
        "entropy": -(probability * np.log(np.clip(probability, 1e-12, 1))).sum(axis=1),
    })
    rows["correct"] = rows["true_class"].eq(rows["top1_class"])
    summary: dict[str, float] = {}
    for k in (1, 2, 3):
        recall = []
        for label in classes:
            member = labels == label
            if member.any():
                recall.append(float((rank[member] <= k).mean()))
        summary[f"macro_recall_at_{k}"] = float(np.mean(recall))
        oracle = classes[order[:, 0]].copy()
        can_recover = rank <= k
        oracle[can_recover] = labels[can_recover]
        summary[f"oracle_macro_f1_at_{k}"] = float(f1_score(labels, oracle, average="macro"))
    return pd.DataFrame([summary]), rows


def parser_tables(events: list[list[EventDetail]], labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    flat = [(row, event) for row, sample in enumerate(events) for event in sample]
    if not flat:
        empty = pd.DataFrame(columns=["event_type", "count", "rate"])
        return empty, empty
    type_count = Counter(event.event_type for _, event in flat)
    total = sum(type_count.values())
    coverage = pd.DataFrame([
        {"event_type": key, "count": value, "rate": value / total}
        for key, value in sorted(type_count.items())
    ])
    class_rows = []
    for label in sorted(np.unique(labels)):
        selected = [event for row, event in flat if labels[row] == label]
        denom = max(len(selected), 1)
        for event_type, count in Counter(event.event_type for event in selected).items():
            class_rows.append({"class": label, "event_type": event_type, "count": count, "rate": count / denom})
    return coverage, pd.DataFrame(class_rows)


def support_tables(events: list[list[EventDetail]]) -> pd.DataFrame:
    columns = ["gene", "event_count", "same_codon", "position_span"]
    rows = []
    for sample in events:
        by_gene: dict[str, list[EventDetail]] = defaultdict(list)
        for event in sample:
            by_gene[event.gene].append(event)
        for gene, group in by_gene.items():
            if len(group) >= 2:
                positions = [event.position for event in group if event.position is not None]
                rows.append({
                    "gene": gene,
                    "event_count": len(group),
                    "same_codon": len({event.position for event in group if event.position is not None}) < len(positions),
                    "position_span": max(positions) - min(positions) if len(positions) >= 2 else np.nan,
                })
    return pd.DataFrame(rows, columns=columns)


def token_sets_from_events(events: list[list[EventDetail]], level: str) -> list[set[str]]:
    token_sets: list[set[str]] = []
    for sample in events:
        tokens = set()
        for event in sample:
            if level == "gene_type":
                tokens.add(f"{event.gene}__{event.event_type}")
            elif level == "exact" and event.raw:
                tokens.add(f"{event.gene}__{event.raw}")
            elif level == "codon" and event.position is not None:
                tokens.add(f"{event.gene}__POS{event.position}")
            else:
                raise ValueError(f"지원하지 않는 token level: {level}")
        token_sets.append(tokens)
    return token_sets


def _event_weight_map(token_sets: list[set[str]], fit_idx: np.ndarray, labels: np.ndarray, classes: np.ndarray, fit_log_odds) -> tuple[dict[str, np.ndarray], Counter]:
    selected = [token_sets[i] for i in fit_idx]
    weights = fit_log_odds(selected, labels[fit_idx], classes, empirical_bayes=True)
    support = Counter(token for tokens in selected for token in tokens)
    return weights, support


def fit_point_process(events: list[list[EventDetail]], fit_idx: np.ndarray, labels: np.ndarray, classes: np.ndarray, fit_log_odds) -> dict:
    """exact→codon→continuous local density→gene×type 순의 fold-train backoff model."""
    exact_sets = token_sets_from_events(events, "exact")
    codon_sets = token_sets_from_events(events, "codon")
    type_sets = token_sets_from_events(events, "gene_type")
    exact_w, exact_n = _event_weight_map(exact_sets, fit_idx, labels, classes, fit_log_odds)
    codon_w, codon_n = _event_weight_map(codon_sets, fit_idx, labels, classes, fit_log_odds)
    type_w, type_n = _event_weight_map(type_sets, fit_idx, labels, classes, fit_log_odds)
    index = {label: i for i, label in enumerate(classes)}
    class_n = np.bincount([index[labels[i]] for i in fit_idx], minlength=len(classes)).astype(np.float64)
    position_by_key_class: dict[tuple[str, str], list[list[int]]] = {}
    position_by_key_all: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in fit_idx:
        ci = index[labels[i]]
        for event in events[i]:
            if event.position is None:
                continue
            key = (event.gene, event.event_type)
            if key not in position_by_key_class:
                position_by_key_class[key] = [[] for _ in classes]
            position_by_key_class[key][ci].append(event.position)
            position_by_key_all[key].append(event.position)
    return {
        "exact_w": exact_w, "exact_n": exact_n, "codon_w": codon_w, "codon_n": codon_n,
        "type_w": type_w, "type_n": type_n, "position_by_key_class": position_by_key_class,
        "position_by_key_all": position_by_key_all, "class_n": class_n, "classes": classes,
        "bandwidth": 20.0, "reliability": 20.0,
    }


def _density_log_odds(model: dict, event: EventDetail) -> tuple[np.ndarray, int]:
    classes = model["classes"]
    if event.position is None:
        return np.zeros(len(classes), dtype=np.float32), 0
    key = (event.gene, event.event_type)
    arrays = model["position_by_key_class"].get(key)
    if arrays is None:
        return np.zeros(len(classes), dtype=np.float32), 0
    bandwidth = model["bandwidth"]
    class_n = model["class_n"]
    all_positions = np.asarray(model["position_by_key_all"][key], dtype=float)
    local_count = int((np.abs(all_positions - event.position) <= bandwidth).sum())
    density = np.zeros(len(classes), dtype=np.float64)
    for ci, values in enumerate(arrays):
        if values:
            delta = (np.asarray(values, dtype=float) - event.position) / bandwidth
            density[ci] = np.exp(-0.5 * delta * delta).sum() / class_n[ci]
    total_density = density * class_n
    other = (total_density.sum() - total_density) / np.maximum(class_n.sum() - class_n, 1.0)
    score = np.log((density + 1e-5) / (other + 1e-5))
    return np.clip(score, -4.0, 4.0).astype(np.float32), local_count


def point_event_score(model: dict, event: EventDetail) -> np.ndarray:
    classes = model["classes"]
    zero = np.zeros(len(classes), dtype=np.float32)
    exact_token = f"{event.gene}__{event.raw}" if event.raw else ""
    codon_token = f"{event.gene}__POS{event.position}" if event.position is not None else ""
    type_token = f"{event.gene}__{event.event_type}"
    exact = model["exact_w"].get(exact_token, zero)
    codon = model["codon_w"].get(codon_token, zero)
    base = model["type_w"].get(type_token, zero)
    density, local_n = _density_log_odds(model, event)
    r = model["reliability"]
    w_exact = model["exact_n"].get(exact_token, 0) / (model["exact_n"].get(exact_token, 0) + r)
    remaining = 1.0 - w_exact
    w_codon = remaining * model["codon_n"].get(codon_token, 0) / (model["codon_n"].get(codon_token, 0) + r)
    remaining -= w_codon
    w_density = remaining * local_n / (local_n + r)
    w_type = 1.0 - w_exact - w_codon - w_density
    return (w_exact * exact + w_codon * codon + w_density * density + w_type * base).astype(np.float32)


def point_score_matrix(model: dict, event_rows: Iterable[list[EventDetail]]) -> np.ndarray:
    n_classes = len(model["classes"])
    output = []
    for sample in event_rows:
        scores = np.asarray([point_event_score(model, event) for event in sample], dtype=np.float32)
        output.append(aggregate_event_evidence(scores, n_classes))
    return np.asarray(output, dtype=np.float32)


def cross_fitted_scores(
    builder: Callable[[np.ndarray], dict],
    applier: Callable[[dict, np.ndarray], np.ndarray],
    fit_idx: np.ndarray,
    labels: np.ndarray,
    seed: int,
    width: int,
) -> np.ndarray:
    result = np.zeros((len(fit_idx), width), dtype=np.float32)
    split = StratifiedKFold(5, shuffle=True, random_state=seed)
    for inner_train, inner_valid in split.split(fit_idx, labels[fit_idx]):
        ti, vi = fit_idx[inner_train], fit_idx[inner_valid]
        result[inner_valid] = applier(builder(ti), vi)
    return result


def lowrank_weight_builder(token_sets: list[set[str]], fit_idx: np.ndarray, labels: np.ndarray, classes: np.ndarray, fit_log_odds, rank: int) -> dict[str, np.ndarray]:
    raw = fit_log_odds([token_sets[i] for i in fit_idx], labels[fit_idx], classes, empirical_bayes=True)
    if not raw:
        return {}
    keys = sorted(raw)
    matrix = np.asarray([raw[key] for key in keys], dtype=np.float32)
    components = min(rank, matrix.shape[0], matrix.shape[1])
    if components < 1:
        return raw
    svd = TruncatedSVD(n_components=components, random_state=0)
    reconstructed = svd.inverse_transform(svd.fit_transform(matrix)).astype(np.float32)
    return {key: reconstructed[i] for i, key in enumerate(keys)}


def apply_weight_scores(token_sets: list[set[str]], idx: np.ndarray, weights: dict[str, np.ndarray], n_classes: int) -> np.ndarray:
    out = np.zeros((len(idx), n_classes), dtype=np.float32)
    for row, original in enumerate(idx):
        active = [weights[token] for token in token_sets[original] if token in weights]
        if active:
            out[row] = np.sum(active, axis=0) / np.sqrt(len(active))
    return out
