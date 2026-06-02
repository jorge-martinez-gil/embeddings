"""Evaluation metrics for STS and retrieval.

All retrieval metrics consume a *score matrix* ``S`` of shape
``(n_queries, n_corpus)`` where higher is better, plus a *qrels* mapping
``query_id -> {doc_id: relevance}``. Documents and queries are identified by
their row index in ``S``. Implementations are dependency-free (numpy only) so
the package stays lightweight and trivially auditable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from embedopt.utils.types import FloatArray

QrelMap = Mapping[int, Mapping[int, float]]
_Float64Array = NDArray[np.float64]
_Int64Array = NDArray[np.int64]


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation between two equally-sized sequences.

    Implemented via Pearson correlation of average-ranked values, which handles
    ties correctly. Returns ``0.0`` if either input has zero variance after
    ranking.
    """
    if len(x) != len(y):
        raise ValueError("Inputs must have the same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 points")
    rx = _average_rank(np.asarray(x, dtype=np.float64))
    ry = _average_rank(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _average_rank(values: _Float64Array) -> _Float64Array:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values)
    n = values.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def cosine_pairs(a: FloatArray, b: FloatArray) -> FloatArray:
    """Row-wise cosine similarity of two equally-shaped matrices."""
    if a.shape != b.shape:
        raise ValueError("Inputs must have identical shape")
    num = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom == 0.0, 1.0, denom)
    out: FloatArray = (num / denom).astype(np.float32, copy=False)
    return out


def recall_at_k(scores: FloatArray, qrels: QrelMap, k: int) -> float:
    """Mean Recall@k over queries with at least one relevant document."""
    if k <= 0:
        raise ValueError("k must be positive")
    rankings = _topk_indices(scores, k)
    hits: list[float] = []
    for qid, qrel in qrels.items():
        relevant = {doc for doc, rel in qrel.items() if rel > 0}
        if not relevant:
            continue
        topk = rankings[qid]
        hit = sum(1 for d in topk if d in relevant)
        hits.append(hit / len(relevant))
    return float(np.mean(hits)) if hits else 0.0


def mrr_at_k(scores: FloatArray, qrels: QrelMap, k: int) -> float:
    """Mean Reciprocal Rank cut at depth ``k``."""
    if k <= 0:
        raise ValueError("k must be positive")
    rankings = _topk_indices(scores, k)
    rrs: list[float] = []
    for qid, qrel in qrels.items():
        relevant = {doc for doc, rel in qrel.items() if rel > 0}
        if not relevant:
            continue
        rr = 0.0
        for rank, d in enumerate(rankings[qid], start=1):
            if d in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def ndcg_at_k(scores: FloatArray, qrels: QrelMap, k: int) -> float:
    """Mean nDCG@k with binary or graded relevance (gain ``2**rel - 1``)."""
    if k <= 0:
        raise ValueError("k must be positive")
    rankings = _topk_indices(scores, k)
    out: list[float] = []
    for qid, qrel in qrels.items():
        if not any(r > 0 for r in qrel.values()):
            continue
        gains = np.array(
            [(2.0 ** qrel.get(int(d), 0.0)) - 1.0 for d in rankings[qid]],
            dtype=np.float64,
        )
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
        dcg = float((gains * discounts).sum())
        ideal_rels = sorted(qrel.values(), reverse=True)[:k]
        ideal_gains = np.array([(2.0**r) - 1.0 for r in ideal_rels], dtype=np.float64)
        ideal_disc = 1.0 / np.log2(np.arange(2, ideal_gains.size + 2, dtype=np.float64))
        idcg = float((ideal_gains * ideal_disc).sum())
        if idcg > 0:
            out.append(dcg / idcg)
    return float(np.mean(out)) if out else 0.0


def map_score(scores: FloatArray, qrels: QrelMap, k: int | None = None) -> float:
    """Mean Average Precision (optionally truncated at depth ``k``)."""
    cutoff = k if k is not None else scores.shape[1]
    if cutoff <= 0:
        raise ValueError("k must be positive")
    rankings = _topk_indices(scores, cutoff)
    aps: list[float] = []
    for qid, qrel in qrels.items():
        relevant = {doc for doc, rel in qrel.items() if rel > 0}
        if not relevant:
            continue
        hits = 0
        precisions: list[float] = []
        for rank, d in enumerate(rankings[qid], start=1):
            if d in relevant:
                hits += 1
                precisions.append(hits / rank)
        ap = sum(precisions) / len(relevant) if precisions else 0.0
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def _topk_indices(scores: FloatArray, k: int) -> _Int64Array:
    n_q, n_c = scores.shape
    k_eff = min(k, n_c)
    if k_eff == 0:
        return np.zeros((n_q, 0), dtype=np.int64)
    # Argpartition for top-k, then sort just the slice for stable ranking.
    part = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(n_q)[:, None]
    sorted_within = np.argsort(-scores[rows, part], axis=1)
    topk: _Int64Array = part[rows, sorted_within].astype(np.int64, copy=False)
    return topk
