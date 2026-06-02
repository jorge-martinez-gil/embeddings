"""End-to-end evaluation runners.

The runners are intentionally thin: they accept already-constructed embedders
and compressors so callers can mock or swap any component for unit tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from embedopt.compression.base import Compressor
from embedopt.evaluation.datasets import RetrievalDataset, STSDataset
from embedopt.evaluation.metrics import (
    QrelMap,
    cosine_pairs,
    map_score,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    spearman_correlation,
)
from embedopt.models.backbones import TextEmbedder
from embedopt.utils.types import FloatArray, as_float_array


@dataclass(slots=True, frozen=True)
class STSResult:
    """Metric bundle for STS evaluation."""

    spearman: float
    mean_predicted: float
    n_pairs: int


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """Metric bundle for retrieval evaluation."""

    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    map: float
    n_queries: int
    n_corpus: int


def _embed(embedder: TextEmbedder, texts: list[str]) -> FloatArray:
    return as_float_array(embedder.encode(texts))


def evaluate_sts(
    embedder: TextEmbedder,
    compressor: Compressor,
    dataset: STSDataset,
) -> STSResult:
    """Evaluate STS Spearman correlation.

    The compressor is fit on the union of A- and B-side embeddings (a single
    pass over the data), so codecs that need training (PQ, scalar) see the
    same distribution they will be scored against.
    """
    a = _embed(embedder, list(dataset.sentences_a))
    b = _embed(embedder, list(dataset.sentences_b))
    if not compressor.trained:
        compressor.fit(np.vstack([a, b]).astype(np.float32, copy=False))
    a_codes = compressor.transform(a)
    # Score each (a_i, b_i) pair: compressors expect a (1, d) query × (1, d) corpus,
    # so we loop. This is O(n) and the pair count is small in practice.
    sims: list[float] = []
    for i in range(a.shape[0]):
        codes_i = type(a_codes)(
            codes=a_codes.codes[i : i + 1],
            bytes_per_vector=a_codes.bytes_per_vector,
        )
        score = compressor.score(b[i : i + 1], codes_i)
        sims.append(float(score[0, 0]))
    spearman = spearman_correlation(sims, list(dataset.scores))
    # Sanity reference: cosine similarity of raw vectors (uncompressed).
    raw_cos = cosine_pairs(a, b)
    mean_pred = float(np.mean(raw_cos))
    return STSResult(spearman=spearman, mean_predicted=mean_pred, n_pairs=a.shape[0])


def evaluate_retrieval(
    embedder: TextEmbedder,
    compressor: Compressor,
    dataset: RetrievalDataset,
    *,
    k: int = 10,
) -> RetrievalResult:
    """Evaluate Recall/MRR/nDCG/MAP at depth ``k``."""
    corpus = _embed(embedder, list(dataset.corpus))
    queries = _embed(embedder, list(dataset.queries))
    if not compressor.trained:
        compressor.fit(corpus)
    compressed = compressor.transform(corpus)
    scores = compressor.score(queries, compressed)
    qrels: QrelMap = dataset.qrels
    return RetrievalResult(
        recall_at_10=recall_at_k(scores, qrels, k=k),
        mrr_at_10=mrr_at_k(scores, qrels, k=k),
        ndcg_at_10=ndcg_at_k(scores, qrels, k=k),
        map=map_score(scores, qrels, k=k),
        n_queries=queries.shape[0],
        n_corpus=corpus.shape[0],
    )


def quality_score(result: Mapping[str, float], primary: str = "ndcg_at_10") -> float:
    """Extract a single scalar quality metric from a result dict."""
    if primary not in result:
        raise KeyError(f"Result has no metric named {primary!r}; keys: {list(result)}")
    return float(result[primary])
