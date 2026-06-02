"""Aggregate profiling: bytes-per-vector + query latency for a compressor.

This module exposes a single :func:`profile_compressor` helper that, given a
compressor, an embedded corpus, and a small query batch, returns the headline
efficiency objectives consumed by the multi-objective optimizer:

* ``bytes_per_vector`` — storage cost of one compressed vector
* ``corpus_bytes`` — derived from the above and ``n_corpus``
* ``query_latency_ms`` — median wall-time for a single-query top-k search
* ``query_p95_ms`` — tail latency for the same operation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from embedopt.compression.base import CompressedSet, Compressor
from embedopt.profiling.timing import time_callable
from embedopt.utils.types import FloatArray


@dataclass(slots=True, frozen=True)
class EfficiencyStats:
    """Profiling outputs used as MOO objectives."""

    bytes_per_vector: int
    corpus_bytes: int
    fit_ms: float
    encode_ms: float
    query_latency_ms: float
    query_p95_ms: float


def profile_compressor(
    compressor: Compressor,
    corpus: FloatArray,
    queries: FloatArray,
    *,
    n_repeats: int = 15,
    n_warmup: int = 2,
) -> tuple[EfficiencyStats, CompressedSet]:
    """Profile ``compressor`` on a corpus / query pair and return its statistics.

    The compressed corpus is also returned so the caller (typically the Pareto
    pipeline) can re-use it for quality evaluation without paying the encode
    cost twice.
    """
    fit_ms = 0.0
    if not compressor.trained:
        fit_t0 = time.perf_counter_ns()
        compressor.fit(corpus)
        fit_ms = (time.perf_counter_ns() - fit_t0) / 1e6
    encode_t0 = time.perf_counter_ns()
    compressed = compressor.transform(corpus)
    encode_ms = (time.perf_counter_ns() - encode_t0) / 1e6
    # Single-query latency is the most common deployment scenario; we time
    # one query at a time over the full corpus.
    single_query = queries[:1]

    def _one_query() -> None:
        compressor.score(single_query, compressed)

    timing = time_callable(_one_query, n_repeats=n_repeats, n_warmup=n_warmup)
    n_corpus = corpus.shape[0]
    stats = EfficiencyStats(
        bytes_per_vector=compressed.bytes_per_vector,
        corpus_bytes=int(compressed.bytes_per_vector) * n_corpus,
        fit_ms=float(fit_ms),
        encode_ms=float(encode_ms),
        query_latency_ms=timing.median_ms,
        query_p95_ms=timing.p95_ms,
    )
    return stats, compressed
