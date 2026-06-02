from __future__ import annotations

import numpy as np

from embedopt.compression import IdentityCompressor, TruncateCompressor
from embedopt.profiling import (
    measure_peak_memory,
    profile_compressor,
    time_callable,
)


def test_time_callable_returns_positive_stats() -> None:
    stats = time_callable(lambda: sum(range(1000)), n_repeats=10, n_warmup=1)
    assert stats.median_ms >= 0.0
    assert stats.p95_ms >= stats.median_ms - 1e-9
    assert stats.n_repeats == 10


def test_measure_peak_memory_records_allocation() -> None:
    def alloc() -> None:
        _ = [0] * 100_000

    stats = measure_peak_memory(alloc)
    assert stats.peak_bytes > 0


def test_profile_compressor_returns_stats() -> None:
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((30, 16)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = corpus[:3]
    stats, compressed = profile_compressor(
        IdentityCompressor(), corpus, queries, n_repeats=5, n_warmup=1
    )
    assert stats.bytes_per_vector == 16 * 4
    assert stats.corpus_bytes == 30 * 16 * 4
    assert stats.fit_ms >= 0.0
    assert stats.encode_ms >= 0.0
    assert stats.query_latency_ms >= 0.0
    assert compressed.codes.shape == (30, 16)


def test_profile_compressor_truncate_byte_cost() -> None:
    rng = np.random.default_rng(1)
    corpus = rng.standard_normal((10, 32)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = corpus[:2]
    stats, _ = profile_compressor(
        TruncateCompressor(keep_dim=8), corpus, queries, n_repeats=3, n_warmup=1
    )
    assert stats.bytes_per_vector == 8 * 4
