from __future__ import annotations

import numpy as np
import pytest

from embedopt.compression import IdentityCompressor, TruncateCompressor
from embedopt.indexing import (
    KNOWN_BACKENDS,
    FaissHNSWIndex,
    FaissIVFIndex,
    FaissIVFPQIndex,
    FaissOPQIndex,
    dense_views_for_compressor,
    evaluate_dense_index,
    make_dense_index,
)


def test_exact_numpy_index_evaluates_retrieval_metrics() -> None:
    corpus = np.eye(4, dtype=np.float32)
    queries = corpus[:2]
    qrels = {0: {0: 1.0}, 1: {1: 1.0}}
    out = evaluate_dense_index(
        corpus_vectors=corpus,
        query_vectors=queries,
        qrels=qrels,
        backend="exact-numpy",
        k=2,
        n_repeats=3,
        n_warmup=1,
    )
    assert out.backend == "exact-numpy"
    assert out.index_bytes == corpus.nbytes
    assert out.recall_at_10 == 1.0
    assert out.ndcg_at_10 == 1.0


def test_dense_views_project_truncated_queries() -> None:
    corpus = np.eye(4, dtype=np.float32)
    queries = corpus[:2]
    compressor = TruncateCompressor(keep_dim=2)
    compressed = compressor.transform(corpus)
    views = dense_views_for_compressor(compressor, queries, compressed)
    assert views is not None
    query_view, corpus_view = views
    assert query_view.shape == (2, 2)
    assert corpus_view.shape == (4, 2)


def test_identity_dense_view_is_available() -> None:
    corpus = np.eye(3, dtype=np.float32)
    queries = corpus[:1]
    compressor = IdentityCompressor()
    compressed = compressor.transform(corpus)
    assert dense_views_for_compressor(compressor, queries, compressed) is not None


def test_known_backends_lists_production_ann() -> None:
    # The runner's --index-backends flag and downstream tools key off this
    # tuple; locking it in makes accidental renames a test failure.
    assert KNOWN_BACKENDS == (
        "exact-numpy",
        "faiss-flat",
        "faiss-ivf",
        "faiss-ivfpq",
        "faiss-hnsw",
        "faiss-opq",
    )


def test_make_dense_index_constructs_each_known_backend() -> None:
    # We can build a Python instance without faiss installed; only the
    # ``.build()`` call needs it.
    for backend in KNOWN_BACKENDS:
        idx = make_dense_index(backend)
        assert idx.name == backend


def test_make_dense_index_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        make_dense_index("no-such-backend")


def _ann_corpus(seed: int = 0) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Small clustered corpus suitable for IVF training (n=320, d=16)."""
    rng = np.random.default_rng(seed)
    n_clusters, per_cluster, d = 16, 20, 16
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
    for c in centers:
        noise = 0.05 * rng.standard_normal((per_cluster, d)).astype(np.float32)
        block = c[None, :] + noise
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        rows.append(block)
    return np.vstack(rows).astype(np.float32, copy=False)


def test_faiss_ivf_index_recovers_self_neighbors() -> None:
    pytest.importorskip("faiss")
    corpus = _ann_corpus(seed=1)
    queries = corpus[:8]
    # IVF must find each query's own row at rank 1 with nprobe set high enough.
    index = FaissIVFIndex(nlist=8, nprobe=8)
    index.build(corpus)
    scores, idx = index.search(queries, k=1)
    assert scores.shape == (8, 1)
    assert (idx.flatten() == np.arange(8)).mean() >= 0.875


def test_faiss_ivfpq_index_returns_topk_and_reports_bytes() -> None:
    pytest.importorskip("faiss")
    corpus = _ann_corpus(seed=2)
    queries = corpus[:4]
    # d=16 must be divisible by m; m=4 -> 4 subspaces of 4 dims each.
    index = FaissIVFPQIndex(nlist=8, nprobe=8, m=4, n_bits=4)
    index.build(corpus)
    scores, idx = index.search(queries, k=5)
    assert scores.shape == (4, 5)
    assert idx.shape == (4, 5)
    assert index.index_bytes > 0


def test_faiss_ivfpq_rejects_indivisible_dim() -> None:
    pytest.importorskip("faiss")
    corpus = _ann_corpus(seed=3)  # d=16
    index = FaissIVFPQIndex(m=7, n_bits=4)  # 16 % 7 != 0
    with pytest.raises(ValueError):
        index.build(corpus)


def test_faiss_ivfpq_auto_degrades_on_smoke_corpus() -> None:
    """20-vector smoke corpora can't train 2**8=256 centroids; we
    walk down to a feasible n_bits instead of failing outright."""
    pytest.importorskip("faiss")
    rng = np.random.default_rng(7)
    corpus = rng.standard_normal((20, 16)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    index = FaissIVFPQIndex(m=4, n_bits=8, nlist=2, nprobe=2)
    index.build(corpus)
    # 2 ** 4 = 16 <= 20, so 4-bit is the largest feasible setting.
    assert index._effective_n_bits == 4
    scores, idx = index.search(corpus[:5], k=3)
    assert scores.shape == (5, 3)


def test_faiss_ivfpq_auto_degrade_can_be_disabled() -> None:
    pytest.importorskip("faiss")
    rng = np.random.default_rng(8)
    corpus = rng.standard_normal((20, 16)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    index = FaissIVFPQIndex(m=4, n_bits=8, auto_degrade_n_bits=False)
    with pytest.raises(ValueError, match="IVF-PQ needs at least"):
        index.build(corpus)


def test_evaluate_dense_index_runs_with_faiss_ivf() -> None:
    pytest.importorskip("faiss")
    corpus = _ann_corpus(seed=4)
    queries = corpus[:8]
    qrels = {i: {i: 1.0} for i in range(8)}
    out = evaluate_dense_index(
        corpus_vectors=corpus,
        query_vectors=queries,
        qrels=qrels,
        backend="faiss-ivf",
        k=5,
        n_repeats=2,
        n_warmup=1,
    )
    assert out.backend == "faiss-ivf"
    assert out.index_bytes > 0
    # On a clustered corpus with generous nprobe, IVF should easily recall
    # each query's own row.
    assert out.recall_at_10 >= 0.5


def test_evaluate_dense_index_runs_with_faiss_ivfpq() -> None:
    pytest.importorskip("faiss")
    corpus = _ann_corpus(seed=5)
    queries = corpus[:8]
    qrels = {i: {i: 1.0} for i in range(8)}
    out = evaluate_dense_index(
        corpus_vectors=corpus,
        query_vectors=queries,
        qrels=qrels,
        backend="faiss-ivfpq",
        k=5,
        n_repeats=2,
        n_warmup=1,
    )
    assert out.backend == "faiss-ivfpq"
    assert out.index_bytes > 0
    # IVF-PQ is lossy; we only assert the search ran and produced valid metrics.
    assert 0.0 <= out.recall_at_10 <= 1.0
    assert 0.0 <= out.ndcg_at_10 <= 1.0


def test_faiss_hnsw_index_recovers_self_neighbors() -> None:
    """HNSW with default efSearch should recover the trivial self-query top-1."""
    pytest.importorskip("faiss")
    corpus = _ann_corpus(seed=10)
    index = FaissHNSWIndex(m=16, ef_construction=64, ef_search=32)
    index.build(corpus)
    scores, idx = index.search(corpus[:8], k=1)
    assert scores.shape == (8, 1)
    assert (idx.flatten() == np.arange(8)).mean() >= 0.875
    assert index.index_bytes > 0


def test_faiss_hnsw_evaluate_dense_index() -> None:
    pytest.importorskip("faiss")
    corpus = _ann_corpus(seed=11)
    qrels = {i: {i: 1.0} for i in range(8)}
    out = evaluate_dense_index(
        corpus_vectors=corpus,
        query_vectors=corpus[:8],
        qrels=qrels,
        backend="faiss-hnsw",
        k=5,
        n_repeats=2,
        n_warmup=1,
    )
    assert out.backend == "faiss-hnsw"
    assert out.index_bytes > 0
    assert out.recall_at_10 >= 0.5


def test_faiss_opq_index_returns_topk_and_reports_bytes() -> None:
    pytest.importorskip("faiss")
    corpus = _ann_corpus(seed=12)
    index = FaissOPQIndex(m=4, n_bits=4)
    index.build(corpus)
    s, ids = index.search(corpus[:4], k=5)
    assert s.shape == (4, 5) and ids.shape == (4, 5)
    assert index.index_bytes > 0
    assert index._effective_n_bits == 4


def test_faiss_opq_auto_degrades_on_smoke_corpus() -> None:
    pytest.importorskip("faiss")
    rng = np.random.default_rng(13)
    corpus = rng.standard_normal((20, 16)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    index = FaissOPQIndex(m=4, n_bits=8)
    index.build(corpus)
    # 2 ** 4 = 16 <= 20, so 4-bit is the largest feasible setting.
    assert index._effective_n_bits == 4


def test_faiss_opq_rejects_indivisible_dim() -> None:
    pytest.importorskip("faiss")
    idx = FaissOPQIndex(m=7, n_bits=4)
    with pytest.raises(ValueError):
        idx.build(_ann_corpus(14))


def test_faiss_opq_evaluate_dense_index() -> None:
    pytest.importorskip("faiss")
    corpus = _ann_corpus(seed=15)
    qrels = {i: {i: 1.0} for i in range(8)}
    out = evaluate_dense_index(
        corpus_vectors=corpus,
        query_vectors=corpus[:8],
        qrels=qrels,
        backend="faiss-opq",
        k=5,
        n_repeats=2,
        n_warmup=1,
    )
    assert out.backend == "faiss-opq"
    assert out.index_bytes > 0
    assert 0.0 <= out.recall_at_10 <= 1.0
