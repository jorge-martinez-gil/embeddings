"""Index-level retrieval evaluation.

The compressor protocol measures the representation itself. This module adds
an index layer so experiments can also ask what happens after vectors are
handed to a retrieval backend. FAISS is used when installed; the exact NumPy
backend keeps tests and smoke runs dependency-free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

from embedopt.compression.base import CompressedSet, Compressor
from embedopt.evaluation.metrics import QrelMap
from embedopt.profiling.timing import time_callable
from embedopt.utils.types import FloatArray


class DenseSearchIndex(Protocol):
    """Minimal dense-vector index interface used by the experiment runner."""

    name: str

    def build(self, vectors: FloatArray) -> None:
        """Build or train the index over corpus vectors."""

    def search(self, queries: FloatArray, *, k: int) -> tuple[FloatArray, np.ndarray]:
        """Return top-k scores and integer corpus IDs for each query."""

    @property
    def index_bytes(self) -> int:
        """Approximate bytes occupied by the built index."""


@dataclass(slots=True)
class DenseIndexEval:
    """Metrics emitted for one dense index backend."""

    backend: str
    build_ms: float
    search_latency_ms: float
    search_p95_ms: float
    index_bytes: int
    recall_at_10: float
    ndcg_at_10: float
    exact_recall_at_10: float | None = None


@dataclass(slots=True)
class ExactNumpyIndex:
    """Dependency-free exact inner-product index."""

    name: str = "exact-numpy"
    _vectors: FloatArray | None = None

    def build(self, vectors: FloatArray) -> None:
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, queries: FloatArray, *, k: int) -> tuple[FloatArray, np.ndarray]:
        if self._vectors is None:
            raise RuntimeError("Index has not been built")
        scores = (queries @ self._vectors.T).astype(np.float32, copy=False)
        k_eff = min(k, scores.shape[1])
        idx = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
        part = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part, axis=1)
        top_idx = np.take_along_axis(idx, order, axis=1)
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        return top_scores, top_idx.astype(np.int64, copy=False)

    @property
    def index_bytes(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.nbytes)


def _default_nlist(n_corpus: int) -> int:
    """Pick a reasonable ``nlist`` for IVF from the corpus size.

    Rule of thumb borrowed from the FAISS guidelines: cluster count grows
    with ``sqrt(n)`` and is clamped so the index trains cleanly on very
    small / very large corpora. We also require at least one training
    sample per cluster so :class:`FaissIVFIndex` doesn't crash on toy
    inputs where ``sqrt(n)`` over-estimates.
    """
    if n_corpus <= 0:
        return 1
    candidate = int(np.sqrt(max(n_corpus, 1)))
    return max(1, min(candidate, max(1, n_corpus)))


@dataclass(slots=True)
class FaissIVFIndex:
    """FAISS inverted-file index (``IndexIVFFlat``) with inner-product metric.

    This is the textbook production ANN baseline: a coarse k-means quantizer
    partitions the corpus into ``nlist`` cells, queries visit only the top
    ``nprobe`` cells, and the resulting recall / latency trade-off is what
    every vector database with IVF support (FAISS, Milvus ``IVF_FLAT``,
    Qdrant ``Quantization=None`` with IVF, Vespa ``hnsw=false``) exposes.

    Parameters
    ----------
    nlist:
        Number of Voronoi cells. If ``None`` we pick ``sqrt(n_corpus)`` at
        build time, which is the FAISS recommendation for small / medium
        corpora.
    nprobe:
        Cells visited per query. ``None`` defaults to ``max(1, nlist // 8)``,
        a recall-favoring choice that still beats brute force on latency for
        ``n_corpus >= 1e4``.
    """

    nlist: int | None = None
    nprobe: int | None = None
    name: str = "faiss-ivf"
    _index: Any = None
    _quantizer: Any = None
    _dim: int = 0
    _fallback_bytes: int = 0

    def build(self, vectors: FloatArray) -> None:
        try:
            import faiss
        except ModuleNotFoundError as exc:  # pragma: no cover - optional path
            raise ModuleNotFoundError(
                "FAISS IVF backend requires faiss. "
                "Install faiss-cpu (or faiss-gpu)."
            ) from exc
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = int(arr.shape[0]), int(arr.shape[1])
        self._dim = d
        self._fallback_bytes = int(arr.nbytes)
        nlist = self.nlist if self.nlist is not None else _default_nlist(n)
        nlist = max(1, min(nlist, n)) if n > 0 else 1
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(arr)
        index.add(arr)
        index.nprobe = (
            self.nprobe if self.nprobe is not None else max(1, nlist // 8)
        )
        # FAISS keeps a raw pointer to the quantizer; hold a Python
        # reference so it survives as long as ``self._index`` does. Do
        # NOT set ``index.own_fields = True`` — that double-frees the
        # quantizer when Python's GC also drops it.
        self._quantizer = quantizer
        self._index = index

    def search(self, queries: FloatArray, *, k: int) -> tuple[FloatArray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index has not been built")
        q = np.ascontiguousarray(queries, dtype=np.float32)
        scores, idx = self._index.search(q, k)
        return cast(FloatArray, scores.astype(np.float32, copy=False)), idx.astype(np.int64)

    @property
    def index_bytes(self) -> int:
        if self._index is None:
            return 0
        try:
            import faiss

            return int(faiss.serialize_index(self._index).size)
        except Exception:  # pragma: no cover - best-effort accounting
            return self._fallback_bytes


@dataclass(slots=True)
class FaissIVFPQIndex:
    """FAISS IVF + Product Quantization (``IndexIVFPQ``) with inner product.

    Combines coarse IVF partitioning with PQ-compressed residual vectors —
    the canonical "compressed ANN" backend used by FAISS ``IndexIVFPQ``,
    Milvus ``IVF_PQ``, OpenSearch ``faiss/ivf_pq``, and ScaNN's hashed
    variants. Storage per vector is ``M`` bytes (one ``uint8`` per
    subspace) so the index is the closest production analog to this
    repo's standalone ``ProductQuantizeCompressor``.

    Parameters
    ----------
    nlist, nprobe:
        Same meaning as :class:`FaissIVFIndex`.
    m:
        Number of PQ subspaces. ``dim`` must be divisible by ``m``.
    n_bits:
        Bits per PQ code (``2 ** n_bits`` centroids per subspace). FAISS
        supports up to 8 in the standard build.
    """

    nlist: int | None = None
    nprobe: int | None = None
    m: int = 8
    n_bits: int = 8
    auto_degrade_n_bits: bool = True
    """If True, fall back to a smaller ``n_bits`` when the corpus is too
    small to train ``2 ** n_bits`` centroids. Useful for smoke runs on
    20-vector datasets where the full 8-bit codebook can't train but a
    4- or 2-bit variant still measures something meaningful."""

    name: str = "faiss-ivfpq"
    _index: Any = None
    _quantizer: Any = None
    _dim: int = 0
    _effective_n_bits: int = 0
    _fallback_bytes: int = 0

    def build(self, vectors: FloatArray) -> None:
        try:
            import faiss
        except ModuleNotFoundError as exc:  # pragma: no cover - optional path
            raise ModuleNotFoundError(
                "FAISS IVF-PQ backend requires faiss. "
                "Install faiss-cpu (or faiss-gpu)."
            ) from exc
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = int(arr.shape[0]), int(arr.shape[1])
        if d % self.m != 0:
            raise ValueError(
                f"IVF-PQ requires dim ({d}) divisible by m ({self.m})"
            )
        if self.n_bits < 1 or self.n_bits > 8:
            raise ValueError("IVF-PQ n_bits must be between 1 and 8")
        self._dim = d
        self._fallback_bytes = int(arr.nbytes)
        nlist = self.nlist if self.nlist is not None else _default_nlist(n)
        nlist = max(1, min(nlist, n)) if n > 0 else 1
        # PQ training needs at least 2 ** n_bits samples per subspace.
        # When ``auto_degrade_n_bits`` is on we walk down to the largest
        # ``n_bits`` the corpus can support (>= 1); otherwise we raise.
        effective_n_bits = self.n_bits
        if self.auto_degrade_n_bits:
            while effective_n_bits > 1 and n < (1 << effective_n_bits):
                effective_n_bits -= 1
        if n < (1 << effective_n_bits):
            raise ValueError(
                f"IVF-PQ needs at least {1 << effective_n_bits} training "
                f"vectors at n_bits={effective_n_bits}, got {n}"
            )
        self._effective_n_bits = effective_n_bits
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(
            quantizer, d, nlist, self.m, effective_n_bits, faiss.METRIC_INNER_PRODUCT
        )
        index.train(arr)
        index.add(arr)
        index.nprobe = (
            self.nprobe if self.nprobe is not None else max(1, nlist // 8)
        )
        # See note in FaissIVFIndex.build: keep a Python reference to the
        # quantizer instead of setting ``own_fields = True``.
        self._quantizer = quantizer
        self._index = index

# (Sentinel comment to anchor the next Edit; the OPQ class follows below.)

    def search(self, queries: FloatArray, *, k: int) -> tuple[FloatArray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index has not been built")
        q = np.ascontiguousarray(queries, dtype=np.float32)
        scores, idx = self._index.search(q, k)
        return cast(FloatArray, scores.astype(np.float32, copy=False)), idx.astype(np.int64)

    @property
    def index_bytes(self) -> int:
        if self._index is None:
            return 0
        try:
            import faiss

            return int(faiss.serialize_index(self._index).size)
        except Exception:  # pragma: no cover - best-effort accounting
            return self._fallback_bytes


@dataclass(slots=True)
class FaissHNSWIndex:
    """FAISS HNSW graph index (``IndexHNSWFlat``) with inner-product metric.

    HNSW is the dominant graph-based ANN structure in modern vector
    databases (pgvector ``hnsw``, Qdrant ``hnsw``, Milvus ``HNSW``,
    Weaviate, Vespa). It typically pareto-dominates IVF on
    recall-vs-latency at the cost of build time and ~2x storage overhead
    for the graph adjacency lists.

    Parameters
    ----------
    m:
        Number of bidirectional links per node. Higher = better recall,
        more memory. FAISS default is 32 which we keep here.
    ef_construction:
        Beam width during graph build. Higher = better graph quality,
        slower build. We use 200 (a common balanced default).
    ef_search:
        Beam width during query. Higher = better recall, higher latency.
        Tunable at query time without rebuilding.
    """

    m: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    name: str = "faiss-hnsw"
    _index: Any = None
    _dim: int = 0
    _fallback_bytes: int = 0

    def build(self, vectors: FloatArray) -> None:
        try:
            import faiss
        except ModuleNotFoundError as exc:  # pragma: no cover - optional path
            raise ModuleNotFoundError(
                "FAISS HNSW backend requires faiss. "
                "Install faiss-cpu (or faiss-gpu)."
            ) from exc
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        d = int(arr.shape[1])
        self._dim = d
        self._fallback_bytes = int(arr.nbytes)
        index = faiss.IndexHNSWFlat(d, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add(arr)
        self._index = index

    def search(self, queries: FloatArray, *, k: int) -> tuple[FloatArray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index has not been built")
        q = np.ascontiguousarray(queries, dtype=np.float32)
        scores, idx = self._index.search(q, k)
        return cast(FloatArray, scores.astype(np.float32, copy=False)), idx.astype(np.int64)

    @property
    def index_bytes(self) -> int:
        if self._index is None:
            return 0
        try:
            import faiss

            return int(faiss.serialize_index(self._index).size)
        except Exception:  # pragma: no cover - best-effort accounting
            return self._fallback_bytes


@dataclass(slots=True)
class FaissOPQIndex:
    """FAISS Optimized Product Quantization (``IndexPreTransform`` over OPQ + PQ).

    OPQ~\\cite{ge2014opq} pre-rotates the vector space with a learned
    orthogonal matrix that minimizes residual energy after PQ encoding;
    it routinely matches or beats vanilla PQ at the same byte budget
    and is the standard production baseline (FAISS ``OPQ``, ScaNN,
    Milvus ``IVF_PQ`` with rotation). Storage is the same one byte per
    subspace as :class:`FaissIVFPQIndex`.

    Parameters
    ----------
    m:
        Number of PQ subspaces. ``dim`` must be divisible by ``m``.
    n_bits:
        Bits per PQ code; ``2 ** n_bits`` centroids per subspace.
    auto_degrade_n_bits:
        If True, fall back to a smaller ``n_bits`` when the corpus is
        too small to train ``2 ** n_bits`` centroids. Mirrors
        :class:`FaissIVFPQIndex` so smoke runs do not abort.
    """

    m: int = 8
    n_bits: int = 8
    auto_degrade_n_bits: bool = True
    name: str = "faiss-opq"
    _index: Any = None
    _dim: int = 0
    _effective_n_bits: int = 0
    _fallback_bytes: int = 0

    def build(self, vectors: FloatArray) -> None:
        try:
            import faiss
        except ModuleNotFoundError as exc:  # pragma: no cover - optional path
            raise ModuleNotFoundError(
                "FAISS OPQ backend requires faiss. "
                "Install faiss-cpu (or faiss-gpu)."
            ) from exc
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = int(arr.shape[0]), int(arr.shape[1])
        if d % self.m != 0:
            raise ValueError(f"OPQ requires dim ({d}) divisible by m ({self.m})")
        if self.n_bits < 1 or self.n_bits > 8:
            raise ValueError("OPQ n_bits must be between 1 and 8")
        effective_n_bits = self.n_bits
        if self.auto_degrade_n_bits:
            while effective_n_bits > 1 and n < (1 << effective_n_bits):
                effective_n_bits -= 1
        if n < (1 << effective_n_bits):
            raise ValueError(
                f"OPQ needs at least {1 << effective_n_bits} training "
                f"vectors at n_bits={effective_n_bits}, got {n}"
            )
        # FAISS's OPQMatrix internal training uses 256 centroids per subspace
        # regardless of the outer factory n_bits, so OPQ has a hard floor of
        # 256 training vectors. Surface this clearly rather than letting
        # FAISS raise a cryptic Clustering error.
        if n < 256:
            raise ValueError(
                f"FaissOPQIndex requires at least 256 training vectors for "
                f"the OPQ rotation; got {n}. Use FaissIVFPQIndex for smaller "
                f"corpora (it supports auto-degrade)."
            )
        self._dim = d
        self._effective_n_bits = effective_n_bits
        self._fallback_bytes = int(arr.nbytes)
        # FAISS factory string keeps the OPQ rotation and the outer PQ
        # at the same n_bits.
        factory = f"OPQ{self.m}_{d},PQ{self.m}x{effective_n_bits}"
        index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
        index.train(arr)
        index.add(arr)
        self._index = index

    def search(self, queries: FloatArray, *, k: int) -> tuple[FloatArray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index has not been built")
        q = np.ascontiguousarray(queries, dtype=np.float32)
        scores, idx = self._index.search(q, k)
        return cast(FloatArray, scores.astype(np.float32, copy=False)), idx.astype(np.int64)

    @property
    def index_bytes(self) -> int:
        if self._index is None:
            return 0
        try:
            import faiss

            return int(faiss.serialize_index(self._index).size)
        except Exception:  # pragma: no cover - best-effort accounting
            return self._fallback_bytes


@dataclass(slots=True)
class FaissFlatIndex:
    """FAISS exact flat inner-product index.

    The class imports FAISS lazily so the package remains usable without the
    optional dependency. Install ``faiss-cpu`` or ``faiss-gpu`` to enable it.
    """

    name: str = "faiss-flat"
    _index: Any = None
    _dim: int = 0
    _fallback_bytes: int = 0

    def build(self, vectors: FloatArray) -> None:
        try:
            import faiss
        except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime path
            raise ModuleNotFoundError(
                "FAISS backend requested but faiss is not installed. "
                "Install faiss-cpu or faiss-gpu, or use --index-backends exact-numpy."
            ) from exc
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        self._dim = int(arr.shape[1])
        self._fallback_bytes = int(arr.nbytes)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(arr)

    def search(self, queries: FloatArray, *, k: int) -> tuple[FloatArray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index has not been built")
        q = np.ascontiguousarray(queries, dtype=np.float32)
        scores, idx = self._index.search(q, k)
        return cast(FloatArray, scores.astype(np.float32, copy=False)), idx.astype(np.int64)

    @property
    def index_bytes(self) -> int:
        if self._index is None:
            return 0
        try:
            import faiss

            return int(faiss.serialize_index(self._index).size)
        except Exception:  # pragma: no cover - best-effort accounting
            return self._fallback_bytes


def _renormalize(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return cast(FloatArray, (vectors / norms).astype(np.float32, copy=False))


def dense_views_for_compressor(
    compressor: Compressor,
    queries: FloatArray,
    compressed: CompressedSet,
) -> tuple[FloatArray, FloatArray] | None:
    """Return dense query/corpus views for index backends, if possible.

    Identity and truncation already store dense float32 codes. Scalar
    quantization can be decoded through its implementation helper. Binary and
    product quantization stay in compressed-domain scoring and return ``None``.
    """
    codes = compressed.codes
    if isinstance(codes, np.ndarray) and codes.dtype == np.float32 and codes.ndim == 2:
        corpus = cast(FloatArray, codes)
    elif hasattr(compressor, "_decode") and isinstance(codes, np.ndarray):
        decoded = compressor._decode(codes)
        corpus = _renormalize(cast(FloatArray, decoded))
    else:
        return None

    if queries.shape[1] == corpus.shape[1]:
        query_view = queries.astype(np.float32, copy=False)
    elif queries.shape[1] > corpus.shape[1]:
        query_view = _renormalize(queries[:, : corpus.shape[1]])
    else:
        return None
    return query_view, corpus


def _metrics_from_topk(top_indices: np.ndarray, qrels: QrelMap) -> tuple[float, float]:
    recalls: list[float] = []
    ndcgs: list[float] = []
    for qid, qrel in qrels.items():
        qid_i = int(qid)
        if qid_i >= top_indices.shape[0]:
            continue
        relevant = {doc for doc, rel in qrel.items() if rel > 0}
        if not relevant:
            continue
        ranking = top_indices[qid_i]
        hit = sum(1 for doc in ranking if int(doc) in relevant)
        recalls.append(hit / len(relevant))

        gains = np.array(
            [(2.0 ** qrel.get(int(doc), 0.0)) - 1.0 for doc in ranking],
            dtype=np.float64,
        )
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
        dcg = float((gains * discounts).sum())
        ideal_rels = sorted(qrel.values(), reverse=True)[: ranking.size]
        ideal_gains = np.array([(2.0**rel) - 1.0 for rel in ideal_rels], dtype=np.float64)
        ideal_disc = 1.0 / np.log2(np.arange(2, ideal_gains.size + 2, dtype=np.float64))
        idcg = float((ideal_gains * ideal_disc).sum())
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    recall = float(np.mean(recalls)) if recalls else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    return recall, ndcg


def _exact_recall(top_indices: np.ndarray, reference_topk: np.ndarray | None) -> float | None:
    if reference_topk is None:
        return None
    recalls: list[float] = []
    for got, ref in zip(top_indices, reference_topk, strict=True):
        ref_set = {int(x) for x in ref if x >= 0}
        if not ref_set:
            continue
        got_set = {int(x) for x in got if x >= 0}
        recalls.append(len(got_set & ref_set) / len(ref_set))
    return float(np.mean(recalls)) if recalls else None


def _search_batched(
    index: DenseSearchIndex,
    queries: FloatArray,
    *,
    k: int,
    query_batch_size: int,
) -> tuple[FloatArray, np.ndarray]:
    if query_batch_size < 1:
        raise ValueError("query_batch_size must be >= 1")
    score_chunks: list[FloatArray] = []
    index_chunks: list[np.ndarray] = []
    for start in range(0, queries.shape[0], query_batch_size):
        scores, indices = index.search(queries[start : start + query_batch_size], k=k)
        score_chunks.append(scores)
        index_chunks.append(indices)
    if not score_chunks:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.int64),
        )
    return (
        cast(FloatArray, np.vstack(score_chunks).astype(np.float32, copy=False)),
        np.vstack(index_chunks).astype(np.int64, copy=False),
    )


KNOWN_BACKENDS: tuple[str, ...] = (
    "exact-numpy",
    "faiss-flat",
    "faiss-ivf",
    "faiss-ivfpq",
    "faiss-hnsw",
    "faiss-opq",
)


def make_dense_index(backend: str) -> DenseSearchIndex:
    """Instantiate a dense search backend by name.

    Recognized names are listed in :data:`KNOWN_BACKENDS`. The FAISS-backed
    variants approximate the canonical vector-database index types:

    * ``faiss-flat`` — exact inner-product baseline (``IndexFlatIP``);
      matches pgvector ``vector``, Milvus ``FLAT``, Qdrant ``flat``.
    * ``faiss-ivf``  — inverted file with k-means partitioning
      (``IndexIVFFlat``); matches Milvus ``IVF_FLAT`` and Vespa's IVF mode.
    * ``faiss-ivfpq`` — inverted file with PQ-compressed residuals
      (``IndexIVFPQ``); matches Milvus ``IVF_PQ`` and OpenSearch's
      ``faiss/ivf_pq`` engine.
    * ``faiss-hnsw`` — hierarchical navigable small-world graph
      (``IndexHNSWFlat``); matches pgvector ``hnsw``, Qdrant ``hnsw``,
      Milvus ``HNSW``.
    * ``faiss-opq`` — optimized product quantization (rotation +
      ``IndexPQ``); the standard high-quality PQ baseline.
    """
    if backend == "exact-numpy":
        return ExactNumpyIndex()
    if backend == "faiss-flat":
        return FaissFlatIndex()
    if backend == "faiss-ivf":
        return FaissIVFIndex()
    if backend == "faiss-ivfpq":
        return FaissIVFPQIndex()
    if backend == "faiss-hnsw":
        return FaissHNSWIndex()
    if backend == "faiss-opq":
        return FaissOPQIndex()
    raise ValueError(f"Unknown index backend {backend!r}")


def evaluate_dense_index(
    *,
    corpus_vectors: FloatArray,
    query_vectors: FloatArray,
    qrels: QrelMap,
    backend: str,
    k: int = 10,
    n_repeats: int = 10,
    n_warmup: int = 1,
    reference_topk: np.ndarray | None = None,
    query_batch_size: int = 32,
) -> DenseIndexEval:
    """Build an index, time search, and compute retrieval metrics."""
    index = make_dense_index(backend)
    build_t0 = time.perf_counter_ns()
    index.build(corpus_vectors)
    build_ms = (time.perf_counter_ns() - build_t0) / 1e6

    last_scores: FloatArray | None = None
    last_indices: np.ndarray | None = None

    def _search() -> None:
        nonlocal last_scores, last_indices
        last_scores, last_indices = _search_batched(
            index,
            query_vectors,
            k=k,
            query_batch_size=query_batch_size,
        )

    timing = time_callable(_search, n_repeats=n_repeats, n_warmup=n_warmup)
    assert last_scores is not None and last_indices is not None
    recall, ndcg = _metrics_from_topk(last_indices, qrels)
    return DenseIndexEval(
        backend=backend,
        build_ms=float(build_ms),
        search_latency_ms=timing.median_ms,
        search_p95_ms=timing.p95_ms,
        index_bytes=index.index_bytes,
        recall_at_10=recall,
        ndcg_at_10=ndcg,
        exact_recall_at_10=_exact_recall(last_indices, reference_topk),
    )
