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
        relevant = {doc for doc, rel in qrel.items() if rel > 0}
        if not relevant:
            continue
        ranking = top_indices[int(qid)]
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


def make_dense_index(backend: str) -> DenseSearchIndex:
    """Instantiate a dense search backend by name."""
    if backend == "exact-numpy":
        return ExactNumpyIndex()
    if backend == "faiss-flat":
        return FaissFlatIndex()
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
        last_scores, last_indices = index.search(query_vectors, k=k)

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
