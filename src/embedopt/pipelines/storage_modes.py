"""Direct comparison across the common vector-database storage modes.

Every production vector store (pgvector, Milvus, Qdrant, Weaviate, FAISS,
Pinecone, Elasticsearch dense_vector, OpenSearch k-NN, Vespa) exposes
roughly the same shortlist of on-disk / in-memory layouts for dense
embeddings:

* ``float32`` — the reference, full precision. 4 bytes / dim.
* ``float16`` — IEEE-754 half precision (``halfvec`` in pgvector,
  ``Float16`` in Qdrant, ``FLOAT16`` in Milvus). 2 bytes / dim.
* ``int8``   — affine scalar quantization (``ScalarQuantizer`` in FAISS,
  Milvus ``SQ8``, Qdrant ``Int8``). 1 byte / dim.
* ``binary`` — sign-bit codes with Hamming scoring (FAISS ``IndexBinary``,
  Milvus ``BIN_*``, Qdrant ``Binary``). 1 bit / dim ≈ 0.125 byte / dim.
* ``PQ``     — product quantization with ADC scoring (FAISS ``IndexPQ``,
  Milvus ``IVF_PQ``, ScaNN ``PQ``). ``n_subspaces`` bytes / vector at
  ``n_bits=8``, independent of dim.

This module runs all five side-by-side on the same embedded corpus and
returns a table of (storage mode, bytes / vector, nDCG@10, Recall@10,
MRR@10, median latency, p95 latency, compression ratio, ΔnDCG vs.
``float32``) — i.e. the comparison a vector-DB practitioner needs to pick
a default for their index.

The intent is *not* to win the Pareto front (that is what
:mod:`embedopt.pipelines.pareto` does); it is to be the apples-to-apples
ablation table that goes into the paper, the README, and any blog post
explaining "should I move from float32 to halfvec or jump straight to
PQ?".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from embedopt.compression.registry import build_compressor
from embedopt.evaluation.datasets import RetrievalDataset
from embedopt.evaluation.metrics import (
    map_score,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from embedopt.models.backbones import TextEmbedder
from embedopt.profiling.aggregator import EfficiencyStats, profile_compressor
from embedopt.utils.types import as_float_array


@dataclass(slots=True, frozen=True)
class StorageModeRow:
    """One row in the storage-mode comparison table."""

    mode: str
    """Short name (``float32``, ``float16``, ``int8``, ``binary``, ``pq``)."""

    spec: Mapping[str, Any]
    """Compressor spec dict the row was built from."""

    bytes_per_vector: int
    """Bytes occupied by one compressed vector."""

    compression_ratio: float
    """``float32_bytes / bytes_per_vector``. Higher = more compressed."""

    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    map_at_10: float

    delta_ndcg_vs_fp32: float
    """``ndcg_at_10 - fp32_baseline_ndcg`` (negative means quality loss)."""

    fit_ms: float
    encode_ms: float
    query_latency_ms: float
    query_p95_ms: float

    def as_dict(self) -> dict[str, Any]:
        """Flat dict suitable for CSV / JSON serialization."""
        out = asdict(self)
        out["spec"] = dict(self.spec)
        return out


@dataclass(slots=True, frozen=True)
class StorageModeComparison:
    """Output of :func:`compare_storage_modes`."""

    rows: list[StorageModeRow]
    baseline_mode: str = "float32"
    """The mode whose ``ndcg_at_10`` is the reference for ``delta_ndcg_vs_fp32``."""

    objectives: list[str] = field(
        default_factory=lambda: [
            "bytes_per_vector",
            "ndcg_at_10",
            "query_latency_ms",
        ]
    )

    def by_mode(self) -> dict[str, StorageModeRow]:
        return {r.mode: r for r in self.rows}


def default_storage_mode_specs(dim: int) -> list[tuple[str, Mapping[str, Any]]]:
    """Return the canonical ``(mode_label, spec)`` shortlist for a given dim.

    PQ is configured to roughly mirror typical vector-DB defaults: 8-byte
    codes (``M=8``), with both ``n_bits=8`` (the FAISS / Milvus default)
    and ``n_bits=4`` (the popular "PQ4" point that halves PQ storage again
    and is what reviewers always ask about). If ``dim`` is not divisible
    by 8 we fall back to the largest ``M`` in ``{8, 4, 2}`` that divides
    it; if none do, the PQ rows are omitted entirely.
    """
    modes: list[tuple[str, Mapping[str, Any]]] = [
        ("float32", {"name": "identity"}),
        ("float16", {"name": "float16"}),
        ("int8", {"name": "scalar_int8"}),
        ("binary", {"name": "binary"}),
    ]
    pq_m = next((m for m in (8, 4, 2) if dim % m == 0), None)
    if pq_m is not None:
        # 8-bit PQ: typical vector-DB default (one byte per subspace).
        modes.append(
            (
                f"pq(M={pq_m},nbits=8)",
                {"name": "product_quantize", "n_subspaces": pq_m, "n_bits": 8},
            )
        )
        # 4-bit PQ: aggressive ablation point. Byte budget reported to the
        # optimizer remains one ``uint8`` per subspace (the compressor does
        # not nibble-pack), so this row trades quality for centroid-table
        # size and codebook training time rather than for raw bytes / vec.
        modes.append(
            (
                f"pq(M={pq_m},nbits=4)",
                {"name": "product_quantize", "n_subspaces": pq_m, "n_bits": 4},
            )
        )
    return modes


def compare_storage_modes(
    embedder: TextEmbedder,
    dataset: RetrievalDataset,
    modes: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
    *,
    k: int = 10,
    profile_repeats: int = 15,
    profile_warmup: int = 2,
) -> StorageModeComparison:
    """Run every storage-mode spec on the same embedded corpus and compare.

    The corpus / queries are embedded once and re-used across all modes,
    so the comparison reports *only* the cost of changing the storage
    layout — not backbone or dataset variance.
    """
    corpus = as_float_array(embedder.encode(list(dataset.corpus)))
    queries = as_float_array(embedder.encode(list(dataset.queries)))
    if modes is None:
        modes = default_storage_mode_specs(corpus.shape[1])

    qrels = dataset.qrels
    rows: list[StorageModeRow] = []
    baseline_ndcg: float | None = None
    fp32_bpv: int | None = None
    for mode_name, spec in modes:
        compressor = build_compressor(spec)
        stats: EfficiencyStats
        stats, compressed = profile_compressor(
            compressor,
            corpus,
            queries,
            n_repeats=profile_repeats,
            n_warmup=profile_warmup,
        )
        scores = compressor.score(queries, compressed)
        ndcg = ndcg_at_k(scores, qrels, k=k)
        if mode_name == "float32":
            baseline_ndcg = ndcg
            fp32_bpv = stats.bytes_per_vector
        delta = 0.0 if baseline_ndcg is None else ndcg - baseline_ndcg
        ratio = (
            float(fp32_bpv) / float(stats.bytes_per_vector)
            if fp32_bpv and stats.bytes_per_vector
            else 1.0
        )
        rows.append(
            StorageModeRow(
                mode=mode_name,
                spec=dict(spec),
                bytes_per_vector=stats.bytes_per_vector,
                compression_ratio=ratio,
                ndcg_at_10=ndcg,
                recall_at_10=recall_at_k(scores, qrels, k=k),
                mrr_at_10=mrr_at_k(scores, qrels, k=k),
                map_at_10=map_score(scores, qrels, k=k),
                delta_ndcg_vs_fp32=delta,
                fit_ms=stats.fit_ms,
                encode_ms=stats.encode_ms,
                query_latency_ms=stats.query_latency_ms,
                query_p95_ms=stats.query_p95_ms,
            )
        )
    return StorageModeComparison(rows=rows)


def format_comparison_table(comparison: StorageModeComparison) -> str:
    """Render a human-readable comparison table (monospace, fixed columns)."""
    header = (
        f"{'mode':<18s}  {'B/vec':>6s}  {'x f32':>6s}  "
        f"{'nDCG@10':>8s}  {'ΔnDCG':>7s}  {'Rec@10':>7s}  "
        f"{'MRR@10':>7s}  {'lat ms':>7s}  {'p95 ms':>7s}"
    )
    lines = [header, "-" * len(header)]
    for r in comparison.rows:
        lines.append(
            f"{r.mode:<18s}  {r.bytes_per_vector:>6d}  {r.compression_ratio:>6.2f}  "
            f"{r.ndcg_at_10:>8.4f}  {r.delta_ndcg_vs_fp32:>+7.4f}  "
            f"{r.recall_at_10:>7.4f}  {r.mrr_at_10:>7.4f}  "
            f"{r.query_latency_ms:>7.3f}  {r.query_p95_ms:>7.3f}"
        )
    return "\n".join(lines)
