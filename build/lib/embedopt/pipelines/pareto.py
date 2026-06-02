"""End-to-end multi-objective compression sweep.

Given a frozen embedder, a retrieval dataset, and a list of compressor specs,
this pipeline:

1. Embeds the corpus and queries once (saves ``len(specs)`` re-encodes).
2. For each spec, fits the compressor on the corpus, profiles it (bytes /
   latency), then evaluates retrieval quality against the same compressed
   corpus.
3. Filters the resulting candidates to the Pareto front under
   ``(quality, bytes_per_vector, query_latency_ms)`` and reports the
   hypervolume of the front against a fixed reference point.

The pipeline is the experimental backbone of the paper: every figure showing
the quality/size/latency frontier is a one-line call to :func:`run_pareto_sweep`
with a different embedder or search space.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from embedopt.compression.base import Compressor
from embedopt.compression.registry import build_compressor, spec_label
from embedopt.evaluation.datasets import RetrievalDataset
from embedopt.evaluation.metrics import (
    map_score,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from embedopt.models.backbones import TextEmbedder
from embedopt.moo.hypervolume import hypervolume
from embedopt.moo.objectives import Objective, to_min_matrix
from embedopt.moo.pareto import pareto_indices
from embedopt.profiling.aggregator import EfficiencyStats, profile_compressor
from embedopt.utils.types import FloatArray, as_float_array


@dataclass(slots=True)
class CompressionCandidate:
    """One ``(spec, metrics, efficiency)`` triple in the sweep."""

    spec: Mapping[str, Any]
    label: str
    metrics: Mapping[str, float]
    efficiency: EfficiencyStats

    @property
    def quality(self) -> float:
        """Headline quality metric (defaults to nDCG@10 if present)."""
        return float(self.metrics.get("ndcg_at_10", 0.0))


@dataclass(slots=True)
class ParetoSweepResult:
    """Output of :func:`run_pareto_sweep`."""

    candidates: list[CompressionCandidate]
    pareto_idx: list[int]
    hypervolume: float
    objectives: list[Objective] = field(default_factory=list)


def _evaluate_quality(
    compressor: Compressor,
    queries: FloatArray,
    compressed_corpus: Any,
    qrels: Mapping[int, Mapping[int, float]],
    *,
    k: int,
) -> dict[str, float]:
    scores = compressor.score(queries, compressed_corpus)
    return {
        "recall_at_10": recall_at_k(scores, qrels, k=k),
        "mrr_at_10": mrr_at_k(scores, qrels, k=k),
        "ndcg_at_10": ndcg_at_k(scores, qrels, k=k),
        "map": map_score(scores, qrels, k=k),
    }


def default_search_space(dim: int) -> list[Mapping[str, Any]]:
    """A reasonable default search space, parameterized by backbone dim ``d``.

    Includes one identity baseline, four Matryoshka truncation levels (limited
    to actual dim), an int8 scalar codec, a binary codec, and product
    quantization at two ``M`` values when ``d`` is divisible by both.
    """
    truncate_dims = [d for d in (32, 64, 128, 256) if d <= dim]
    pq_options: list[Mapping[str, Any]] = []
    for m in (8, 16):
        if dim % m == 0:
            pq_options.append({"name": "product_quantize", "n_subspaces": m, "n_bits": 8})
    space: list[Mapping[str, Any]] = [{"name": "identity"}]
    for kd in truncate_dims:
        space.append({"name": "truncate", "keep_dim": kd})
    space.append({"name": "scalar_int8"})
    space.append({"name": "binary"})
    space.extend(pq_options)
    return space


def run_pareto_sweep(
    embedder: TextEmbedder,
    dataset: RetrievalDataset,
    specs: Sequence[Mapping[str, Any]] | None = None,
    *,
    k: int = 10,
    profile_repeats: int = 15,
    profile_warmup: int = 2,
) -> ParetoSweepResult:
    """Run a full multi-objective sweep over ``specs`` and return the Pareto result."""
    corpus = as_float_array(embedder.encode(list(dataset.corpus)))
    queries = as_float_array(embedder.encode(list(dataset.queries)))
    spec_list = list(specs) if specs is not None else default_search_space(corpus.shape[1])

    candidates: list[CompressionCandidate] = []
    for spec in spec_list:
        compressor = build_compressor(spec)
        stats, compressed = profile_compressor(
            compressor,
            corpus,
            queries,
            n_repeats=profile_repeats,
            n_warmup=profile_warmup,
        )
        metrics = _evaluate_quality(compressor, queries, compressed, dataset.qrels, k=k)
        candidates.append(
            CompressionCandidate(
                spec=spec,
                label=spec_label(spec),
                metrics=metrics,
                efficiency=stats,
            )
        )

    objectives = [
        Objective(name="ndcg_at_10", sense="max"),
        Objective(name="bytes_per_vector", sense="min"),
        Objective(name="query_latency_ms", sense="min"),
    ]
    rows = [
        {
            "ndcg_at_10": c.metrics["ndcg_at_10"],
            "bytes_per_vector": float(c.efficiency.bytes_per_vector),
            "query_latency_ms": c.efficiency.query_latency_ms,
        }
        for c in candidates
    ]
    pts = to_min_matrix(objectives, rows)
    pareto = pareto_indices(pts)
    # Reference for HV: dominate every point by a small margin in canonical min form.
    if pts.size:
        ref = pts.max(axis=0) + np.abs(pts.max(axis=0)) * 0.05 + 1e-6
    else:
        ref = np.array([0.0, 0.0, 0.0])
    hv = hypervolume(pts[pareto], ref) if pareto else 0.0

    return ParetoSweepResult(
        candidates=candidates,
        pareto_idx=pareto,
        hypervolume=hv,
        objectives=objectives,
    )
