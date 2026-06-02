"""Multi-seed variance evaluation for codebook-trained compressors.

PQ (and OPQ) are not deterministic for a fixed corpus: k-means
initialization, alternating-SVD seeding, and codeword tie-breaking all
respond to the RNG seed. A single-seed nDCG number therefore mixes
algorithmic effect with codebook-fitting noise.

This module exposes one helper, :func:`evaluate_compressor_seed_variance`,
which evaluates the same compressor spec at ``len(seeds)`` independent
seeds and returns one :class:`SeedVarianceRow` per (spec, seed)
together with an aggregated mean/std summary. The result feeds the
seed-variance CSV emitted by ``scripts/run_seed_variance.py`` and the
variance-bar figure rendered by ``scripts/plot_paper_figures.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from embedopt.compression.registry import build_compressor, spec_label
from embedopt.evaluation.datasets import RetrievalDataset
from embedopt.evaluation.metrics import (
    map_score,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from embedopt.models.backbones import TextEmbedder
from embedopt.utils.types import FloatArray, as_float_array


@dataclass(slots=True, frozen=True)
class SeedVarianceRow:
    """One (spec, seed) measurement."""

    spec_label: str
    seed: int
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    map_at_10: float


@dataclass(slots=True, frozen=True)
class SeedVarianceSummary:
    """Aggregated mean/std across seeds for a single spec."""

    spec_label: str
    n_seeds: int
    ndcg_mean: float
    ndcg_std: float
    recall_mean: float
    recall_std: float
    mrr_mean: float
    mrr_std: float
    map_mean: float
    map_std: float
    rows: list[SeedVarianceRow] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["rows"] = [asdict(r) for r in self.rows]
        return out


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


def _override_seed(spec: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Return a copy of ``spec`` with the seed kwarg overridden.

    Composed specs forward the seed override to every sub-stage that
    accepts a seed (PQ and OPQ both do).
    """
    new: dict[str, Any] = dict(spec)
    if spec.get("name") == "composed":
        new["stages"] = [_override_seed(stage, seed) for stage in spec["stages"]]
    elif spec.get("name") in {"product_quantize", "opq"}:
        new["seed"] = int(seed)
    return new


def evaluate_compressor_seed_variance(
    spec: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    embedder: TextEmbedder | None = None,
    dataset: RetrievalDataset | None = None,
    corpus: FloatArray | None = None,
    queries: FloatArray | None = None,
    qrels: Mapping[int, Mapping[int, float]] | None = None,
    k: int = 10,
) -> SeedVarianceSummary:
    """Evaluate ``spec`` at every seed in ``seeds`` and return mean/std.

    Either pass ``embedder`` + ``dataset`` (the embedder is invoked once
    and the result reused across seeds) or pre-embedded ``corpus`` /
    ``queries`` / ``qrels``.
    """
    if corpus is None or queries is None or qrels is None:
        if embedder is None or dataset is None:
            raise ValueError(
                "Pass either embedder+dataset or corpus/queries/qrels"
            )
        corpus = as_float_array(embedder.encode(list(dataset.corpus)))
        queries = as_float_array(embedder.encode(list(dataset.queries)))
        qrels = dataset.qrels

    rows: list[SeedVarianceRow] = []
    label = spec_label(spec)
    for s in seeds:
        seeded_spec = _override_seed(spec, int(s))
        compressor = build_compressor(seeded_spec)
        if not compressor.trained:
            compressor.fit(corpus)
        compressed = compressor.transform(corpus)
        scores = compressor.score(queries, compressed)
        rows.append(
            SeedVarianceRow(
                spec_label=label,
                seed=int(s),
                ndcg_at_10=ndcg_at_k(scores, qrels, k=k),
                recall_at_10=recall_at_k(scores, qrels, k=k),
                mrr_at_10=mrr_at_k(scores, qrels, k=k),
                map_at_10=map_score(scores, qrels, k=k),
            )
        )

    ndcg_m, ndcg_s = _mean_std([r.ndcg_at_10 for r in rows])
    recall_m, recall_s = _mean_std([r.recall_at_10 for r in rows])
    mrr_m, mrr_s = _mean_std([r.mrr_at_10 for r in rows])
    map_m, map_s = _mean_std([r.map_at_10 for r in rows])
    return SeedVarianceSummary(
        spec_label=label,
        n_seeds=len(rows),
        ndcg_mean=ndcg_m,
        ndcg_std=ndcg_s,
        recall_mean=recall_m,
        recall_std=recall_s,
        mrr_mean=mrr_m,
        mrr_std=mrr_s,
        map_mean=map_m,
        map_std=map_s,
        rows=rows,
    )


def default_pq_specs(dim: int) -> list[Mapping[str, Any]]:
    """Canonical PQ + OPQ grid for the seed-variance ablation.

    Only includes ``(M, n_bits)`` cells for which ``M`` divides
    ``dim`` and ``n_bits`` is supported by both PQ and OPQ.
    """
    specs: list[Mapping[str, Any]] = []
    for m in (4, 8, 16, 32, 64):
        if dim % m != 0:
            continue
        for b in (4, 6, 8):
            specs.append({"name": "product_quantize", "n_subspaces": m, "n_bits": b})
        # OPQ only at b in {4, 8} to keep the grid small; FAISS supports more.
        for b in (4, 8):
            specs.append({"name": "opq", "n_subspaces": m, "n_bits": b})
    return specs
