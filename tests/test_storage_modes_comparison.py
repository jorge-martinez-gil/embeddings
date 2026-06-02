"""Smoke tests for the direct storage-mode comparison pipeline."""

from __future__ import annotations

import numpy as np

from embedopt.evaluation.datasets import make_synthetic_retrieval
from embedopt.models.backbones import HashingEmbedder
from embedopt.pipelines.storage_modes import (
    compare_storage_modes,
    default_storage_mode_specs,
    format_comparison_table,
)


def test_default_storage_mode_specs_shape() -> None:
    modes = default_storage_mode_specs(dim=128)
    names = [m for m, _ in modes]
    # Must cover the five storage modes the comparison advertises plus a
    # PQ-bits ablation row (4-bit) so reviewers can see the PQ4 point.
    assert names[:4] == ["float32", "float16", "int8", "binary"]
    pq_names = [n for n in names if n.startswith("pq(")]
    assert len(pq_names) == 2
    pq_specs = [s for n, s in modes if n.startswith("pq(")]
    assert all(s["name"] == "product_quantize" for s in pq_specs)
    bit_widths = sorted(int(s["n_bits"]) for s in pq_specs)
    assert bit_widths == [4, 8]


def test_default_storage_mode_specs_omits_pq_on_indivisible_dim() -> None:
    # 7 is divisible by none of {8, 4, 2}; PQ must be omitted entirely.
    modes = default_storage_mode_specs(dim=7)
    names = [m for m, _ in modes]
    assert all(not n.startswith("pq") for n in names)


def test_compare_storage_modes_runs_end_to_end_offline() -> None:
    embedder = HashingEmbedder(dim_=64)
    dataset = make_synthetic_retrieval(n_queries_per_topic=2, seed=0)
    comparison = compare_storage_modes(
        embedder,
        dataset,
        profile_repeats=3,
        profile_warmup=1,
    )
    by_mode = comparison.by_mode()
    # All five canonical modes plus the PQ4 ablation row are present.
    assert set(by_mode) >= {"float32", "float16", "int8", "binary"}
    pq_modes = [name for name in by_mode if name.startswith("pq(")]
    assert len(pq_modes) == 2

    f32 = by_mode["float32"]
    f16 = by_mode["float16"]
    int8 = by_mode["int8"]
    binary = by_mode["binary"]

    # Byte budgets match the textbook story.
    assert f32.bytes_per_vector == 64 * 4
    assert f16.bytes_per_vector == 64 * 2
    assert int8.bytes_per_vector == 64
    assert binary.bytes_per_vector == 64 // 8

    # Compression ratios are reported relative to float32.
    assert f32.compression_ratio == 1.0
    assert f16.compression_ratio == 2.0
    assert int8.compression_ratio == 4.0
    assert binary.compression_ratio == 32.0

    # float32 is the baseline, so its delta is exactly zero.
    assert f32.delta_ndcg_vs_fp32 == 0.0
    # float16 must be essentially lossless on cosine-normalized vectors.
    assert abs(f16.delta_ndcg_vs_fp32) < 1e-3

    # All metrics are finite and within their valid ranges.
    for row in comparison.rows:
        assert 0.0 <= row.ndcg_at_10 <= 1.0
        assert 0.0 <= row.recall_at_10 <= 1.0
        assert 0.0 <= row.mrr_at_10 <= 1.0
        assert row.query_latency_ms >= 0.0
        assert row.query_p95_ms >= row.query_latency_ms - 1e-6
        assert np.isfinite(row.compression_ratio)


def test_format_comparison_table_includes_every_mode() -> None:
    embedder = HashingEmbedder(dim_=64)
    dataset = make_synthetic_retrieval(n_queries_per_topic=2, seed=1)
    comparison = compare_storage_modes(embedder, dataset, profile_repeats=2, profile_warmup=1)
    text = format_comparison_table(comparison)
    for row in comparison.rows:
        assert row.mode in text
    # Sanity: header + separator + one line per row.
    assert text.count("\n") >= len(comparison.rows) + 1
