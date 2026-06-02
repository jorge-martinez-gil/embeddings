from __future__ import annotations

import numpy as np

from embedopt.compression import (
    BinaryQuantizeCompressor,
    ComposedCompressor,
    ProductQuantizeCompressor,
    TruncateCompressor,
    build_compressor,
)


def _toy_corpus(seed: int = 0, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((40, dim)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


def test_truncate_then_pq_byte_cost_and_self_neighbor() -> None:
    corpus = _toy_corpus(dim=32)
    chain = ComposedCompressor(
        stages=[
            TruncateCompressor(keep_dim=16),
            ProductQuantizeCompressor(n_subspaces=8, n_bits=4, seed=1),
        ]
    )
    chain.fit(corpus)
    cs = chain.transform(corpus)
    assert cs.bytes_per_vector == 8  # PQ: one byte per subspace
    sims = chain.score(corpus, cs)
    self_top = (np.argmax(sims, axis=1) == np.arange(corpus.shape[0])).mean()
    assert self_top > 0.6


def test_truncate_then_binary_byte_cost() -> None:
    corpus = _toy_corpus(dim=32)
    chain = ComposedCompressor(stages=[TruncateCompressor(keep_dim=16), BinaryQuantizeCompressor()])
    chain.fit(corpus)
    cs = chain.transform(corpus)
    assert cs.bytes_per_vector == 16 // 8


def test_registry_builds_composed_spec() -> None:
    spec = {
        "name": "composed",
        "stages": [
            {"name": "truncate", "keep_dim": 16},
            {"name": "binary"},
        ],
    }
    c = build_compressor(spec)
    assert c.name == "composed"
