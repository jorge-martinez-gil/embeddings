from __future__ import annotations

import numpy as np

from embedopt.models.backbones import HashingEmbedder, RandomProjectionEmbedder


def test_hashing_embedder_is_deterministic() -> None:
    enc = HashingEmbedder(dim_=64)
    a = enc.encode(["hello world", "foo bar"])
    b = enc.encode(["hello world", "foo bar"])
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_hashing_embedder_outputs_unit_norm() -> None:
    enc = HashingEmbedder(dim_=128)
    out = np.asarray(enc.encode(["the quick brown fox jumps"]))
    norm = float(np.linalg.norm(out[0]))
    assert abs(norm - 1.0) < 1e-5


def test_hashing_embedder_separates_distinct_inputs() -> None:
    enc = HashingEmbedder(dim_=256)
    out = np.asarray(enc.encode(["machine learning", "garlic and onion"]))
    # cosine should be much less than 1 for unrelated topics
    cos = float(out[0] @ out[1])
    assert cos < 0.5


def test_random_projection_dim() -> None:
    enc = RandomProjectionEmbedder(dim_=32, hash_dim=128, seed=1)
    out = np.asarray(enc.encode(["hello"]))
    assert out.shape == (1, 32)
    assert abs(float(np.linalg.norm(out[0])) - 1.0) < 1e-5
