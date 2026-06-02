"""Scalar quantization (per-dimension affine to ``int8``).

For each coordinate ``j`` we fit ``min_j``, ``max_j`` on a training sample and
encode ``q_j(v) = round((v_j - min_j) / (max_j - min_j) * 255) - 128``. Decode
inverts the transform; scoring dequantizes the corpus once per query batch and
falls back to dense matmul.

This is the simplest non-trivial codec and a key Pareto baseline: it
near-quarters the memory of ``float32`` with negligible quality loss for many
sentence encoders, and serves as the reference point above which more
aggressive codecs (binary, PQ) must justify their additional error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from embedopt.compression.base import CompressedSet
from embedopt.utils.types import FloatArray


@dataclass(slots=True)
class ScalarQuantizeCompressor:
    """Per-dimension affine quantization to ``int8``."""

    name: str = "scalar_int8"
    _scale: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _offset: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _trained: bool = False

    @property
    def trained(self) -> bool:
        return self._trained

    def fit(self, vectors: FloatArray) -> None:
        if vectors.size == 0:
            raise ValueError("Need a non-empty training sample")
        v_min = vectors.min(axis=0).astype(np.float32)
        v_max = vectors.max(axis=0).astype(np.float32)
        span = (v_max - v_min).astype(np.float32)
        span = np.where(span == 0.0, np.float32(1.0), span)
        self._scale = (span / np.float32(255.0)).astype(np.float32, copy=False)
        self._offset = v_min
        self._trained = True

    def transform(self, vectors: FloatArray) -> CompressedSet:
        if not self._trained:
            self.fit(vectors)
        normalized = (vectors - self._offset) / self._scale
        clipped = np.clip(normalized, 0.0, 255.0)
        codes = (np.round(clipped) - 128).astype(np.int8)
        bpv = int(codes.shape[1])
        return CompressedSet(codes=codes, bytes_per_vector=bpv)

    def _decode(self, codes: np.ndarray[Any, Any]) -> FloatArray:
        as_float = codes.astype(np.float32) + 128.0
        out: FloatArray = (as_float * self._scale + self._offset).astype(np.float32, copy=False)
        return out

    def score(self, queries: FloatArray, corpus: CompressedSet) -> FloatArray:
        codes = cast(np.ndarray[Any, Any], corpus.codes)
        decoded = self._decode(codes)
        norms = np.linalg.norm(decoded, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        decoded_norm = decoded / norms
        sims: FloatArray = (queries @ decoded_norm.T).astype(np.float32, copy=False)
        return sims

    def bytes_per_vector(self, dim: int) -> int:
        return int(dim)
