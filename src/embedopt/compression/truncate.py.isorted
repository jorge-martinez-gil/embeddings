"""Matryoshka-style dimensionality truncation.

Keeps the first ``keep_dim`` coordinates of every vector and re-normalizes.
For Matryoshka-trained backbones this is loss-aware; for arbitrary backbones
it is a strong PCA-free baseline that stays competitive when the leading
coordinates carry most of the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from embedopt.compression.base import CompressedSet
from embedopt.utils.types import FloatArray


@dataclass(slots=True)
class TruncateCompressor:
    """Keep the first ``keep_dim`` coordinates and re-normalize."""

    keep_dim: int
    name: str = "truncate"

    @property
    def trained(self) -> bool:
        return True

    def fit(self, vectors: FloatArray) -> None:  # noqa: D401 - protocol no-op
        return None

    def transform(self, vectors: FloatArray) -> CompressedSet:
        if self.keep_dim <= 0:
            raise ValueError("keep_dim must be positive")
        if self.keep_dim > vectors.shape[1]:
            raise ValueError(f"keep_dim={self.keep_dim} exceeds embedding dim {vectors.shape[1]}")
        sub = vectors[:, : self.keep_dim]
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        codes = (sub / norms).astype(np.float32, copy=False)
        bpv = int(self.keep_dim * 4)
        return CompressedSet(codes=codes, bytes_per_vector=bpv)

    def score(self, queries: FloatArray, corpus: CompressedSet) -> FloatArray:
        codes = cast(FloatArray, corpus.codes)
        keep = codes.shape[1]
        q = queries[:, :keep]
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        q_norm = q / norms
        return cast(FloatArray, (q_norm @ codes.T).astype(np.float32, copy=False))

    def bytes_per_vector(self, dim: int) -> int:
        return int(min(self.keep_dim, dim) * 4)
