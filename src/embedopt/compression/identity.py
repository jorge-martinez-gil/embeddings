"""Identity (no-op) compressor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from embedopt.compression.base import CompressedSet
from embedopt.utils.types import FloatArray


@dataclass(slots=True)
class IdentityCompressor:
    """Pass-through baseline. Stores raw ``float32`` vectors."""

    name: str = "identity"

    @property
    def trained(self) -> bool:
        return True

    def fit(self, vectors: FloatArray) -> None:  # noqa: D401 - protocol no-op
        return None

    def transform(self, vectors: FloatArray) -> CompressedSet:
        codes = np.ascontiguousarray(vectors, dtype=np.float32)
        bpv = int(codes.shape[1] * 4)
        return CompressedSet(codes=codes, bytes_per_vector=bpv)

    def score(self, queries: FloatArray, corpus: CompressedSet) -> FloatArray:
        codes = cast(FloatArray, corpus.codes)
        return cast(FloatArray, (queries @ codes.T).astype(np.float32, copy=False))

    def bytes_per_vector(self, dim: int) -> int:
        return int(dim * 4)
