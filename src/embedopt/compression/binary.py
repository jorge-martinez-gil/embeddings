"""Binary (sign) quantization with packed-bit Hamming scoring.

Each coordinate is encoded as one bit (``sign(v_j) > 0``). The corpus is
packed into ``uint8`` blocks of 8 bits per byte, giving a 32x storage win over
``float32``. Scoring is computed as ``2 * matches / d - 1`` which, for unit
vectors with i.i.d. Gaussian-like coordinates, is a monotone surrogate for
cosine similarity (the order is what matters for retrieval metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from embedopt.compression.base import CompressedSet
from embedopt.utils.types import ByteArray, FloatArray


@dataclass(slots=True)
class BinaryQuantizeCompressor:
    """Sign quantization with packed Hamming similarity."""

    name: str = "binary"

    @property
    def trained(self) -> bool:
        return True

    def fit(self, vectors: FloatArray) -> None:  # noqa: D401 - protocol no-op
        return None

    def transform(self, vectors: FloatArray) -> CompressedSet:
        bits = (vectors > 0).astype(np.uint8)
        packed = np.packbits(bits, axis=1)
        bpv = int(packed.shape[1])
        return CompressedSet(codes=packed, bytes_per_vector=bpv)

    def score(self, queries: FloatArray, corpus: CompressedSet) -> FloatArray:
        codes = cast(ByteArray, corpus.codes)
        # Pack the queries the same way for symmetric Hamming.
        q_bits = (queries > 0).astype(np.uint8)
        q_packed = np.packbits(q_bits, axis=1)
        n_bits = queries.shape[1]
        # XOR pairs and popcount to get Hamming distances.
        # Use a precomputed popcount lookup table on uint8.
        popcount = (
            np.unpackbits(np.arange(256, dtype=np.uint8).reshape(-1, 1), axis=1)
            .sum(axis=1)
            .astype(np.int32)
        )
        n_q = q_packed.shape[0]
        n_c = codes.shape[0]
        sims = np.empty((n_q, n_c), dtype=np.float32)
        for i in range(n_q):
            xor = np.bitwise_xor(codes, q_packed[i])
            distances = popcount[xor].sum(axis=1)
            # Cosine surrogate from Hamming distance on sign codes.
            sims[i] = (1.0 - 2.0 * distances.astype(np.float32) / float(n_bits)).astype(np.float32)
        return sims

    def bytes_per_vector(self, dim: int) -> int:
        return int((dim + 7) // 8)
