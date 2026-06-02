"""Embedding backbone abstractions.

The framework treats the *backbone* as a frozen text encoder; all downstream
components (compression, evaluation, profiling) consume its outputs only. Two
deterministic backbones are provided for offline reproducibility:

* :class:`HashingEmbedder` — feature hashing on whitespace tokens, no
  dependencies, fully deterministic and dependency-free. Useful for CI and for
  controlled compression experiments where backbone confounds are undesirable.
* :class:`RandomProjectionEmbedder` — seeded random projection of hashed
  features into a configurable dimension. Useful for sweeping ``d``.

The original :class:`SentenceTransformerEmbedder` remains for headline numbers
once the ``[models]`` extra is installed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import numpy as np

from embedopt.utils.types import FloatArray


class TextEmbedder(Protocol):
    """Protocol for text embedding backbones."""

    name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to L2-normalized embedding vectors."""

    @property
    def dim(self) -> int:
        """Output embedding dimension."""


def _l2_normalize(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return cast(FloatArray, (matrix / norms).astype(np.float32, copy=False))


@dataclass(slots=True)
class HashingEmbedder:
    """Deterministic feature-hashing embedder.

    Tokens are split on whitespace, lower-cased, and mapped via SHA-256 to a
    bucket in ``[0, dim)`` with a sign bit, producing a sparse signed-count
    vector that is then L2-normalized. The encoder is purely deterministic and
    has no learned parameters, which makes it ideal for unit tests and for
    isolating compression effects from backbone variance.
    """

    dim_: int = 128
    name: str = "hashing"

    @property
    def dim(self) -> int:
        return self.dim_

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.dim_ <= 0:
            raise ValueError("dim must be positive")
        raw = np.zeros((len(texts), self.dim_), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dim_
                sign = 1.0 if (digest[8] & 1) == 0 else -1.0
                raw[i, bucket] += sign
        normalized = _l2_normalize(raw)
        result: list[list[float]] = normalized.tolist()
        return result


@dataclass(slots=True)
class RandomProjectionEmbedder:
    """Seeded random projection of hashed features into ``dim`` dimensions.

    Combines :class:`HashingEmbedder` with a fixed Gaussian projection matrix.
    Useful when sweeping the *backbone* dimension is part of the experimental
    matrix; the projection is reseeded deterministically from ``seed``.
    """

    dim_: int = 64
    hash_dim: int = 1024
    seed: int = 0
    name: str = "random_projection"

    _projection: FloatArray = field(init=False, repr=False)
    _hasher: HashingEmbedder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        proj = rng.standard_normal((self.hash_dim, self.dim_)).astype(np.float32)
        self._projection = proj
        self._hasher = HashingEmbedder(dim_=self.hash_dim)

    @property
    def dim(self) -> int:
        return self.dim_

    def encode(self, texts: list[str]) -> list[list[float]]:
        hashed = np.asarray(self._hasher.encode(texts), dtype=np.float32)
        projected = hashed @ self._projection
        out = _l2_normalize(projected)
        result: list[list[float]] = out.tolist()
        return result


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """Sentence-Transformers embedder wrapper.

    Requires the ``[models]`` extra. Output is L2-normalized to match the
    contract of :class:`TextEmbedder`.
    """

    model_name: str
    name: str = "sentence_transformer"

    _model: Any = None
    _dim: int = 0

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime CLI
            raise ModuleNotFoundError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install -e .[models]"
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        out: list[list[float]] = vectors.tolist()
        return out
