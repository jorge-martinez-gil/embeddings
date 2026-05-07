"""Embedding backbone abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast


class TextEmbedder(Protocol):
    """Protocol for text embedding backbones."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to embedding vectors."""


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """Sentence-Transformers embedder wrapper."""

    model_name: str

    _model: Any = None

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime CLI
            raise ModuleNotFoundError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install -e .[models]"
            ) from exc

        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return cast(list[list[float]], vectors.tolist())
