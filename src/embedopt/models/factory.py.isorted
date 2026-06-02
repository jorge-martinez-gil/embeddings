"""Backbone factory.

A thin wrapper that maps a short name (``"e5-base"``, ``"bge-base"``,
``"mxbai-large"``, ``"hashing"``) to a properly-configured :class:`TextEmbedder`.
This is the single seam the paper experiments use to swap backbones, so that
every other component (compression, evaluation, profiling) sees the same
interface regardless of which encoder is in play.

E5 and BGE require asymmetric query/passage prefixes; the paper-grade
backbones below apply them automatically when the framework calls
``encode(texts, kind=...)`` via :class:`PrefixedEmbedder`. The default
``encode(texts)`` (no kind) treats inputs as passages, matching the
:class:`TextEmbedder` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from embedopt.models.backbones import (
    HashingEmbedder,
    RandomProjectionEmbedder,
    SentenceTransformerEmbedder,
    TextEmbedder,
)

Kind = Literal["query", "passage"]


@dataclass(slots=True)
class PrefixedEmbedder:
    """Wraps a :class:`TextEmbedder` and prepends per-kind prefixes.

    E5 expects ``"query: "`` / ``"passage: "``; BGE expects an instruction on
    queries only; mxbai uses a single combined instruction format. The exact
    strings are configurable so paper authors can sweep them as ablations.
    """

    inner: TextEmbedder
    name: str
    query_prefix: str = ""
    passage_prefix: str = ""

    @property
    def dim(self) -> int:
        return self.inner.dim

    def encode(self, texts: list[str], *, kind: Kind = "passage") -> list[list[float]]:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        prefixed = [prefix + t for t in texts] if prefix else list(texts)
        return self.inner.encode(prefixed)


# Catalog of backbone configurations used in the paper. Each entry returns a
# fresh embedder when invoked, so the same key can be reused without leaking
# state.
_PAPER_BACKBONES: dict[str, dict[str, Any]] = {
    "hashing": {"factory": "hashing", "kwargs": {"dim_": 256}},
    "random-proj-256": {"factory": "random_projection", "kwargs": {"dim_": 256, "seed": 0}},
    "e5-base": {
        "factory": "sentence_transformer",
        "model_name": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge-base": {
        "factory": "sentence_transformer",
        "model_name": "BAAI/bge-base-en-v1.5",
        "query_prefix": ("Represent this sentence for searching relevant passages: "),
        "passage_prefix": "",
    },
    "mxbai-large": {
        "factory": "sentence_transformer",
        "model_name": "mixedbread-ai/mxbai-embed-large-v1",
        "query_prefix": ("Represent this sentence for searching relevant passages: "),
        "passage_prefix": "",
    },
}


@dataclass(slots=True, frozen=True)
class BackboneSpec:
    """Resolved backbone description, returned by :func:`describe_backbone`."""

    name: str
    family: str
    dim: int = 0
    extras: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def list_backbones() -> list[str]:
    """Names of every backbone known to the factory."""
    return sorted(_PAPER_BACKBONES.keys())


def make_backbone(name: str) -> TextEmbedder:
    """Instantiate a backbone by name.

    Returns either a bare :class:`TextEmbedder` (for the deterministic
    backbones used in tests) or a :class:`PrefixedEmbedder` wrapping a
    :class:`SentenceTransformerEmbedder` for the paper-grade ones.
    """
    if name not in _PAPER_BACKBONES:
        raise KeyError(f"Unknown backbone {name!r}; known: {sorted(_PAPER_BACKBONES)}")
    cfg = _PAPER_BACKBONES[name]
    factory = str(cfg["factory"])
    if factory == "hashing":
        return HashingEmbedder(**cast(dict[str, Any], cfg["kwargs"]))
    if factory == "random_projection":
        return RandomProjectionEmbedder(**cast(dict[str, Any], cfg["kwargs"]))
    if factory == "sentence_transformer":
        inner = SentenceTransformerEmbedder(model_name=str(cfg["model_name"]))
        return PrefixedEmbedder(
            inner=inner,
            name=name,
            query_prefix=str(cfg.get("query_prefix", "")),
            passage_prefix=str(cfg.get("passage_prefix", "")),
        )
    raise ValueError(f"Unknown factory {factory!r} for backbone {name!r}")
