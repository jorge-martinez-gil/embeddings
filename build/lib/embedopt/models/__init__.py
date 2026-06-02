"""Embedding backbone implementations."""

from embedopt.models.backbones import (
    HashingEmbedder,
    RandomProjectionEmbedder,
    SentenceTransformerEmbedder,
    TextEmbedder,
)
from embedopt.models.factory import (
    BackboneSpec,
    Kind,
    PrefixedEmbedder,
    list_backbones,
    make_backbone,
)

__all__ = [
    "BackboneSpec",
    "HashingEmbedder",
    "Kind",
    "PrefixedEmbedder",
    "RandomProjectionEmbedder",
    "SentenceTransformerEmbedder",
    "TextEmbedder",
    "list_backbones",
    "make_backbone",
]
