from __future__ import annotations

import pytest

from embedopt.models.factory import list_backbones, make_backbone


def test_list_backbones_includes_paper_grade() -> None:
    names = list_backbones()
    assert "hashing" in names
    assert "e5-base" in names
    assert "bge-base" in names
    assert "mxbai-large" in names


def test_make_backbone_hashing_roundtrip() -> None:
    enc = make_backbone("hashing")
    out = enc.encode(["hello world"])
    assert len(out) == 1
    assert enc.dim == 256


def test_make_backbone_unknown_raises() -> None:
    with pytest.raises(KeyError):
        make_backbone("does-not-exist")
