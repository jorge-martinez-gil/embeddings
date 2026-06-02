from __future__ import annotations

import json
import tempfile
from pathlib import Path

from embedopt.utils.manifest import RunManifest


def test_manifest_hash_is_canonical() -> None:
    m1 = RunManifest(name="x", config={"a": 1, "b": 2}, seed=0)
    m2 = RunManifest(name="x", config={"b": 2, "a": 1}, seed=0)
    assert m1.config_hash == m2.config_hash


def test_manifest_round_trip() -> None:
    m = RunManifest(name="exp", config={"x": [1, 2, 3]}, seed=11, extra={"git": "abc"})
    with tempfile.TemporaryDirectory() as tmp:
        path = m.write(Path(tmp) / "manifest.json")
        data = json.loads(path.read_text())
    assert data["name"] == "exp"
    assert data["config_hash"] == m.config_hash
    assert data["extra"]["git"] == "abc"
