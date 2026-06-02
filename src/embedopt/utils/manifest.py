"""Run manifests: a structured record of *what* was run and *how*.

A manifest captures the seed set, configuration hash, package version, and
arbitrary metadata. Manifests are emitted as JSON next to result tables so a
reviewer can re-execute a paper experiment from a single artifact.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embedopt import __version__


@dataclass(slots=True)
class RunManifest:
    """Metadata about an experiment run."""

    name: str
    config: Mapping[str, Any]
    seed: int
    embedopt_version: str = __version__
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    started_at: float = field(default_factory=lambda: time.time())
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        """Deterministic SHA256 of the canonicalized config."""
        canonical = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config_hash"] = self.config_hash
        return d

    def write(self, path: str | Path) -> Path:
        """Write the manifest JSON to ``path``. Parent dirs are created."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p
