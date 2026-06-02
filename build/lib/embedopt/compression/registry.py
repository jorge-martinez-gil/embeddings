"""Registry mapping spec dictionaries to concrete :class:`Compressor` instances.

A *compressor spec* is a small ``dict`` (typically loaded from a Hydra config)
of the form::

    {"name": "truncate", "keep_dim": 128}

Compositions are expressed as::

    {"name": "composed", "stages": [{"name": "truncate", "keep_dim": 64},
                                     {"name": "product_quantize", "n_subspaces": 8}]}

This module centralizes spec-to-instance construction so that the multi-
objective sweep can iterate over a search space described as a list of specs
without importing concrete classes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embedopt.compression.base import Compressor
from embedopt.compression.binary import BinaryQuantizeCompressor
from embedopt.compression.composed import ComposedCompressor
from embedopt.compression.identity import IdentityCompressor
from embedopt.compression.product import ProductQuantizeCompressor
from embedopt.compression.scalar import ScalarQuantizeCompressor
from embedopt.compression.truncate import TruncateCompressor


def build_compressor(spec: Mapping[str, Any]) -> Compressor:
    """Instantiate a compressor from a spec dict.

    The ``name`` key selects the class; remaining keys are forwarded as
    constructor kwargs. The special ``composed`` name expects a ``stages`` key
    whose value is a list of sub-specs.
    """
    if "name" not in spec:
        raise ValueError("compressor spec must include a 'name' field")
    name = str(spec["name"])
    if name == "composed":
        stages_raw = spec.get("stages", [])
        if not isinstance(stages_raw, (list, tuple)) or len(stages_raw) < 1:
            raise ValueError("composed spec needs non-empty 'stages' list")
        stages = [build_compressor(s) for s in stages_raw]
        return ComposedCompressor(stages=stages)
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    if name == "identity":
        return IdentityCompressor(**kwargs)
    if name == "truncate":
        return TruncateCompressor(**kwargs)
    if name == "scalar_int8":
        return ScalarQuantizeCompressor(**kwargs)
    if name == "binary":
        return BinaryQuantizeCompressor(**kwargs)
    if name == "product_quantize":
        return ProductQuantizeCompressor(**kwargs)
    raise ValueError(f"Unknown compressor name: {name!r}")


def spec_label(spec: Mapping[str, Any]) -> str:
    """Human-readable label for a spec, useful for plot legends and tables."""
    name = str(spec["name"])
    if name == "composed":
        inner = "+".join(spec_label(s) for s in spec.get("stages", []))
        return f"composed({inner})"
    rest = ",".join(f"{k}={v}" for k, v in spec.items() if k != "name")
    return f"{name}({rest})" if rest else name
