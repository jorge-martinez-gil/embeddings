"""Direct comparison of common vector-DB storage modes.

Runs ``float32`` (identity), ``float16``, ``int8`` (scalar quantization),
``binary`` (sign codes + Hamming), and ``PQ`` (product quantization) on
the same embedded corpus and emits:

* a monospace comparison table printed to stdout, and
* a CSV (``storage_modes_comparison.csv`` by default) with one row per
  mode and columns covering bytes / vec, compression ratio, retrieval
  metrics, delta vs. float32, fit / encode / query latency.

By default the script uses the dependency-free ``HashingEmbedder`` and
the synthetic retrieval dataset shipped with the package, so it runs
offline in a few seconds. Use ``--dataset beir-local:data/scifact`` (and
the ``[paper]`` extra) to point it at a BEIR shard, or
``--backbone e5-base`` to use a paper-grade encoder.

Example::

    python scripts/compare_storage_modes.py --dim 128 --seed 0
    python scripts/compare_storage_modes.py \
        --backbone hashing --dim 256 --output results/storage_modes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from embedopt.evaluation.datasets import RetrievalDataset, make_synthetic_retrieval
from embedopt.models.backbones import HashingEmbedder, TextEmbedder
from embedopt.pipelines.storage_modes import (
    compare_storage_modes,
    default_storage_mode_specs,
    format_comparison_table,
)
from embedopt.utils.seeding import set_global_seed


def _load_dataset(spec: str, *, seed: int) -> RetrievalDataset:
    """Load a dataset from a CLI ``--dataset`` value.

    Accepts:
      * ``synthetic`` (default): the deterministic synthetic retrieval dataset.
      * ``beir-local:<path>``: a local BEIR shard, lazily importing the loader.
    """
    if spec == "synthetic":
        return make_synthetic_retrieval(n_queries_per_topic=4, seed=seed)
    if spec.startswith("beir-local:"):
        path = spec.split(":", 1)[1]
        try:
            from embedopt.evaluation.beir import load_beir_dataset_local
        except ImportError as exc:  # pragma: no cover - optional extra
            raise SystemExit(
                "BEIR loader requires the [paper] extra. "
                "Install with: pip install -e .[paper]"
            ) from exc
        return load_beir_dataset_local(Path(path))
    raise SystemExit(f"Unknown --dataset spec: {spec!r}")


def _make_embedder(name: str, dim: int) -> TextEmbedder:
    """Construct a backbone from a CLI ``--backbone`` value."""
    if name == "hashing":
        return HashingEmbedder(dim_=dim)
    # Anything else: route through the paper-grade backbone factory.
    try:
        from embedopt.models import make_backbone
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SystemExit(
            "Paper-grade backbones require the [paper] extra. "
            "Install with: pip install -e .[paper]"
        ) from exc
    return make_backbone(name)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write the comparison rows to CSV.

    The ``spec`` column is JSON-encoded so the CSV stays one-row-per-mode
    even when the spec dict has nested kwargs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            out["spec"] = json.dumps(out["spec"], sort_keys=True)
            writer.writerow(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Direct comparison of common vector-DB storage modes "
            "(float32, float16, int8, binary, PQ) on a fixed corpus."
        )
    )
    parser.add_argument(
        "--backbone",
        default="hashing",
        help="Backbone name. 'hashing' is offline; others go through the "
        "paper backbone factory (requires [paper] extra).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Embedding dim (ignored for paper-grade backbones).",
    )
    parser.add_argument(
        "--dataset",
        default="synthetic",
        help="Dataset spec. 'synthetic' or 'beir-local:<path>'.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=20,
        help="Latency repeats per mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "storage_modes_comparison.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    set_global_seed(args.seed)
    embedder = _make_embedder(args.backbone, args.dim)
    dataset = _load_dataset(args.dataset, seed=args.seed)

    # ``HashingEmbedder`` accepts the requested dim; paper backbones expose
    # their native dim via the ``dim`` property.
    backbone_dim = embedder.dim
    modes = default_storage_mode_specs(backbone_dim)
    comparison = compare_storage_modes(
        embedder,
        dataset,
        modes=modes,
        profile_repeats=args.profile_repeats,
    )
    print(
        f"backbone = {args.backbone} (dim={backbone_dim})  "
        f"dataset = {dataset.name}  "
        f"n_corpus = {len(dataset.corpus)}  n_queries = {len(dataset.queries)}"
    )
    print(format_comparison_table(comparison))
    _write_csv(args.output, [r.as_dict() for r in comparison.rows])
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
