"""Multi-seed variance ablation for codebook-trained compressors.

PQ and OPQ are stochastic in their codebook training. A single-seed
nDCG number conflates algorithmic effect with codebook-fitting noise.
This script evaluates the canonical PQ + OPQ grid at ``--seeds`` seeds
on the same embedded corpus and emits:

* ``<backbone>__<dataset>__seed_variance.csv`` -- one row per (spec, seed)
  with nDCG@10, Recall@10, MRR@10, MAP.
* ``<backbone>__<dataset>__seed_variance_summary.csv`` -- one row per
  spec with mean/std across seeds.

Example::

    python scripts/run_seed_variance.py --dim 128 --seeds 0 1 2 3 4
    python scripts/run_seed_variance.py \
        --backbone hashing --dim 256 --seeds 0 1 2 3 4 \
        --output-dir results-seed-variance
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

from embedopt.evaluation.datasets import RetrievalDataset, make_synthetic_retrieval
from embedopt.models.backbones import HashingEmbedder, TextEmbedder
from embedopt.pipelines.seed_variance import (
    SeedVarianceSummary,
    default_pq_specs,
    evaluate_compressor_seed_variance,
)
from embedopt.utils.seeding import set_global_seed
from embedopt.utils.types import as_float_array


def _load_dataset(spec: str, *, seed: int) -> RetrievalDataset:
    if spec == "synthetic":
        return make_synthetic_retrieval(n_queries_per_topic=4, seed=seed)
    if spec.startswith("beir-local:"):
        path = spec.split(":", 1)[1]
        from embedopt.evaluation.beir import load_beir_dataset_local

        return load_beir_dataset_local(Path(path))
    raise SystemExit(f"Unknown --dataset spec: {spec!r}")


def _make_embedder(name: str, dim: int) -> TextEmbedder:
    if name == "hashing":
        return HashingEmbedder(dim_=dim)
    from embedopt.models import make_backbone

    return make_backbone(name)


def _write_rows_csv(path: Path, summaries: Sequence[SeedVarianceSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "spec_label", "seed",
        "ndcg_at_10", "recall_at_10", "mrr_at_10", "map_at_10",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            for r in s.rows:
                writer.writerow(
                    {
                        "spec_label": r.spec_label,
                        "seed": r.seed,
                        "ndcg_at_10": r.ndcg_at_10,
                        "recall_at_10": r.recall_at_10,
                        "mrr_at_10": r.mrr_at_10,
                        "map_at_10": r.map_at_10,
                    }
                )


def _write_summary_csv(path: Path, summaries: Sequence[SeedVarianceSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "spec_label", "n_seeds",
        "ndcg_mean", "ndcg_std",
        "recall_mean", "recall_std",
        "mrr_mean", "mrr_std",
        "map_mean", "map_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "spec_label": s.spec_label,
                    "n_seeds": s.n_seeds,
                    "ndcg_mean": s.ndcg_mean,
                    "ndcg_std": s.ndcg_std,
                    "recall_mean": s.recall_mean,
                    "recall_std": s.recall_std,
                    "mrr_mean": s.mrr_mean,
                    "mrr_std": s.mrr_std,
                    "map_mean": s.map_mean,
                    "map_std": s.map_std,
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed codebook-variance ablation for PQ and OPQ. "
            "Quantifies how much of a spec's reported quality is the spec "
            "itself vs. the seed of its codebook training."
        )
    )
    parser.add_argument("--backbone", default="hashing")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Seeds to evaluate per spec. Default: 0..4.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Output directory (the CSV file names include backbone+dataset).",
    )
    args = parser.parse_args(argv)

    set_global_seed(args.seeds[0])
    embedder = _make_embedder(args.backbone, args.dim)
    dataset = _load_dataset(args.dataset, seed=args.seeds[0])

    corpus = as_float_array(embedder.encode(list(dataset.corpus)))
    queries = as_float_array(embedder.encode(list(dataset.queries)))
    qrels = dataset.qrels

    specs = default_pq_specs(corpus.shape[1])
    if not specs:
        raise SystemExit(
            f"No PQ specs fit dim={corpus.shape[1]}; pick a dim divisible by 4."
        )

    print(
        f"backbone={args.backbone}(dim={corpus.shape[1]})  dataset={dataset.name}  "
        f"corpus={corpus.shape[0]}  queries={queries.shape[0]}  "
        f"seeds={args.seeds}  specs={len(specs)}"
    )

    summaries: list[SeedVarianceSummary] = []
    skipped: list[tuple[str, str]] = []
    print(f"{'spec':<48s}  {'mean nDCG':>10s}  {'std':>7s}  n")
    from embedopt.compression.registry import spec_label as _spec_label
    for spec in specs:
        try:
            summary = evaluate_compressor_seed_variance(
                spec, args.seeds, corpus=corpus, queries=queries, qrels=qrels,
            )
        except ValueError as exc:
            # OPQ has a 256-vector training floor; PQ at large n_bits has its
            # own 2**n_bits floor. Skip with a clear message so smoke corpora
            # still produce a useful summary CSV for the trainable specs.
            label = _spec_label(spec)
            skipped.append((label, str(exc)))
            print(f"{label:<48s}  {'-- skipped --':>18s}")
            continue
        summaries.append(summary)
        print(
            f"{summary.spec_label:<48s}  {summary.ndcg_mean:>10.4f}  "
            f"{summary.ndcg_std:>7.4f}  {summary.n_seeds}"
        )
    if skipped:
        print(f"\n{len(skipped)} spec(s) skipped (corpus too small for codebook training):")
        for label, msg in skipped:
            print(f"  {label}: {msg}")

    base = f"{args.backbone}__{dataset.name}__seed_variance"
    _write_rows_csv(args.output_dir / f"{base}.csv", summaries)
    _write_summary_csv(args.output_dir / f"{base}_summary.csv", summaries)
    print(f"\nwrote {args.output_dir / f'{base}.csv'}")
    print(f"wrote {args.output_dir / f'{base}_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
