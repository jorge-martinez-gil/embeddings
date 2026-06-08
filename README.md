# EmbedCatalog

**A reproducible catalog for choosing embedding layouts under quality, memory,
and latency constraints.**

EmbedCatalog treats post-hoc embedding compression as a physical-design problem
for vector retrieval systems. Given a frozen text embedder, a workload, and
relevance judgments, it evaluates candidate vector layouts, measures retrieval
quality, storage, and latency, and returns a statistically annotated Pareto
catalog that operators can scan against their deployment budget.

Implementation note: the public method and paper contribution are now called
**EmbedCatalog**. The Python package and CLI are still exposed as `embedopt` for
backward compatibility with existing scripts, manifests, and experiment logs.

## Why This Matters

Modern retrieval systems often store one dense vector per object. At production
scale, the embedding column can become one of the largest materialized
structures in the system. Choosing between float32, float16, scalar int8,
binary codes, PQ, OPQ, truncation, or composed layouts is therefore not just a
modeling decision; it is a physical layout decision.

EmbedCatalog answers that decision empirically:

1. Freeze the embedding backbone.
2. Sweep a unified space of post-hoc vector layouts.
3. Measure quality, bytes per vector, and query latency under one protocol.
4. Attach uncertainty with paired bootstrap confidence intervals, paired
   randomization p-values, and codebook seed variance.
5. Emit a Pareto-filtered catalog plus reproducible manifests.

## Results At A Glance

The current artifact includes a 3 x 3 BEIR matrix over E5-base, BGE-base, and
MXBAI-large on SciFact, NFCorpus, and ArguAna, plus E5-base extensions to FiQA
and TREC-COVID.

| Finding | Practical reading | Evidence |
| --- | --- | --- |
| Float16 saves 2x storage with observed absolute nDCG@10 deltas below 1e-3. | A conservative first storage reduction for many vector collections. | `results/*__candidates.csv`, `figures/fig_storage_modes_bar.png` |
| Scalar int8 saves 4x storage with no detected nDCG@10 degradation at alpha = 0.05 in the main 9-pair matrix. | A strong default candidate when memory dominates and exact float scoring is not required. | candidate CSVs with `ci_lower`, `ci_upper`, `p_value`, `significant_05` |
| PQ, OPQ, binary, and truncate-then-PQ reach 48x to 128x reductions at looser quality budgets. | The best aggressive layout depends on backbone, dataset, and budget. | `figures/fig_pareto_quality_storage.png`, `figures/fig_pq_ablation_heatmap.png` |
| ANN execution changes the deployment frontier. | The representation and index backend should be selected together. | `results/*__index.csv`, `figures/fig_index_tradeoff.png` |
| Every reported number is tied to a manifest. | Results are replayable from config, seed, package version, and platform metadata. | `results/*__manifest.json`, `results/summary.json` |


## Claim-To-Artifact Map

| Paper claim | Where to inspect it | How to regenerate |
| --- | --- | --- |
| Unified post-hoc layout search over float32, float16, truncation, int8, binary, PQ, OPQ, and compositions. | `scripts/run_paper_experiments.py`, `src/embedopt/compression/`, `results/*__candidates.csv` | `bash scripts/run_all.sh` |
| Pareto catalogs over quality, storage, and latency. | `src/embedopt/moo/`, `results/summary.json`, `figures/fig_pair_pareto_frontiers.png` | `python scripts/plot_paper_figures.py --results-dir results --output-dir figures` |
| Statistical testing against the uncompressed baseline. | candidate CSV columns: `delta_vs_identity`, `ci_lower`, `ci_upper`, `p_value`, `significant_05` | `python scripts/run_paper_experiments.py ... --bootstrap-resamples 5000 --significance-resamples 5000` |
| Codebook seed variance for PQ and OPQ. | `results/e5-base__scifact__seed_variance_summary.csv` | `python scripts/run_seed_variance.py --backbone e5-base --dataset beir-local:data/scifact --seeds 0 1 2 3 4 --output-dir results` |
| ANN backend sensitivity. | `results/*__index.csv`, `figures/fig_index_tradeoff.png` | add `--index-backends exact-numpy faiss-flat faiss-ivf faiss-ivfpq faiss-hnsw faiss-opq` |
| Reproducibility and provenance. | `results/*__manifest.json` | manifests are written automatically by the experiment runner |

## Repository Layout

```text
src/embedopt/
  compression/     Vector layout operators: identity, float16, truncation,
                   scalar int8, binary, PQ, OPQ, and composed pipelines
  evaluation/      Retrieval metrics, BEIR loaders, per-query statistics,
                   bootstrap confidence intervals, randomization tests
  indexing/        Exact NumPy and optional FAISS-family ANN backends
  models/          Hashing, random projection, and sentence-transformer
                   backbones with query/passage prefix handling
  moo/             Pareto filtering, NSGA-II, scalarizers, hypervolume
  pipelines/       Smoke runs, Pareto sweeps, storage-mode comparisons
  profiling/       Bytes per vector, fit/encode time, median and p95 latency
  utils/           Seeding, run manifests, shared types

scripts/
  run_all.sh                    One-command paper experiment driver
  run_all.bat                   Windows driver
  run_paper_experiments.py      Main reproducibility script
  plot_paper_figures.py         Publication figure generator
  compare_storage_modes.py      Float32/float16/int8/binary/PQ comparison
  run_seed_variance.py          PQ and OPQ seed-variance ablation

results/                        Candidate CSVs, index CSVs, manifests
figures/                        PDF and PNG paper figures
notebooks/colab_a100.ipynb      Colab A100 walkthrough
```

## Quickstart

The offline smoke path needs no GPU and no external benchmark download.

```bash
python -m pip install -e ".[dev]"
pytest
embedopt version
embedopt pareto --dim 128 --seed 0
embedopt storage-modes --dim 128 --seed 0
```

Run the full paper driver on Linux, macOS, or Google Colab:

```bash
bash scripts/run_all.sh
```

Run the same driver on Windows:

```cmd
scripts\run_all.bat
```

Before launching the full matrix, a smoke run verifies the full experiment
path in a few seconds:

```bash
bash scripts/run_all.sh --smoke
```

## Reproducing The Paper Matrix

The default driver runs the EDBT proof-of-concept preset over SciFact,
NFCorpus, and ArguAna. It creates a virtual environment when needed, installs
the paper extras, downloads BEIR data, reuses cached embeddings, resumes from
intermediate checkpoints, and writes all outputs to `results/`.

```bash
bash scripts/run_all.sh
```

Useful overrides:

```bash
BENCHMARK_PRESET=beir-small     bash scripts/run_all.sh
BENCHMARK_PRESET=beir-full      bash scripts/run_all.sh
BACKBONES="e5-base"             bash scripts/run_all.sh
DATASETS="scifact nfcorpus"     bash scripts/run_all.sh
OUTPUT_DIR=results-2026-05      bash scripts/run_all.sh
BATCH_SIZE=256                  bash scripts/run_all.sh
SCORE_BATCH_SIZE=8              bash scripts/run_all.sh
SCORE_DEVICE=cuda               bash scripts/run_all.sh
SKIP_INSTALL=1 SKIP_DOWNLOAD=1  bash scripts/run_all.sh
```

Direct invocation is also supported:

```bash
python -m pip install -e ".[paper,index]"
python scripts/run_paper_experiments.py \
  --backbones e5-base bge-base mxbai-large \
  --datasets beir-local:data/scifact \
             beir-local:data/nfcorpus \
             beir-local:data/arguana \
  --batch-size 512 \
  --score-batch-size 32 \
  --score-device auto \
  --profile-repeats 20 \
  --bootstrap-resamples 5000 \
  --significance-resamples 5000 \
  --truncate-dims 32,64,128,256,512 \
  --pq-subspaces 4,8,16,32,64 \
  --pq-bits 4,6,8 \
  --index-backends exact-numpy faiss-flat faiss-ivf faiss-ivfpq \
  --output-dir results
```

## Output Artifacts

For each `(backbone, dataset)` pair, EmbedCatalog writes:

| Artifact | Contents |
| --- | --- |
| `results/<backbone>__<dataset>__candidates.csv` | One row per layout spec with nDCG@10, delta versus identity, confidence interval, p-value, bytes per vector, fit/encode time, and median/p95 query latency. |
| `results/<backbone>__<dataset>__index.csv` | One row per `(layout, backend)` pair with index build time, index size, recall/nDCG, exact-recall, and median/p95 search latency. |
| `results/<backbone>__<dataset>__manifest.json` | Config hash, seed, package version, platform info, Pareto labels, hypervolume, and output paths. |
| `results/summary.json` | Global backbone/dataset summary with hypervolume and Pareto-set size. |

Generate publication figures from any result directory:

```bash
python scripts/plot_paper_figures.py --results-dir results --output-dir figures
```

Available figure families:

| Figure | Purpose |
| --- | --- |
| `fig_pareto_quality_storage` | Quality versus bytes with Pareto points highlighted. |
| `fig_pair_pareto_frontiers` | Per-backbone and per-dataset frontier panels. |
| `fig_latency_storage_quality` | Three-objective latency/storage/quality view. |
| `fig_pq_ablation_heatmap` | PQ subspaces by bit-width ablation. |
| `fig_significance_ci_forest` | Paired-bootstrap confidence intervals. |
| `fig_training_cost` | Compressor fit and encoding cost. |
| `fig_index_tradeoff` | Index bytes versus index nDCG, colored by latency. |
| `fig_storage_modes_bar` | Float32, float16, int8, binary, PQ8, and PQ4 comparison. |
| `fig_hypervolume` | Frontier hypervolume by backbone and dataset. |

## Search Space

EmbedCatalog evaluates conservative, aggressive, and composed layouts under the
same interface:

| Family | Examples | Why it is included |
| --- | --- | --- |
| Raw precision | float32, float16 | Baselines and low-risk storage reductions. |
| Dimensional truncation | keep 32, 64, 128, 256, 512 dimensions | Tests whether leading dimensions preserve retrieval structure. |
| Scalar quantization | int8 affine quantization | Common vector-database storage mode with 4x footprint reduction. |
| Binary quantization | one bit per dimension | Extreme compression point for loose quality budgets. |
| Product quantization | PQ with multiple subspace and bit-width settings | Standard codebook-based vector compression. |
| Optimized PQ | OPQ rotation plus PQ | Tests whether learned rotations recover PQ quality at the same byte budget. |
| Compositions | truncate-then-PQ, truncate-then-binary | Explores layout pipelines beyond single operators. |

Each candidate implements `fit`, `transform`, `score`, and
`bytes_per_vector`, so it can be registered, evaluated, stored, and replayed as
a single specification string.

## Storage-Mode Comparison

For practitioners, the repository includes a direct comparison of the storage
layouts exposed by common vector engines:

| Mode | EmbedCatalog compressor | Typical production analogue |
| --- | --- | --- |
| float32 | `IdentityCompressor` | pgvector `vector`, Milvus `FLOAT`, FAISS `IndexFlat` |
| float16 | `Float16Compressor` | pgvector `halfvec`, Milvus `FLOAT16`, Qdrant `Float16` |
| int8 | `ScalarQuantizeCompressor` | FAISS `IndexScalarQuantizer`, Milvus `SQ8`, Qdrant `Int8` |
| binary | `BinaryQuantizeCompressor` | FAISS binary indexes, Milvus `BIN_*`, Qdrant `Binary` |
| PQ | `ProductQuantizeCompressor` | FAISS `IndexPQ`, Milvus `IVF_PQ`, OpenSearch FAISS PQ |

Run it locally:

```bash
python scripts/compare_storage_modes.py \
  --dim 128 \
  --seed 0 \
  --output results/storage_modes_comparison.csv
```

## ANN Backends

The `--index-backends` flag can compare representation choices against several
FAISS-family execution paths:

| Backend | FAISS analogue | Role |
| --- | --- | --- |
| `exact-numpy` | NumPy brute force | Dependency-free reference. |
| `faiss-flat` | `IndexFlatIP` | Exact dense inner product. |
| `faiss-ivf` | `IndexIVFFlat` | Inverted-file approximate search. |
| `faiss-ivfpq` | `IndexIVFPQ` | Inverted-file search with PQ codes. |
| `faiss-hnsw` | `IndexHNSWFlat` | Graph-based approximate search. |
| `faiss-opq` | `IndexPreTransform(OPQMatrix, IndexPQ)` | OPQ rotation plus PQ index. |

This is intentionally a codec-layer and FAISS-layer benchmark. It maps results
to documented production analogues in vector systems, but it does not claim to
benchmark full DBMS products end to end.

## Reproducibility Checklist

The artifact is designed to make paper review easy:

| Requirement | Status |
| --- | --- |
| Offline test suite | `pytest` runs deterministic unit and smoke tests. |
| Deterministic seeds | Synthetic data, NSGA-II, random projection, PQ/OPQ fitting, bootstrap, and randomization tests use explicit seeds. |
| Result provenance | Every experiment writes a `RunManifest` with config hash, package version, platform info, and repo state. |
| Resume support | Candidate-level checkpoints and cached embeddings are restored on interrupted runs. |
| Statistical uncertainty | Candidate CSVs include paired bootstrap CIs and paired randomization p-values. |
| Seed variance | PQ and OPQ layouts can be re-evaluated across codebook seeds. |
| Figure regeneration | Paper figures are produced directly from result CSVs. |
| CPU fallback | The scoring path supports CPU execution for CI and smoke runs. |

## CI

GitHub Actions runs:

```bash
ruff
black --check
isort --check-only
mypy --strict src
pytest
```

## License

This repository is released under the MIT License. See `LICENSE`.
