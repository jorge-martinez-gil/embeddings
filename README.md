# embedopt

`embedopt` is a research framework for **multi-objective post-hoc optimization
of text embeddings**. Given a frozen embedder, it discovers the Pareto front
of *(retrieval quality, storage size, query latency)* across a configurable
search space of compression operators (Matryoshka truncation, scalar
quantization, binary codes, product quantization, **and chained compositions
of those operators**), and reports the hypervolume-indicator of that front so
different backbones can be compared under a single scalar.

The package is built to support a venue-quality publication: every component
is deterministic under a seed, every experiment can be reproduced from a
single config + manifest, and the full headline experiment runs end-to-end on
a Google Colab A100 (paper-grade backbones × BEIR datasets) from a single
script.

## Architecture

```
src/embedopt/
├── compression/   Identity / Float16 / Truncate / ScalarInt8 / Binary / ProductQuantize / Composed
├── evaluation/    metrics (Spearman, Recall@K, MRR, nDCG, MAP) + per-query stats
│                  + paired-bootstrap CIs + synthetic & BEIR datasets + runner
├── models/        TextEmbedder protocol + Hashing / RandomProjection /
│                  SentenceTransformer + paper-grade backbone factory
├── moo/           Pareto sort, NSGA-II, exact 2D/3D hypervolume, scalarizers
├── pipelines/     smoke + pareto sweep
├── profiling/     latency (median/p95), tracemalloc memory, byte accountant
└── utils/         seeding, run manifest, type aliases
scripts/run_paper_experiments.py   # Colab A100 entrypoint
notebooks/colab_a100.ipynb         # one-click reproducer
```

## Quickstart (offline, no GPU)

```bash
python -m pip install -e .[dev]
pytest                                       # 69 tests, fully offline
embedopt version
embedopt pareto --dim 128 --seed 0           # synthetic Pareto sweep
embedopt evaluate identity --dim 128
embedopt profile product_quantize --dim 64
embedopt storage-modes --dim 128 --seed 0    # float32 vs. float16 vs. int8 vs. binary vs. PQ
```

## Storage modes — direct comparison

For practitioners deciding between the storage layouts every modern vector
database exposes (`pgvector` `vector` / `halfvec`, FAISS `IndexFlat` /
`IndexScalarQuantizer` / `IndexBinary` / `IndexPQ`, Milvus `FLOAT` /
`FLOAT16` / `SQ8` / `BIN_*` / `IVF_PQ`, Qdrant `Float32` / `Float16` /
`Int8` / `Binary`), the framework ships a dedicated side-by-side ablation
that runs all five canonical modes on the same embedded corpus:

| Mode       | Backing compressor                | Bytes / dim                | Compression vs. float32 | Typical DB name           |
|------------|-----------------------------------|----------------------------|-------------------------|---------------------------|
| float32    | `IdentityCompressor`              | 4                          | 1×                      | `vector`, `FLOAT`         |
| float16    | `Float16Compressor`               | 2                          | 2×                      | `halfvec`, `FLOAT16`      |
| int8       | `ScalarQuantizeCompressor`        | 1                          | 4×                      | `SQ8`, `Int8`             |
| binary     | `BinaryQuantizeCompressor`        | 1/8                        | 32×                     | `BIN_FLAT`, `Binary`      |
| PQ-8bit    | `ProductQuantizeCompressor`       | `n_subspaces / dim`        | dim-dependent           | `IVF_PQ`, `IndexPQ`       |
| PQ-4bit    | `ProductQuantizeCompressor`       | `n_subspaces / dim`        | same as PQ-8bit*        | `PQ4`, `IVF_PQ4`          |

*PQ-4bit uses 16 centroids per subspace instead of 256 — the byte budget
this framework reports is the same one ``uint8`` per subspace because the
compressor stores codes un-packed, so the PQ4 row is an *ablation*
showing how centroid count affects quality holding storage fixed (real
deployments that nibble-pack would halve those PQ bytes too).

Run the comparison from the CLI:

```bash
embedopt storage-modes --dim 128 --seed 0
```

Or as a standalone script that also drops a CSV next to the table:

```bash
python scripts/compare_storage_modes.py --dim 128 --seed 0 \
    --output results/storage_modes_comparison.csv
```

The script reports `bytes_per_vector`, the compression ratio against
float32, `nDCG@10`, the Δ-nDCG vs. float32, `Recall@10`, `MRR@10`, plus
fit / encode / median / p95 query latency for each mode — a single table
that answers "should I move from `float32` to `halfvec`, or jump straight
to PQ?" without re-running a Pareto sweep. Point it at a BEIR shard with
`--dataset beir-local:data/scifact` once the `[paper]` extra is
installed.

## Codebook seed variance (PQ + OPQ)

PQ and OPQ are stochastic in their codebook training, so a single-seed
nDCG number conflates the spec's effect with codebook-fitting noise.
`scripts/run_seed_variance.py` evaluates the canonical PQ+OPQ grid at
multiple seeds on the same embedded corpus and emits:

* `<backbone>__<dataset>__seed_variance.csv` — one row per (spec, seed).
* `<backbone>__<dataset>__seed_variance_summary.csv` — one row per spec
  with mean / std nDCG / Recall / MRR / MAP across seeds.

```bash
python scripts/run_seed_variance.py --dim 128 --seeds 0 1 2 3 4
# or against a paper-grade backbone + BEIR shard:
python scripts/run_seed_variance.py --backbone e5-base \
    --dataset beir-local:data/scifact --seeds 0 1 2 3 4 \
    --output-dir results
```

The summary CSV is the artifact behind the "mean ± std" variance bars
in the paper's PQ ablation figure.

## Paper experiments — one command

A driver script handles venv creation, installing `[paper]`, downloading the
BEIR datasets, and running the full backbone × dataset matrix. On Colab,
Linux, or macOS:

```bash
bash scripts/run_all.sh
```

By default this runs the `edbt-poc` preset: `scifact`, `nfcorpus`, and
`arguana`. These are real BEIR retrieval benchmarks, but small enough for fast
iteration and paper proof-of-concept runs. Use `BENCHMARK_PRESET=beir-full`
only when you intentionally want the original large BEIR matrix.

In a Colab notebook cell, either `!bash scripts/run_all.sh` or
`!sh scripts/run_all.sh` works; the script re-enters Bash automatically and
uses Colab's current Python environment by default so the A100-enabled PyTorch
runtime stays visible.

On Windows:

```cmd
scripts\run_all.bat
```

Smoke run (offline, no GPU, ~5 seconds) to verify the whole pipeline before
launching the real thing:

```bash
bash scripts/run_all.sh --smoke
```

Override anything via env vars (the script picks up sane defaults otherwise):

```bash
BENCHMARK_PRESET=beir-small     bash scripts/run_all.sh   # adds fiqa and trec-covid
BENCHMARK_PRESET=beir-full      bash scripts/run_all.sh   # original 10-dataset BEIR matrix
BACKBONES="e5-base"             bash scripts/run_all.sh
DATASETS="scifact nfcorpus"     bash scripts/run_all.sh
OUTPUT_DIR=results-2026-05      bash scripts/run_all.sh
BATCH_SIZE=256                  bash scripts/run_all.sh   # lower if the GPU runs out of memory
SCORE_BATCH_SIZE=8              bash scripts/run_all.sh   # lower RAM during top-k scoring
SCORE_DEVICE=cuda               bash scripts/run_all.sh   # require GPU top-k scoring
SKIP_INSTALL=1 SKIP_DOWNLOAD=1  bash scripts/run_all.sh   # re-run with cached venv+data
```

The default encoder `BATCH_SIZE` is `512`, which is intended for large-memory
GPUs such as A100/H100. If encoding is killed by GPU memory pressure, retry
with `BATCH_SIZE=256` or `BATCH_SIZE=128`; completed checkpoints and cached
embeddings will still be reused.

For very large corpora, prefer `SCORE_BATCH_SIZE=8` or `SCORE_BATCH_SIZE=4`.
This keeps quality evaluation and index evaluation from allocating a full
`queries x corpus` score matrix.
Full-corpus top-k quality scoring uses `SCORE_DEVICE=auto` by default, which
uses PyTorch CUDA when available and otherwise falls back to NumPy on CPU.
Use `SCORE_DEVICE=cuda` when you want the run to fail fast if the GPU is not
visible. CUDA acceleration covers dense scoring and PQ/Truncate+PQ quality
scoring; unsupported specs fall back to CPU. This accelerates quality scoring
only; the reported compressor latency columns still use the compressor's
standard scoring implementation.

Experiment runs are resumable by default. If a requested
`<backbone>__<dataset>__candidates.csv` and a matching manifest already exist
in `OUTPUT_DIR`, that pair is skipped on the next startup/run; when
`--index-backends` is provided, the matching index CSV must also exist. Pass
`--force` to `scripts/run_paper_experiments.py` when you intentionally want to
recompute completed pairs.

While the benchmark is still running, each completed compressor spec is also
checkpointed under `OUTPUT_DIR/_intermediate/<backbone>__<dataset>/` using
readable files such as `identity.json`, `truncate(keep_dim=32).json`, and
`truncate(keep_dim=64).json`. If the process is interrupted, the next run
restores those lower-granularity results instead of recomputing them.
Backbone embeddings are also cached under `OUTPUT_DIR/_embeddings/`, so a
resume does not need to re-encode the UKP/BEIR corpus before reaching the
candidate checkpoints. Once the full requested benchmark finishes and
`summary.json` is written, these transient checkpoint/cache trees are removed
automatically; if any run fails, they are kept for the next resume.

Or, if you'd rather drive the experiment script yourself (e.g. inside a
notebook cell):

```bash
python -m pip install -e .[paper]
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
    --index-backends exact-numpy \
    --output-dir results
```

For FAISS index-level evaluation, install the optional index extra and add the
backend:

```bash
python -m pip install -e .[paper,index]
python scripts/run_paper_experiments.py ... --index-backends exact-numpy faiss-flat
```

### Production ANN backends

The `--index-backends` flag accepts any combination of:

| Backend         | FAISS class            | Approximates                                              |
|-----------------|------------------------|-----------------------------------------------------------|
| `exact-numpy`   | (NumPy)                | brute-force inner product, dependency-free reference      |
| `faiss-flat`    | `IndexFlatIP`          | pgvector `vector`, Milvus `FLAT`, Qdrant `flat`           |
| `faiss-ivf`     | `IndexIVFFlat`         | Milvus `IVF_FLAT`, Vespa IVF, Pinecone IVF                |
| `faiss-ivfpq`   | `IndexIVFPQ`           | Milvus `IVF_PQ`, OpenSearch `faiss/ivf_pq`, ScaNN PQ      |
| `faiss-hnsw`    | `IndexHNSWFlat`        | pgvector `hnsw`, Qdrant `hnsw`, Milvus `HNSW`, Weaviate   |
| `faiss-opq`     | `IndexPreTransform(OPQMatrix, IndexPQ)` | FAISS `OPQ`, Milvus `IVF_PQ` w/ rotation, ScaNN |

`faiss-ivf` and `faiss-ivfpq` pick a sensible `nlist = sqrt(n_corpus)` and
`nprobe = nlist // 8` at build time, so the same flag works on a 500-row
smoke corpus and on a 1M-row BEIR shard without tuning. Pass them
alongside the existing backends to compare in-database ANN structures
against compressor-side scoring:

```bash
python scripts/run_paper_experiments.py ... \
    --index-backends exact-numpy faiss-flat faiss-ivf faiss-ivfpq
```

The resulting `<backbone>__<dataset>__index.csv` gets one row per
(spec, backend) pair, with index build time, on-disk size, top-k recall,
nDCG@10, median + p95 search latency, and exact-recall against the
brute-force baseline — i.e. the table that answers "if I deploy this
compressor in Milvus IVF\_PQ, what do I actually lose?".

For a guided Colab walkthrough, open `notebooks/colab_a100.ipynb`.

For each `(backbone, dataset)` pair the script writes:
- `results/<backbone>__<dataset>__candidates.csv` — every spec with nDCG@10,
  Δ vs. identity, paired-bootstrap 95% CI, bytes/vec, median + p95 latency.
- `results/<backbone>__<dataset>__manifest.json` — config hash, seed,
  Pareto-set spec labels, hypervolume, repo state.
- `results/summary.json` — all (backbone, dataset, hypervolume) tuples.

Newer runs also add paired randomization p-values, compressor fit/encode time,
and, when `--index-backends` is provided, a separate
`results/<backbone>__<dataset>__index.csv` with index build time, index bytes,
index recall/nDCG, median latency, and p95 latency.

A smoke run that exercises the full code path on the synthetic dataset:

```bash
python scripts/run_paper_experiments.py --smoke --output-dir results-smoke
```

## Paper figures

Generate publication figures directly from a result directory:

```bash
python scripts/plot_paper_figures.py --results-dir results --output-dir figures
```

The script writes PDF and high-DPI PNG versions of:
- `fig_pareto_quality_storage` — quality vs. bytes with Pareto points outlined.
- `fig_pair_pareto_frontiers` — per-backbone/per-dataset Pareto frontier panels.
- `fig_latency_storage_quality` — latency/storage/quality trade-off.
- `fig_pq_ablation_heatmap` — PQ subspaces × bit-width ablation.
- `fig_significance_ci_forest` — paired-bootstrap confidence intervals.
- `fig_training_cost` — compressor fit/codebook training plus encoding cost.
- `fig_index_tradeoff` — index bytes vs. index nDCG, colored by latency.
- `fig_storage_modes_bar` — grouped bar chart of nDCG@10 and Recall@10 per
  storage mode (float32 / float16 / int8 / binary / PQ8 / PQ4), with
  bytes/vec and compression ratio annotated. Reads
  `storage_modes_comparison.csv` if present.
- `fig_hypervolume` — frontier hypervolume by backbone/dataset.

Use `--formats pdf png svg` if you want SVG exports too.

## The headline experiment, conceptually

`embedopt pareto` (and the paper script) performs, for each compressor spec:

1. Fit on the embedded corpus (PQ trains codebooks; scalar fits per-dim min/max).
2. Profile per-vector bytes and per-query latency (median + p95).
3. Evaluate Recall/MRR/nDCG/MAP on the dataset.

Then the framework filters non-dominated points under
`(quality, bytes_per_vector, query_latency_ms)` and reports the hypervolume
of the resulting front against a fixed reference. Example output on the
synthetic suite:

```
spec                                       nDCG@10   B/vec  lat ms  pareto
identity                                    0.7421     512   0.082    *
truncate(keep_dim=32)                       0.6210     128   0.041    *
scalar_int8                                 0.7402     128   0.165
binary                                      0.6105      16   0.213    *
product_quantize(n_subspaces=8,n_bits=8)    0.7088       8   0.422    *
composed(truncate(keep_dim=64)+pq(M=8))     0.6841       8   0.180
hypervolume = 0.382174
```

## Reproducibility

Every randomized component (synthetic data, codebook init, NSGA-II,
random-projection backbone, paired-bootstrap resampling) takes an explicit
seed; `set_global_seed(seed)` threads a derived `SeedSet` through Python
`random` and `numpy`. Each experiment is persisted alongside a `RunManifest`
(`embedopt.utils.RunManifest`) that records the canonicalized config, its
SHA-256 hash, package version, and platform info — enough to replay a result
from disk.

## Paper-readiness checklist

What's done in this repo:

* End-to-end Pareto sweep with five base compressors and three objectives.
* **Composed compressors** (Truncate → PQ, Truncate → Binary) so NSGA-II
  searches a strictly larger space than enumeration over the base methods.
* Vectorized PQ scoring (~10× faster than the per-subspace loop on
  BEIR-sized corpora).
* NSGA-II for larger search spaces; exact 2D / 3D hypervolume; weighted-sum
  and Tchebycheff scalarizers.
* **BEIR adapter** (`load_beir_dataset_local`, `load_beir_dataset_hf`) with
  fast EDBT proof-of-concept defaults (`scifact`, `nfcorpus`, `arguana`),
  plus opt-in `beir-small` and full BEIR presets.
* **Paper-grade backbones** wired in: `intfloat/e5-base-v2`,
  `BAAI/bge-base-en-v1.5`, `mixedbread-ai/mxbai-embed-large-v1` — with the
  correct query/passage prefixes baked into a `PrefixedEmbedder`.
* **Statistical significance**: paired-bootstrap 95% CIs on per-query nDCG@10
  vs. the identity baseline, written into the per-spec CSV.
* **Paired significance tests**: paired randomization p-values are emitted
  alongside confidence intervals for every candidate.
* **Ablation grids**: CLI knobs cover truncation dimensions, PQ subspaces,
  PQ bit-widths, and composed truncate-then-PQ settings.
* **Index-level evaluation**: dependency-free exact NumPy indexing plus
  optional FAISS flat indexing via the `[index]` extra.
* **PQ training cost accounting**: fit and encode time are emitted per
  compressor so codebook training is visible alongside query latency.
* **Colab A100 entrypoint**: `scripts/run_paper_experiments.py` and a
  `notebooks/colab_a100.ipynb` that downloads BEIR, runs the matrix, and
  plots Pareto fronts + a hypervolume table.
* Deterministic offline tests covering every module above (74 tests) plus
  CLI smoke tests; `mypy --strict`, `ruff`, `black`, `isort` clean.

What still belongs in the camera-ready (not in code, but in the writeup):

1. Per-dataset latency on the *deployment* hardware — re-profile on the
   target CPU/GPU and report against the same hypervolume reference.
2. Ablation: hold every objective fixed except one and trace how Pareto
   shifts — clarifies which compression axis dominates per backbone.
3. NSGA-II vs. enumeration runtime/quality trade-off plots — justifies the
   evolutionary search when the composition space gets large.

## CI

GitHub Actions runs `ruff`, `black --check`, `isort --check-only`,
`mypy --strict src`, and `pytest` on every push. All 74 tests run offline.
