# Colab (GPU) Re-run Guide

Companion to `RECONCILIATION_AND_RERUN.md`. Same goal — regenerate the full artifact so
the paper reproduces — but as ordered Colab cells. Run top to bottom.

> **What the GPU buys you:** it accelerates the embedding/scoring path (sentence-transformers /
> torch), which is the dominant cost. The ANN backends use **faiss-cpu** (installed by the
> `index` extra) and run on CPU — that is expected and matches the paper's "FAISS-family" scope.
> Pick **A100 / High-RAM** (Colab Pro) if available: TREC-COVID (171k docs), FiQA (57k), and the
> seed-variance grid (×9 pairs) are the heavy parts.

---

## Cell 0 — GPU runtime
Runtime ▸ Change runtime type ▸ **GPU** (A100 preferred). Then:
```python
!nvidia-smi
```

## Cell 1 — Clone + install (this is the fix: the `index` extra was missing before)
```python
!git clone https://github.com/jorge-martinez-gil/embeddings.git
%cd embeddings
!pip -q install -e ".[paper,index]"
import faiss, torch
print("faiss", faiss.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
```

## Cell 2 — (Recommended) persist cache + results on Drive so a disconnect doesn't lose work
```python
from google.colab import drive; drive.mount('/content/drive')
import os
PERSIST='/content/drive/MyDrive/embedopt_run'
os.makedirs(f'{PERSIST}/results', exist_ok=True)
# symlink so the pipeline's caches/results land on Drive and survive timeouts
!rm -rf results && ln -s {PERSIST}/results results
# (the embedding cache is keyed by (backbone,dataset); point it at Drive too if the
#  code exposes a cache dir env var — otherwise re-embedding repeats on a fresh session)
```

## Cell 3 — Download BEIR datasets to /content (the `beir-local:` loader takes a path)
```python
!pip -q install beir
from beir import util
base='/content/datasets'; import os; os.makedirs(base, exist_ok=True)
URL='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip'
for d in ['scifact','nfcorpus','arguana','fiqa','trec-covid']:
    util.download_and_unzip(URL.format(d), base)
!ls /content/datasets
```
Dataset specs below are `beir-local:/content/datasets/<name>`.

## Cell 4 — Main sweep (all backbones × main datasets, all six backends, paper-grade resamples)
```python
!python scripts/run_paper_experiments.py \
  --backbones e5-base bge-base mxbai-large \
  --datasets beir-local:/content/datasets/scifact beir-local:/content/datasets/nfcorpus beir-local:/content/datasets/arguana \
  --index-backends exact-numpy faiss-flat faiss-ivf faiss-ivfpq faiss-hnsw faiss-opq \
  --bootstrap-resamples 5000 --significance-resamples 5000 \
  --profile-repeats 20 --score-device cuda --force \
  --output-dir results
```
*Resumable:* the runner checkpoints per spec and per (backbone,dataset). If the session drops,
re-run the same cell — completed pairs are skipped unless `--force`. For long jobs, consider
running one backbone per cell so partial progress is saved to Drive.

## Cell 5 — Robustness datasets (E5 only)
```python
!python scripts/run_paper_experiments.py \
  --backbones e5-base \
  --datasets beir-local:/content/datasets/fiqa beir-local:/content/datasets/trec-covid \
  --index-backends exact-numpy faiss-flat faiss-ivf faiss-ivfpq faiss-hnsw faiss-opq \
  --bootstrap-resamples 5000 --significance-resamples 5000 \
  --profile-repeats 20 --score-device cuda --force \
  --output-dir results
```

## Cell 6 — Seed-variance grid (PQ+OPQ) for ALL nine main pairs
```python
%%bash
for BB in e5-base bge-base mxbai-large; do
  for DS in scifact nfcorpus arguana; do
    python scripts/run_seed_variance.py --backbone $BB \
      --dataset beir-local:/content/datasets/$DS --seeds 0 1 2 3 4 --output-dir results
  done
done
```

## Cell 7 — Storage-mode comparison per dataset (so float16's "every pair" claim is backed)
```python
# run the storage-mode driver once per dataset; check its --help for the dataset flag name
!python scripts/compare_storage_modes.py --help
```

## Cell 8 — Coverage sanity check (must pass before regenerating tables)
```python
import csv, glob, os
from collections import Counter
for f in sorted(glob.glob("results/*__candidates.csv")):
    s=[r['spec'] for r in csv.DictReader(open(f))]
    print(os.path.basename(f), len(s),
          "float16" if any('float16' in x for x in s) else "NO-float16",
          "opq" if any('opq' in x.lower() for x in s) else "NO-opq")
for f in sorted(glob.glob("results/*__index.csv")):
    ok=Counter(r['index_backend'] for r in csv.DictReader(open(f)) if r['index_status']=='ok')
    print(os.path.basename(f), dict(ok))   # expect ok rows for all six backends
```

## Cell 9 — Regenerate the whole empirical section (tables + TOST + figures)
```python
!python scripts/regen_tables.py --emit-tex results/paper_tables.tex
!python scripts/tost_equivalence.py --results results --spec scalar_int8 --eps 0.01 --eps2 0.005 --out results/tost_scalar_int8.csv
!python scripts/tost_equivalence.py --results results --spec float16     --eps 0.01 --eps2 0.005 --out results/tost_float16.csv
!python scripts/plot_paper_figures.py --results-dir results --output-dir figures
```
Then in the manuscript: `\input{results/paper_tables.tex}` (or paste each block), and recompile.

## Cell 10 — Download everything (if not already on Drive)
```python
!zip -qr artifact_regen.zip results figures
from google.colab import files; files.download('artifact_regen.zip')
```

---

### Watch-outs specific to Colab
- **Session timeouts.** Free Colab caps wall-clock (~12 h) and idles out. Use Drive persistence
  (Cell 2) and run heavy pairs in separate cells so a drop never loses a completed pair.
- **Re-embedding cost.** A fresh VM loses the embedding cache unless it's on Drive; the embed pass
  is the expensive step, so persisting it makes re-runs cheap.
- **faiss-gpu is not needed.** The `index` extra's faiss-cpu is what the backends use; installing
  faiss-gpu can conflict with the Colab CUDA/torch build — avoid it.
- **HV reference point.** After regen, confirm `plot_paper_figures.py` and the runner use the same
  hypervolume reference point the paper text assumes (the committed HV differed ~2.7×).
- **Then follow** `RECONCILIATION_AND_RERUN.md` §5 verification checklist before submitting.
