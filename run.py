# 3.0  Clean environment with BOTH extras (this is the part that was missing: ".[...,index]")
python -m pip install -e ".[paper,index]"
python -c "import faiss, torch; print('faiss', faiss.__version__, '| cuda', torch.cuda.is_available())"

# 3.1  Main sweep: all backbones × main datasets, all six backends, paper-grade resamples.
#      (Re-run with current code so float16 + OPQ are included in the spec set.)
python scripts/run_paper_experiments.py \
  --backbones e5-base bge-base mxbai-large \
  --datasets beir-local:data/scifact beir-local:data/nfcorpus beir-local:data/arguana \
  --index-backends exact-numpy faiss-flat faiss-ivf faiss-ivfpq faiss-hnsw faiss-opq \
  --bootstrap-resamples 5000 --significance-resamples 5000 \
  --profile-repeats 20 --score-device cuda --force \
  --output-dir results

# 3.2  Robustness datasets (E5 only), same flags.
python scripts/run_paper_experiments.py \
  --backbones e5-base \
  --datasets beir-local:data/fiqa beir-local:data/trec-covid \
  --index-backends exact-numpy faiss-flat faiss-ivf faiss-ivfpq faiss-hnsw faiss-opq \
  --bootstrap-resamples 5000 --significance-resamples 5000 \
  --profile-repeats 20 --score-device cuda --force \
  --output-dir results

# 3.3  Seed-variance grid (PQ+OPQ) for EVERY (backbone,dataset) the paper generalizes over,
#      not just e5/scifact. Repeat per pair:
for BB in e5-base bge-base mxbai-large; do
  for DS in scifact nfcorpus arguana; do
    python scripts/run_seed_variance.py --backbone $BB \
      --dataset beir-local:data/$DS --seeds 0 1 2 3 4 --output-dir results
  done
done

# 3.4  Storage-mode comparison per dataset (so float16's "every pair" claim is backed),
#      not a single corpus. Use the storage-mode driver / CLI per dataset.
#      (compare_storage_modes.py — pass each dataset; see its --help for the exact flag.)