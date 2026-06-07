# Artifact ↔ Paper Reconciliation and Re-run Playbook

**Decision:** the manuscript is canonical. The committed `results/` is a **partial run**
and must be regenerated so the artifact reproduces the paper. This document records the
gap, the root cause, the exact re-run commands, and how to regenerate every table, figure,
and the TOST analysis from the fresh run.

---

## 1. What's wrong with the committed `results/`

| Paper claim | Backed by committed artifact? | Evidence |
|---|---|---|
| 73-spec sweep incl. **float16** + **OPQ** | **No** — 62 specs, no float16, no OPQ | `candidates.csv` families: identity 1, truncate 5, int8 1, binary 1, PQ 15, composed 39 = 62 |
| RQ4 six FAISS backends (`fig:idx`) | **No** — every FAISS backend failed | `*__index.csv`: `faiss-*` rows all `missing_backend`; only 7 `exact-numpy` rows `ok` |
| RQ5 seed variance for PQ **and OPQ** | **Partly** — OPQ present but **e5/scifact only** | only `e5-base__scifact__seed_variance*.csv` committed |
| float16 ≈ 2× near-lossless on every pair | **Partly** — one corpus only | `storage_modes_comparison.csv` has float16 on a single corpus (Δ = −0.00123) |
| Table 4 / Pareto / threshold numbers | **No** — different run | int8 p-values & HV differ from the paper (e.g. HV off ≈2.7×; MXBAI p-values differ) |
| Main sweep int8/PQ/binary/composed, 9+2 pairs | **Yes** | 11 `candidates.csv` present, full dataset sizes in manifests |

**Net:** the only solid part is the main compression sweep (62 specs). RQ4 is empty; OPQ,
float16, and the full seed-variance grid are missing.

## 2. Root cause

Two issues in the run that produced the committed CSVs:

1. **FAISS was not installed.** The runner was called with all six `--index-backends`,
   but `faiss-cpu` was absent (the `index` extra wasn't installed), so every FAISS index
   returned `missing_backend`. (Confirmed: `faiss-cpu>=1.8.0` *is* declared in
   `pyproject.toml` under the `index` extra — it just wasn't installed in that environment.)
2. **The sweep predates float16/OPQ in the operator set.** The committed manifest's `specs`
   list contains neither, although the current code (README feature table) sweeps both.

Both are environment/version problems, not code problems. A clean re-run fixes them.

## 3. Re-run the full experiments (the fix)

```bash
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
```

**After 3.1–3.4, sanity-check coverage before regenerating anything:**

```bash
# every candidates.csv should now contain float16 + OPQ and ~73 specs
python - <<'PY'
import csv, glob, os
for f in sorted(glob.glob("results/*__candidates.csv")):
    s=[r['spec'] for r in csv.DictReader(open(f))]
    has16=any('float16' in x for x in s); hasopq=any('opq' in x.lower() for x in s)
    print(os.path.basename(f), len(s), "float16" if has16 else "NO-float16", "opq" if hasopq else "NO-opq")
# index sweep should now have ok FAISS rows
for f in sorted(glob.glob("results/*__index.csv")):
    from collections import Counter
    ok=Counter(r['index_backend'] for r in csv.DictReader(open(f)) if r['index_status']=='ok')
    print(os.path.basename(f), dict(ok))
PY
```

## 4. Regenerate tables, figures, and TOST from the fresh run

```bash
# 4.1  Paper tables (Table: scalar-int8, main-pareto, thresholds, robustness) — prints LaTeX rows.
python scripts/regen_tables.py            # reads results/, emits LaTeX-ready rows + sanity facts

# 4.2  Equivalence (TOST) table for the near-lossless codecs.
python scripts/tost_equivalence.py --results results --spec scalar_int8 --eps 0.01 --eps2 0.005 \
  --out results/tost_scalar_int8.csv
python scripts/tost_equivalence.py --results results --spec float16 --eps 0.01 --eps2 0.005 \
  --out results/tost_float16.csv          # now that float16 is swept

# 4.3  Figures (regenerates fig_pareto, fig_pair_pareto_frontiers, fig_hypervolume,
#       fig_index_tradeoff, fig_storage_modes_bar, fig_pq_ablation_heatmap, fig_latency_storage_quality).
python scripts/plot_paper_figures.py --results-dir results --output-dir figures
```

Then paste the regenerated rows into the corresponding tables and recompile the `.tex`.
`regen_tables.py` and `tost_equivalence.py` automatically pick up the float16/OPQ rows once
they exist, so no manual transcription of numbers is needed.

## 5. Post-regeneration verification checklist

- [ ] Every `candidates.csv` has float16 + OPQ and the spec count the paper states.
- [ ] `*__index.csv` has `ok` rows for all six backends on the 9 main pairs (RQ4 / `fig:idx`).
- [ ] `seed_variance_summary.csv` exists for all nine main pairs (RQ5 generalization claim).
- [ ] Table 4, the Pareto table, the threshold table, and the robustness table match
      `regen_tables.py` output exactly.
- [ ] The TOST table (`tab:tost`) matches `tost_equivalence.py` output for int8 **and** float16.
- [ ] Figures in `figures/` are regenerated from the same run (check timestamps; delete the
      `outputs/smoke-figures/` placeholders so no smoke figure is referenced).
- [ ] `summary.json` and every `manifest.json` carry the new `config_hash`; commit them.
- [ ] One spec row → one manifest → one CSV row (the paper's traceability claim) still holds.

## 6. Notes / decisions still open

- **Spec count.** Confirm the regenerated sweep total (the paper says 73). If the operator
  grid changed, update the "73 specifications" sentences in Sections 4 and 6 to the new count.
- **HV reference point.** The committed HV values differ ~2.7× from the paper. Hypervolume is
  reference-point dependent; ensure `plot_paper_figures.py`/the runner use the *same* reference
  point the paper text assumes, or update the HV numbers to the regenerated ones.
- **Worked example (Section: advisor).** Scenario C cites PQ(32,8) ΔnDCG=−0.07; the committed
  run gives −0.0613 for e5/scifact. Re-read this value from the regenerated e5/scifact CSV.
