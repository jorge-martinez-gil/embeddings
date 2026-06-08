#!/usr/bin/env python3
r"""Codebook seed-variance (RQ5) for the cheap workflow.

Retrains PQ and OPQ at several seeds on cached embeddings and writes the schema
regen_tables.py / the paper expect:

    results/<bb>__<ds>__seed_variance.csv          (spec_label,seed,ndcg_at_10,...)
    results/<bb>__<ds>__seed_variance_summary.csv  (spec_label,n_seeds,ndcg_mean,ndcg_std,...)

Reuses helpers from run_catalog.py (same scripts/ dir). CPU only (faiss-cpu).

Usage:
    python scripts/run_seed_variance_cheap.py --backbones e5-base bge-base mxbai-large \
        --datasets scifact nfcorpus arguana --seeds 0 1 2 3 4 --cache-dir cache --output-dir results
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_catalog as rc  # noqa: E402
import json

def _set_pq_seed(index, seed):
    import faiss
    for getter in (lambda: index.pq,
                   lambda: faiss.downcast_index(index.index).pq):  # OPQ pretransform
        try:
            getter().cp.seed = int(seed); return True
        except Exception:
            continue
    return False

def seeded_topk(C, Q, factory, seed, k=10):
    import faiss
    d = C.shape[1]
    index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
    _set_pq_seed(index, seed)
    index.train(np.ascontiguousarray(C)); index.add(np.ascontiguousarray(C))
    _, I = index.search(np.ascontiguousarray(Q), k)
    return I.astype(np.int64)

def pqopq_specs(d, n):
    for M in [4, 8, 16, 32, 64]:
        if d % M: continue
        for b in [4, 6, 8]:
            yield (f"product_quantize(n_subspaces={M},n_bits={b})", f"PQ{M}x{b}")
    for M in [4, 8, 16, 32, 64]:
        if d % M or n < 256: continue
        for b in [4, 8]:
            yield (f"opq(n_subspaces={M},n_bits={b})", f"OPQ{M}_{d},PQ{M}x{b}")

def run_pair(bb, ds, cache_dir, out_dir, seeds, k=10):
    npz = np.load(os.path.join(cache_dir, f"{bb}__{ds}.npz"))
    meta = json.load(open(os.path.join(cache_dir, f"{bb}__{ds}.json")))
    C = rc._normalize(npz["corpus"].astype(np.float32)); Q = rc._normalize(npz["queries"].astype(np.float32))
    n, d = C.shape; qrels = meta["qrels"]; qids = meta["query_ids"]; row2doc = meta["doc_ids"]
    per_seed = []; summary = []
    for label, factory in pqopq_specs(d, n):
        vals = []
        for s in seeds:
            try:
                idx = seeded_topk(C, Q, factory, s, k)
            except Exception as e:
                print(f"  skip {label} seed {s}: {e}"); continue
            nd = float(rc.ndcg_per_query(idx, row2doc, qrels, qids, k).mean())
            vals.append(nd); per_seed.append((label, s, nd))
        if vals:
            summary.append((label, len(vals), float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{bb}__{ds}")
    with open(base + "__seed_variance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["spec_label", "seed", "ndcg_at_10", "recall_at_10", "mrr_at_10", "map_at_10"])
        for lab, s, nd in per_seed: w.writerow([lab, s, f"{nd:.6f}", 0.0, 0.0, 0.0])
    with open(base + "__seed_variance_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spec_label", "n_seeds", "ndcg_mean", "ndcg_std",
                    "recall_mean", "recall_std", "mrr_mean", "mrr_std", "map_mean", "map_std"])
        for lab, ns, mean, std in summary:
            w.writerow([lab, ns, f"{mean:.6f}", f"{std:.6f}", 0, 0, 0, 0, 0, 0])
    print(f"  {bb}/{ds}: {len(summary)} specs × {len(seeds)} seeds")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--output-dir", default="results")
    a = ap.parse_args()
    for bb in a.backbones:
        for ds in a.datasets:
            run_pair(bb, ds, a.cache_dir, a.output_dir, a.seeds)

if __name__ == "__main__":
    main()
