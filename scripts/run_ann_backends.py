#!/usr/bin/env python3
r"""ANN-backend evaluation (RQ4) on faiss-cpu.

Builds the six FAISS-family backends on each pair's dense (identity) vectors and
writes results/<bb>__<ds>__index.csv in the schema regen_tables.t_ann_backends
reads, so tab:ann-backends populates. Robust: a backend that can't build on a
given corpus is recorded with status!=ok and skipped, never crashing the run.

Usage:
    python scripts/run_ann_backends.py --backbones e5-base bge-base mxbai-large gte-base \
        --datasets scifact nfcorpus arguana fiqa --cache-dir cache --output-dir results
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_catalog as rc

BACKENDS = ["exact-numpy", "faiss-flat", "faiss-ivf", "faiss-ivfpq", "faiss-hnsw", "faiss-opq"]

def _exact_topk(Q, C, k=10):
    return rc.topk_dense(Q, C, k)

def _ndcg(top, row2doc, qrels, qids, k=10):
    return float(rc.ndcg_per_query(top, row2doc, qrels, qids, k).mean())

def _recall(top, ref, k=10):
    out = []
    for g, r in zip(top, ref):
        rs = set(int(x) for x in r[:k]); 
        if not rs: continue
        out.append(len(set(int(x) for x in g[:k]) & rs) / len(rs))
    return float(np.mean(out)) if out else 0.0

def _bytes(index, faiss, fallback):
    try: return int(faiss.serialize_index(index).size)
    except Exception: return int(fallback)

def _median_ms(fn, repeats=10):
    ts = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1e3)
    ts.sort(); return ts[len(ts)//2], ts[min(len(ts)-1, int(0.95*len(ts)))]

def _build_search(backend, C, Q, k, faiss):
    n, d = C.shape
    if backend == "faiss-flat":
        idx = faiss.IndexFlatIP(d); idx.add(C)
    elif backend == "faiss-ivf":
        nlist = max(1, min(int(np.sqrt(n)), n)); quant = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(C); idx.add(C); idx.nprobe = max(1, nlist // 8)
    elif backend == "faiss-ivfpq":
        m = 8 if d % 8 == 0 else next(x for x in (4, 2, 1) if d % x == 0)
        nb = 8
        while nb > 1 and n < (1 << nb): nb -= 1
        nlist = max(1, min(int(np.sqrt(n)), n)); quant = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFPQ(quant, d, nlist, m, nb, faiss.METRIC_INNER_PRODUCT)
        idx.train(C); idx.add(C); idx.nprobe = max(1, nlist // 8)
    elif backend == "faiss-hnsw":
        idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200; idx.hnsw.efSearch = 64; idx.add(C)
    elif backend == "faiss-opq":
        if n < 256: raise RuntimeError("OPQ needs >=256 training vectors")
        m = 8 if d % 8 == 0 else next(x for x in (4, 2, 1) if d % x == 0)
        idx = faiss.index_factory(d, f"OPQ{m}_{d},PQ{m}x8", faiss.METRIC_INNER_PRODUCT)
        idx.train(C); idx.add(C)
    else:
        raise ValueError(backend)
    res = {}
    def _s(): res["S"], res["I"] = idx.search(Q, k)
    med, p95 = _median_ms(_s)
    try:
        nbytes = int(faiss.serialize_index(idx).size)
    except Exception:
        nbytes = int(C.nbytes)
    return res["I"].astype(np.int64), med, p95, nbytes

def run_pair(bb, ds, cache_dir, out_dir, k=10):
    npz = np.load(os.path.join(cache_dir, f"{bb}__{ds}.npz"))
    meta = json.load(open(os.path.join(cache_dir, f"{bb}__{ds}.json")))
    C = rc._normalize(npz["corpus"].astype(np.float32)); Q = rc._normalize(npz["queries"].astype(np.float32))
    qrels = meta["qrels"]; qids = meta["query_ids"]; row2doc = meta["doc_ids"]
    ref = _exact_topk(Q, C, k)
    rows = []
    try:
        import faiss
    except Exception:
        faiss = None
    # exact-numpy baseline
    rows.append(dict(backend="exact-numpy", status="ok", build_ms=0.0,
                     bytes=int(C.nbytes), lat=_median_ms(lambda: _exact_topk(Q, C, k))[0],
                     p95=0.0, recall=1.0, ndcg=_ndcg(ref, row2doc, qrels, qids, k), err=""))
    for be in BACKENDS[1:]:
        if faiss is None:
            rows.append(dict(backend=be, status="missing_backend", build_ms=0, bytes=0, lat=0, p95=0,
                             recall=0, ndcg=0, err="faiss not installed")); continue
        try:
            t0 = time.perf_counter()
            I, med, p95, nbytes = _build_search(be, np.ascontiguousarray(C), np.ascontiguousarray(Q), k, faiss)
            build_ms = (time.perf_counter() - t0) * 1e3
            rows.append(dict(backend=be, status="ok", build_ms=build_ms,
                             bytes=nbytes, lat=med, p95=p95,
                             recall=_recall(I, ref, k), ndcg=_ndcg(I, row2doc, qrels, qids, k), err=""))
        except Exception as e:
            rows.append(dict(backend=be, status="error", build_ms=0, bytes=0, lat=0, p95=0,
                             recall=0, ndcg=0, err=str(e)[:120]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{bb}__{ds}__index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone","dataset","spec","index_backend","index_status","index_build_ms",
                    "index_bytes","index_query_latency_ms","index_query_p95_ms","index_recall_at_10",
                    "index_ndcg_at_10","index_exact_recall_at_10","index_error"])
        for r in rows:
            w.writerow([bb, ds, "identity", r["backend"], r["status"], f"{r['build_ms']:.3f}",
                        r["bytes"], f"{r['lat']:.4f}", f"{r['p95']:.4f}", f"{r['recall']:.4f}",
                        f"{r['ndcg']:.4f}", f"{r['recall']:.4f}", r["err"]])
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"  {bb}/{ds}: {ok}/{len(rows)} backends ok")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--cache-dir", default="cache"); ap.add_argument("--output-dir", default="results")
    a = ap.parse_args()
    for bb in a.backbones:
        for ds in a.datasets:
            run_pair(bb, ds, a.cache_dir, a.output_dir)

if __name__ == "__main__":
    main()
