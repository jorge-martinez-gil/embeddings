#!/usr/bin/env python3
r"""SINGLE entry point for the entire EmbedCatalog 4x4 reproduction.

One command runs the whole pipeline end to end:
    [0] GPU self-test  [1] download  [2] embed once  [3] catalog sweep
    [4] seed variance  [5] ANN backends (RQ4)  [6] tables  [7] TOST

Device is auto-detected: CUDA present -> pure-torch GPU path; else CPU path.
The ANN-backend stage always uses faiss-cpu (reliable on CPU). Every stage is
resumable -- finished pairs are skipped -- so a disconnect just means re-running.

Run it (same command on A100 / free T4 / laptop):
    python scripts/run_all.py

Useful flags:
    --cpu / --gpu            force a device path
    --skip-download          datasets already in --data-dir
    --skip-ann               skip RQ4 (no faiss needed then)
    --no-selftest            skip the GPU self-test gate
    --dry-run                print the plan, run nothing
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_BB = ["e5-base", "bge-base", "mxbai-large", "gte-base"]
DEF_DS = ["scifact", "nfcorpus", "arguana", "fiqa"]
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

def have_gpu():
    try:
        import torch; return bool(torch.cuda.is_available())
    except Exception:
        return False

def banner(m): print("\n" + "=" * 72 + f"\n== {m}\n" + "=" * 72, flush=True)

def run(cmd, dry):
    print("$ " + " ".join(cmd), flush=True)
    if dry: return
    t0 = time.time(); r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"stage failed (exit {r.returncode}): {' '.join(cmd)}")
    print(f"  [{time.time()-t0:.1f}s]", flush=True)

def download(datasets, data_dir, dry):
    banner("STAGE 1/7  download BEIR datasets")
    if dry:
        for d in datasets: print(f"$ download {BEIR_URL.format(d)} -> {data_dir}/{d}")
        return
    os.makedirs(data_dir, exist_ok=True)
    from beir import util
    for d in datasets:
        if os.path.isdir(os.path.join(data_dir, d)): print(f"present: {d}"); continue
        util.download_and_unzip(BEIR_URL.format(d), data_dir)

def main():
    ap = argparse.ArgumentParser(description="Single entry point for the 4x4 reproduction.")
    ap.add_argument("--backbones", nargs="+", default=DEF_BB)
    ap.add_argument("--datasets", nargs="+", default=DEF_DS)
    ap.add_argument("--data-dir", default="/content/datasets")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--bootstrap-resamples", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=512)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true"); g.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-ann", action="store_true")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    gpu = True if a.gpu else False if a.cpu else have_gpu()
    py = sys.executable
    S = lambda name: os.path.join(HERE, name)
    print(f"device path: {'GPU (torch)' if gpu else 'CPU (numpy+faiss)'} | "
          f"backbones={a.backbones} | datasets={a.datasets}")

    if gpu and not a.no_selftest:
        banner("STAGE 0/7  GPU self-test (abort if it fails)")
        run([py, S("run_catalog_gpu.py"), "--selftest"], a.dry_run)

    if a.skip_download:
        banner("STAGE 1/7  download (skipped)")
    else:
        download(a.datasets, a.data_dir, a.dry_run)

    banner("STAGE 2/7  embed once")
    run([py, S("embed_once.py"), "--backbones", *a.backbones, "--datasets", *a.datasets,
         "--data-dir", a.data_dir, "--cache-dir", a.cache_dir, "--batch-size", str(a.batch_size)], a.dry_run)

    banner("STAGE 3/7  catalog sweep")
    run([py, S("run_catalog_gpu.py" if gpu else "run_catalog.py"),
         "--backbones", *a.backbones, "--datasets", *a.datasets,
         "--cache-dir", a.cache_dir, "--output-dir", a.output_dir,
         "--bootstrap-resamples", str(a.bootstrap_resamples)], a.dry_run)

    banner("STAGE 4/7  seed variance (RQ5)")
    run([py, S("run_seed_variance_gpu.py" if gpu else "run_seed_variance_cheap.py"),
         "--backbones", *a.backbones, "--datasets", *a.datasets,
         "--seeds", *map(str, a.seeds), "--cache-dir", a.cache_dir, "--output-dir", a.output_dir], a.dry_run)

    if a.skip_ann:
        banner("STAGE 5/7  ANN backends (skipped)")
    else:
        banner("STAGE 5/7  ANN backends (RQ4, faiss-cpu)")
        run([py, S("run_ann_backends.py"), "--backbones", *a.backbones, "--datasets", *a.datasets,
             "--cache-dir", a.cache_dir, "--output-dir", a.output_dir], a.dry_run)

    banner("STAGE 6/7  regenerate tables")
    run([py, S("regen_tables.py"), "--emit-tex", os.path.join(a.output_dir, "paper_tables.tex")], a.dry_run)

    banner("STAGE 7/7  TOST equivalence (int8 + float16)")
    for spec, out in (("scalar_int8", "tost_scalar_int8.csv"), ("float16", "tost_float16.csv")):
        run([py, S("tost_equivalence.py"), "--results", a.output_dir, "--spec", spec,
             "--eps", "0.01", "--eps2", "0.005", "--out", os.path.join(a.output_dir, out)], a.dry_run)

    banner("DONE")
    print(f"Tables -> {a.output_dir}/paper_tables.tex   (\\input it in the manuscript)")
    print(f"TOST   -> {a.output_dir}/tost_scalar_int8.csv, tost_float16.csv")

if __name__ == "__main__":
    main()
