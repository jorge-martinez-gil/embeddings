#!/usr/bin/env python3
r"""Embed each (backbone, dataset) pair ONCE and cache vectors. A100-tuned.

Only GPU step in the workflow. Uses fp16 + large batches on CUDA for speed.

Caches per pair:
    cache/<backbone>__<dataset>.npz   -> corpus (n,d) f32, queries (m,d) f32
    cache/<backbone>__<dataset>.json  -> {"query_ids","doc_ids","qrels"}

Usage:
    python scripts/embed_once.py --backbones e5-base bge-base mxbai-large gte-base \
        --datasets scifact nfcorpus arguana fiqa --data-dir /content/datasets --cache-dir cache
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

# backbone -> (hf_model_id, query_prefix, passage_prefix)
BACKBONES = {
    "e5-base":     ("intfloat/e5-base-v2", "query: ", "passage: "),
    "bge-base":    ("BAAI/bge-base-en-v1.5",
                    "Represent this sentence for searching relevant passages: ", ""),
    "mxbai-large": ("mixedbread-ai/mxbai-embed-large-v1",
                    "Represent this sentence for searching relevant passages: ", ""),
    "gte-base":    ("thenlper/gte-base", "", ""),   # GTE: no query/passage prefix
}

def load_beir(path):
    from beir.datasets.data_loader import GenericDataLoader
    for split in ("test", "dev", "train"):
        if os.path.exists(os.path.join(path, "qrels", f"{split}.tsv")):
            return GenericDataLoader(data_folder=path).load(split=split)
    raise FileNotFoundError(f"no qrels/*.tsv under {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data-dir", default="/content/datasets")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--batch-size", type=int, default=512)   # large batch for A100
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.cache_dir, exist_ok=True)

    for bb in a.backbones:
        if bb not in BACKBONES:
            print(f"!! unknown backbone {bb}; known: {list(BACKBONES)}"); continue
        model_id, qpref, ppref = BACKBONES[bb]
        model = None
        for ds in a.datasets:
            out_npz = os.path.join(a.cache_dir, f"{bb}__{ds}.npz")
            if os.path.exists(out_npz) and not a.force:
                print(f"cached: {bb}/{ds}"); continue
            corpus, queries, qrels = load_beir(os.path.join(a.data_dir, ds))
            if model is None:
                print(f"loading {model_id} on {device} (fp16={a.fp16 and device=='cuda'}) ...")
                model = SentenceTransformer(model_id, device=device,
                                            trust_remote_code=True)
                if a.fp16 and device == "cuda":
                    model = model.half()
            doc_ids = list(corpus.keys()); query_ids = list(queries.keys())
            doc_texts = [((corpus[i].get("title", "") + " " + corpus[i].get("text", "")).strip())
                         for i in doc_ids]
            q_texts = [queries[i] for i in query_ids]
            print(f"embedding {bb}/{ds}: {len(doc_ids)} docs, {len(query_ids)} queries")
            enc = dict(batch_size=a.batch_size, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)
            C = model.encode([ppref + t for t in doc_texts], **enc).astype(np.float32)
            Q = model.encode([qpref + t for t in q_texts], **enc).astype(np.float32)
            np.savez(out_npz, corpus=C, queries=Q)
            json.dump({"query_ids": [str(x) for x in query_ids],
                       "doc_ids": [str(x) for x in doc_ids],
                       "qrels": {str(q): {str(d): int(r) for d, r in v.items()}
                                 for q, v in qrels.items()}},
                      open(os.path.join(a.cache_dir, f"{bb}__{ds}.json"), "w"))
            print(f"  cached -> {out_npz}  (d={C.shape[1]})")

if __name__ == "__main__":
    main()
