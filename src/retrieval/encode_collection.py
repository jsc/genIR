"""
Encode the full 8,933,787-passage collection with the fine-tuned dense retriever, split
across both GPUs for speed, and build a FAISS index (inner product over L2-normalized
vectors == cosine similarity) for fast nearest-neighbor retrieval.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = ROOT / "train" / "genIR.collection.tsv"
MODEL_PATH = str(Path(__file__).resolve().parent / "dense_retriever_model")
OUT = Path(__file__).resolve().parent
BATCH_SIZE = 512


def main():
    shard = sys.argv[1]  # "0" or "1"
    n_shards = 2
    device = f"cuda:{shard}"
    shard = int(shard)

    print(f"[shard {shard}] loading model on {device}", flush=True)
    model = SentenceTransformer(MODEL_PATH, device=device)

    pids, texts = [], []
    with open(COLLECTION, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i % n_shards != shard:
                continue
            if not row or len(row) < 2:
                continue
            pids.append(int(row[0]))
            texts.append(row[1])

    print(f"[shard {shard}] encoding {len(texts):,} passages", flush=True)
    t0 = time.time()
    embs = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True,
        normalize_embeddings=True, device=device,
    )
    print(f"[shard {shard}] done in {time.time() - t0:.0f}s", flush=True)

    np.save(OUT / f"collection_emb_shard{shard}.npy", embs.astype(np.float32))
    np.save(OUT / f"collection_pids_shard{shard}.npy", np.array(pids, dtype=np.int64))
    print(f"[shard {shard}] saved", flush=True)


if __name__ == "__main__":
    main()
