"""Combine the two encoded shards into one FAISS index (inner product over L2-normalized
vectors == cosine similarity) for dense retrieval."""
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    t0 = time.time()

    # 8.9M x 768 x 4 bytes ~= 27GB total. Concatenating shards first and then calling
    # index.add() briefly holds THREE full-size copies at once (raw shards + concatenated
    # array + FAISS's internal copy) -- that's what OOM-killed the previous run on this
    # 62GB-RAM box. Add each shard directly instead, freeing it before touching the next,
    # so peak usage stays ~1 shard's worth (~14GB) plus FAISS's copy of it.
    index = None
    pid_parts = []
    for shard in (0, 1):
        embs = np.load(ROOT / f"collection_emb_shard{shard}.npy")
        pids = np.load(ROOT / f"collection_pids_shard{shard}.npy")
        if index is None:
            index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        pid_parts.append(pids)
        print(f"shard {shard}: added {embs.shape[0]:,} vectors "
              f"(index total now {index.ntotal:,})", flush=True)
        del embs

    pids = np.concatenate(pid_parts, axis=0)
    print(f"CPU index built with {index.ntotal:,} vectors in {time.time() - t0:.0f}s", flush=True)

    faiss.write_index(index, str(ROOT / "faiss_index.bin"))
    np.save(ROOT / "faiss_pids.npy", pids)
    print(f"Saved index + pid mapping. Total time {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
