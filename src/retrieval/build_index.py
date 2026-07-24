"""
Build a Terrier (BM25) index over the FULL 8,933,787-passage genIR collection, so retrieval
for a query surfaces realistic top-k results from the whole collection (not just the small
known set of relevant/generated pids for that query) -- this is what makes the MRR-degradation
question meaningful: do LLM-generated passages actually get retrieved and crowd out the true
relevant (human) passage in a realistic ranked list?
"""
import csv
import sys
import time
from pathlib import Path

import pyterrier as pt

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = ROOT / "train" / "genIR.collection.tsv"
INDEX_PATH = str(ROOT / "src" / "retrieval" / "terrier_index")


def doc_iter():
    t0 = time.time()
    with open(COLLECTION, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if not row or len(row) < 2:
                continue
            yield {"docno": row[0], "text": row[1]}
            if i % 500_000 == 0:
                print(f"  indexed {i:,} docs so far ({time.time() - t0:.0f}s elapsed)", flush=True)


def main():
    t0 = time.time()
    print("Building Terrier index over the full collection...", flush=True)
    indexer = pt.IterDictIndexer(
        INDEX_PATH, meta={"docno": 12}, threads=8, overwrite=True,
    )
    index_ref = indexer.index(doc_iter())
    print(f"Indexing done in {time.time() - t0:.0f}s. Index at {INDEX_PATH}", flush=True)

    index = pt.IndexFactory.of(index_ref)
    print(index.getCollectionStatistics().toString(), flush=True)


if __name__ == "__main__":
    main()
