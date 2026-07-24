"""
Build a ranker-augmented human-vs-LLM classification dataset, addressing a real limitation
of build_dataset.py's construction: that pipeline's "human" class is just qrels (the ONE
verified-relevant passage per query), so train/val/test end up with only ~1.2-2.0 human
examples per query against ~46 generated ones per query -- a ~97-98% generated "class
imbalance" that is an artifact of how we sampled the human class, NOT a real property of
the collection. The original MS MARCO passage collection was built by pooling the top-100
ranked results per query (one of which became the qrels judgment) -- every query has roughly
100 human passages sitting in the collection, not 1.

This script recovers many more of them, honestly: run BM25 (the same ranker used
elsewhere in this project) for every query, walk its top-20 results, and check each
retrieved pid against the union of ALL THREE gen.*.tsv files (which the dataset README
states contain "the passage IDs for ALL of the generated passages" -- so any retrieved pid
NOT in that union is guaranteed human, regardless of which query it was retrieved for).
This also captures genuine hard negatives: these are real passages a retrieval model
considers topically relevant to the query, not random draws (the bug found in
triples.train.tsv) and not limited to the single qrels judgment.

The existing qrels-derived human pids and gen-derived generated pids are kept and unioned
with the new ranker-derived human pids -- this only ENLARGES the human class, it doesn't
change what "generated" means at all.

Cross-split dedup priority is reversed from build_dataset.py: ranker walks are far more
likely to produce cross-split collisions (a popular real passage can rank highly for many
different queries) than qrels ever did, so protect the small, precious eval splits first --
process test, then val, then train, dropping conflicts from whichever split is LATER in that
order (i.e. trim train, which has abundant supply, before ever trimming val/test).
"""
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import pyterrier as pt

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train"
OUT = ROOT / "src" / "data_ranker_augmented"
OUT.mkdir(parents=True, exist_ok=True)

COLLECTION = TRAIN / "genIR.collection.tsv"
INDEX_PATH = str(ROOT / "src" / "retrieval" / "terrier_index")
TOP_K = 20


def read_pid_col(path, col):
    pids = set()
    with open(path, newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row:
                continue
            pids.add(int(row[col]))
    return pids


def simple_tokenize_query(q):
    import re
    return " ".join(re.findall(r"[a-zA-Z0-9]+", q))


def load_queries(split):
    return pd.read_csv(TRAIN / f"queries.{split}.tsv", sep="\t", header=None, names=["qid", "query"])


def build_gen_and_qrels_labels():
    """Same as build_dataset.py's build_label_map, kept unchanged: qrels -> human (0),
    gen -> generated (1), per split."""
    labels = {"train": {}, "val": {}, "test": {}}
    qids = {"train": {}, "val": {}, "test": {}}

    with open(TRAIN / "qrels.train.tsv", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            qid, pid = int(row[0]), int(row[2])
            labels["train"][pid] = 0
            qids["train"][pid] = qid
    with open(TRAIN / "gen.train.tsv", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            qid, pid = int(row[0]), int(row[1])
            labels["train"][pid] = 1
            qids["train"][pid] = qid

    with open(TRAIN / "qrels.val.tsv", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            qid, pid = int(row[0]), int(row[2])
            labels["val"][pid] = 0
            qids["val"][pid] = qid
    with open(TRAIN / "gen.val.tsv", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            qid, pid = int(row[0]), int(row[1])
            labels["val"][pid] = 1
            qids["val"][pid] = qid

    with open(TEST / "qrels.test.tsv", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            qid, pid = int(row[0]), int(row[2])
            labels["test"][pid] = 0
            qids["test"][pid] = qid
    with open(TEST / "gen.test.tsv", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            qid, pid = int(row[0]), int(row[1])
            labels["test"][pid] = 1
            qids["test"][pid] = qid

    return labels, qids


def add_ranker_walk_human_pids(labels, qids, global_gen_pids, bm25):
    for split in ("train", "val", "test"):
        queries = load_queries(split)
        pt_queries = pd.DataFrame({
            "qid": queries.qid.astype(str),
            "query": queries["query"].apply(simple_tokenize_query),
        })
        t0 = time.time()
        results = bm25.transform(pt_queries)
        n_added = 0
        for qid_str, group in results.groupby("qid"):
            qid = int(qid_str)
            for pid in group.docno.astype(int):
                if pid in global_gen_pids:
                    continue  # already accounted for as generated, elsewhere
                if pid in labels[split]:
                    continue  # already labeled (qrels human, or -- shouldn't happen -- generated)
                labels[split][pid] = 0
                qids[split][pid] = qid
                n_added += 1
        print(f"  [{split}] BM25 top-{TOP_K} for {len(queries)} queries in "
              f"{time.time() - t0:.1f}s -> +{n_added} new human pids", file=sys.stderr)


def dedupe_across_splits_protect_eval(labels, qids):
    """Protect test, then val, then train: drop conflicts from whichever split is LATER
    in that priority order (opposite of build_dataset.py, which protected train)."""
    seen = set()
    for split in ("test", "val", "train"):
        dup = seen & labels[split].keys()
        for pid in dup:
            del labels[split][pid]
            del qids[split][pid]
        if dup:
            print(f"  dropped {len(dup)} pids from {split} (seen in earlier-protected split)",
                  file=sys.stderr)
        seen |= labels[split].keys()


def main():
    print("Loading BM25 index...", file=sys.stderr)
    index = pt.IndexFactory.of(INDEX_PATH)
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=TOP_K)

    print("Building global gen-pid set (union across all three splits)...", file=sys.stderr)
    global_gen_pids = (read_pid_col(TRAIN / "gen.train.tsv", 1)
                        | read_pid_col(TRAIN / "gen.val.tsv", 1)
                        | read_pid_col(TEST / "gen.test.tsv", 1))
    print(f"  {len(global_gen_pids):,} distinct generated pids total", file=sys.stderr)

    labels, qids = build_gen_and_qrels_labels()
    print("Before ranker walk:", file=sys.stderr)
    for split in labels:
        n_pos = sum(1 for v in labels[split].values() if v == 1)
        n_neg = sum(1 for v in labels[split].values() if v == 0)
        print(f"  {split}: {n_neg} human, {n_pos} generated", file=sys.stderr)

    print("\nWalking BM25 top-%d per query..." % TOP_K, file=sys.stderr)
    add_ranker_walk_human_pids(labels, qids, global_gen_pids, bm25)

    dedupe_across_splits_protect_eval(labels, qids)

    print("\nAfter ranker walk + dedup:", file=sys.stderr)
    for split in labels:
        n_pos = sum(1 for v in labels[split].values() if v == 1)
        n_neg = sum(1 for v in labels[split].values() if v == 0)
        print(f"  {split}: {n_neg} human, {n_pos} generated "
              f"({100 * n_neg / (n_neg + n_pos):.1f}% human)", file=sys.stderr)

    all_needed = set()
    for split in labels:
        all_needed |= set(labels[split].keys())
    print(f"\nTotal distinct pids needed: {len(all_needed):,}", file=sys.stderr)

    pid_to_text = {}
    with open(COLLECTION, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i in all_needed:
                pid_to_text[i] = row[1]
            if len(pid_to_text) == len(all_needed):
                break
    print(f"Recovered text for {len(pid_to_text):,}/{len(all_needed):,} pids", file=sys.stderr)

    missing_total = 0
    for split in labels:
        out_path = OUT / f"{split}.tsv"
        n_written = 0
        with open(out_path, "w", newline="") as out:
            writer = csv.writer(out, delimiter="\t")
            writer.writerow(["qid", "pid", "label", "text"])
            for pid, label in labels[split].items():
                text = pid_to_text.get(pid)
                if text is None:
                    missing_total += 1
                    continue
                writer.writerow([qids[split][pid], pid, label, text])
                n_written += 1
        print(f"Wrote {n_written:,} rows to {out_path}", file=sys.stderr)
    print(f"Missing text for {missing_total} pids total", file=sys.stderr)


if __name__ == "__main__":
    main()
