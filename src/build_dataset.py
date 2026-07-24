"""
Build human-vs-LLM passage classification datasets from the GenIR collection.

Label convention: 1 = LLM-generated, 0 = human-written (real).

Sources of pids:
  - train split: train/triples.train.tsv (qid, real_pid, gen_pid) -> the canonical paired
    training set. real_pid -> label 0, gen_pid -> label 1.
  - val split:   train/qrels.val.tsv (real, label 0) + train/gen.val.tsv (generated, label 1)
  - test split:  test/qrels.test.tsv (real, label 0) + test/gen.test.tsv (generated, label 1)

Passage text is pulled out of train/genIR.collection.tsv by exploiting the fact that the
file is sorted by pid and pid == line index (0-based), so we can do a single linear pass
collecting only the pids we need instead of loading 8.9M lines into memory.
"""
import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train"
OUT = ROOT / "src" / "data"
OUT.mkdir(parents=True, exist_ok=True)

COLLECTION = TRAIN / "genIR.collection.tsv"


def read_pid_col(path, col):
    pids = set()
    with open(path, newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row:
                continue
            pids.add(int(row[col]))
    return pids


def build_label_map():
    """Returns dict split -> dict pid -> label, plus dict pid -> qid for grouping."""
    labels = {"train": {}, "val": {}, "test": {}}
    qids = {"train": {}, "val": {}, "test": {}}

    # train: build from qrels + gen, same as val/test. NOTE: triples.train.tsv's "real"
    # column is NOT the query-relevant passage (matches qrels-verified relevant pid only
    # 1.36% of the time, and shows no more query-term overlap than a random passage draw
    # from the whole collection) -- it's an essentially random real passage. Using it as
    # "the human passage" for a given query silently mixed two different populations of
    # "human" text between train (random) and val/test (query-relevant), which is why
    # every earlier model here found zero transferable signal. See
    # [[project-genir-baseline-null-result]] in memory for the full diagnosis.
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

    # val
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

    # test
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


def dedupe_across_splits(labels, qids):
    """A handful of human (real) pids are relevant to queries in more than one split
    (e.g. same MSMARCO passage judged relevant for a train query and a val query).
    Drop those from the later split(s) so no pid straddles a train/eval boundary."""
    seen = set()
    for split in ("train", "val", "test"):
        dup = seen & labels[split].keys()
        for pid in dup:
            del labels[split][pid]
            del qids[split][pid]
        if dup:
            print(f"  dropped {len(dup)} pids from {split} (seen in earlier split)", file=sys.stderr)
        seen |= labels[split].keys()


def main():
    labels, qids = build_label_map()
    dedupe_across_splits(labels, qids)
    all_needed = set()
    for split in labels:
        all_needed |= set(labels[split].keys())
    print(f"Total distinct pids needed: {len(all_needed)}", file=sys.stderr)
    for split in labels:
        n_pos = sum(1 for v in labels[split].values() if v == 1)
        n_neg = sum(1 for v in labels[split].values() if v == 0)
        print(f"  {split}: {n_neg} human, {n_pos} generated", file=sys.stderr)

    pid_to_text = {}
    with open(COLLECTION, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i in all_needed:
                pid_to_text[i] = row[1]
            if len(pid_to_text) == len(all_needed):
                break
    print(f"Recovered text for {len(pid_to_text)}/{len(all_needed)} pids", file=sys.stderr)

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
        print(f"Wrote {n_written} rows to {out_path}", file=sys.stderr)
    print(f"Missing text for {missing_total} pids total", file=sys.stderr)


if __name__ == "__main__":
    main()
