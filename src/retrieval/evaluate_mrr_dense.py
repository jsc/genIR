"""
Same question as evaluate_mrr.py (does filtering recover MRR?) but with the dense
retriever fine-tuned on this collection's own qrels instead of BM25 -- BM25 alone is
known to underperform on this style of task, and off-the-shelf MS MARCO dense retrievers
are invalid here since the collection itself has been modified (see train_dense_retriever.py).

Retrieval: FAISS flat inner-product search (== cosine similarity, since both query and
collection embeddings are L2-normalized) over the pre-built index (build_faiss_index.py),
top-100 per query. Everything downstream (baseline/filtered/oracle MRR, TF-IDF classifier
filtering) mirrors evaluate_mrr.py exactly so the two retrievers are directly comparable.
"""
import csv
import json
import sys
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
MODEL_PATH = str(OUT / "dense_retriever_model")
TOP_K = 100
MRR_CUTOFF = 10
DEVICE = "cuda:0"


def load_queries(split):
    return pd.read_csv(ROOT / "train" / f"queries.{split}.tsv",
                        sep="\t", header=None, names=["qid", "query"])


def load_qrels_gen(split):
    base = ROOT / "test" if split == "test" else ROOT / "train"
    qrels = pd.read_csv(base / f"qrels.{split}.tsv", sep="\t", header=None,
                         names=["qid", "it", "pid", "rel"])
    gen = pd.read_csv(base / f"gen.{split}.tsv", sep="\t", header=None, names=["qid", "pid"])
    relevant_by_qid = qrels.groupby("qid")["pid"].apply(set).to_dict()
    gen_by_qid = gen.groupby("qid")["pid"].apply(set).to_dict()
    return relevant_by_qid, gen_by_qid


def reciprocal_rank(ranked_pids, relevant_set, cutoff=MRR_CUTOFF):
    for rank, pid in enumerate(ranked_pids[:cutoff], start=1):
        if pid in relevant_set:
            return 1.0 / rank
    return 0.0


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"

    print(f"Loading FAISS index ...", flush=True)
    t0 = time.time()
    index = faiss.read_index(str(OUT / "faiss_index.bin"))
    faiss_pids = np.load(OUT / "faiss_pids.npy")
    print(f"Index loaded: {index.ntotal:,} vectors in {time.time() - t0:.0f}s", flush=True)

    model = SentenceTransformer(MODEL_PATH, device=DEVICE)

    queries = load_queries(split)
    relevant_by_qid, gen_by_qid = load_qrels_gen(split)
    queries = queries[queries.qid.isin(relevant_by_qid.keys())].reset_index(drop=True)
    print(f"{len(queries)} {split} queries with qrels", flush=True)

    t0 = time.time()
    q_embs = model.encode(queries["query"].tolist(), batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True, device=DEVICE)
    scores, idx = index.search(q_embs.astype(np.float32), TOP_K)
    print(f"Retrieval done in {time.time() - t0:.1f}s", flush=True)

    # ---- baseline MRR + generated-passage contamination stats ----
    per_query = {}
    mrrs, gen_fractions, first_hit_ranks = [], [], []
    for i, qid in enumerate(queries.qid):
        qid = int(qid)
        ranked_pids = [int(p) for p in faiss_pids[idx[i]]]
        relevant = relevant_by_qid.get(qid, set())
        gen_pids = gen_by_qid.get(qid, set())

        rr = reciprocal_rank(ranked_pids, relevant)
        mrrs.append(rr)
        n_gen_in_topk = sum(1 for p in ranked_pids if p in gen_pids)
        gen_fractions.append(n_gen_in_topk / len(ranked_pids) if ranked_pids else 0.0)

        rel_rank = next((r for r, p in enumerate(ranked_pids, 1) if p in relevant), None)
        first_hit_ranks.append(rel_rank)

        per_query[qid] = {
            "reciprocal_rank": rr, "n_gen_in_topk": n_gen_in_topk,
            "topk_size": len(ranked_pids), "relevant_rank": rel_rank,
            "ranked_pids": ranked_pids,
        }

    baseline_mrr = float(np.mean(mrrs))
    mean_gen_fraction = float(np.mean(gen_fractions))
    hit_rate = float(np.mean([1 for r in first_hit_ranks if r is not None]) if first_hit_ranks else 0.0)
    print(f"\n=== BASELINE (dense, no filtering) ===")
    print(f"MRR@{MRR_CUTOFF}: {baseline_mrr:.4f}")
    print(f"Mean fraction of top-{TOP_K} that are LLM-generated: {mean_gen_fraction:.4f}")
    print(f"Relevant passage found in top-{TOP_K}: {hit_rate:.4f} of queries")
    found = [r for r in first_hit_ranks if r is not None]
    if found:
        print(f"Median rank of first relevant hit (when found): {np.median(found):.1f}")

    with open(OUT / f"mrr_baseline_dense_{split}.json", "w") as f:
        json.dump({
            "baseline_mrr": baseline_mrr, "mean_gen_fraction_topk": mean_gen_fraction,
            "hit_rate_topk": hit_rate, "n_queries": len(mrrs),
            "per_query": per_query,
        }, f, indent=2)

    # ---- filtered MRR using the TF-IDF baseline classifier (same as evaluate_mrr.py) ----
    print("\nScoring retrieved passages with the TF-IDF+LogReg classifier for filtering...", flush=True)
    art = joblib.load(ROOT / "baseline" / "artifacts" / "baseline_models.joblib")
    word_vec, char_vec, clf_tfidf = art["word_vec"], art["char_vec"], art["clf_tfidf"]

    all_pids = sorted({p for pq in per_query.values() for p in pq["ranked_pids"]})
    pid_to_text = {}
    needed = set(all_pids)
    with open(ROOT / "train" / "genIR.collection.tsv", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i in needed:
                pid_to_text[i] = row[1]
            if len(pid_to_text) == len(needed):
                break
    print(f"Recovered text for {len(pid_to_text)}/{len(needed)} retrieved pids", flush=True)

    texts = [pid_to_text.get(p, "") for p in all_pids]
    Xw = word_vec.transform(texts)
    Xc = char_vec.transform(texts)
    X = sparse.hstack([Xw, Xc], format="csr")
    gen_prob = clf_tfidf.predict_proba(X)[:, 1]
    pid_to_genprob = dict(zip(all_pids, gen_prob))

    filtered_results = {}
    for threshold in [0.5, 0.7, 0.9]:
        filtered_mrrs = []
        n_catastrophic = 0
        n_had_relevant_in_topk = 0
        for qid, pq in per_query.items():
            relevant = relevant_by_qid.get(qid, set())
            had_relevant = any(p in relevant for p in pq["ranked_pids"][:MRR_CUTOFF])
            n_had_relevant_in_topk += int(had_relevant)
            filtered = [p for p in pq["ranked_pids"] if pid_to_genprob.get(p, 0.0) < threshold]
            rr = reciprocal_rank(filtered, relevant)
            filtered_mrrs.append(rr)
            if had_relevant and rr == 0.0:
                n_catastrophic += 1
        filtered_mrr = float(np.mean(filtered_mrrs))
        catastrophic_rate = n_catastrophic / n_had_relevant_in_topk if n_had_relevant_in_topk else 0.0
        print(f"Filtered MRR@{MRR_CUTOFF} (drop predicted-generated, threshold={threshold}): "
              f"{filtered_mrr:.4f}  (baseline={baseline_mrr:.4f}, "
              f"delta={filtered_mrr - baseline_mrr:+.4f})")
        print(f"  Catastrophic failures: {n_catastrophic}/{n_had_relevant_in_topk} queries "
              f"({100 * catastrophic_rate:.1f}%)")
        filtered_results[str(threshold)] = {
            "filtered_mrr": filtered_mrr, "n_catastrophic": n_catastrophic,
            "n_had_relevant_in_topk": n_had_relevant_in_topk,
            "catastrophic_rate": catastrophic_rate,
        }

    # ---- oracle filtering (perfect classifier, upper bound) ----
    oracle_mrrs = []
    for qid, pq in per_query.items():
        relevant = relevant_by_qid.get(qid, set())
        gen_pids = gen_by_qid.get(qid, set())
        filtered = [p for p in pq["ranked_pids"] if p not in gen_pids]
        oracle_mrrs.append(reciprocal_rank(filtered, relevant))
    oracle_mrr = float(np.mean(oracle_mrrs))
    print(f"\nOracle MRR@{MRR_CUTOFF} (perfect generated-passage removal, upper bound): "
          f"{oracle_mrr:.4f}  (delta vs baseline={oracle_mrr - baseline_mrr:+.4f})")

    with open(OUT / f"mrr_filtered_dense_{split}.json", "w") as f:
        json.dump({"baseline_mrr": baseline_mrr, "oracle_mrr": oracle_mrr,
                    "mean_gen_fraction_topk": mean_gen_fraction, "hit_rate_topk": hit_rate,
                    "by_threshold": filtered_results}, f, indent=2)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
