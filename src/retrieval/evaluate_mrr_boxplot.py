"""
Per-query reciprocal-rank distributions (not just the mean/MRR) across: oracle, baseline
(no filtering), and each of our classifiers applied as a post-retrieval filter -- for a
given retriever's per-query ranked_pids (loaded from mrr_baseline_{split}.json, produced by
evaluate_mrr.py). Answers: does filtering with a real classifier move the distribution
toward the oracle, leave it unchanged, or make it worse -- and does that differ by query
(hence box plots, not just means)?

CPU-only (reuses already-fit sklearn models) so it can run alongside GPU-bound jobs.
"""
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
MRR_CUTOFF = 10
FILTER_THRESHOLD = 0.5


def load_qrels_gen(split):
    base = ROOT / "test" if split == "test" else ROOT / "train"
    qrels = pd.read_csv(base / f"qrels.{split}.tsv", sep="\t", header=None,
                         names=["qid", "it", "pid", "rel"])
    gen = pd.read_csv(base / f"gen.{split}.tsv", sep="\t", header=None, names=["qid", "pid"])
    return qrels.groupby("qid")["pid"].apply(set).to_dict(), gen.groupby("qid")["pid"].apply(set).to_dict()


def reciprocal_rank(ranked_pids, relevant_set, cutoff=MRR_CUTOFF):
    for rank, pid in enumerate(ranked_pids[:cutoff], start=1):
        if pid in relevant_set:
            return 1.0 / rank
    return 0.0


def main(retriever_tag="bm25"):
    results_by_split = {}
    for split in ["val", "test"]:
        baseline_path = OUT / f"mrr_baseline_{split}.json"
        if not baseline_path.exists():
            print(f"skip {split}: {baseline_path} not found", flush=True)
            continue
        baseline = json.load(open(baseline_path))
        relevant_by_qid, gen_by_qid = load_qrels_gen(split)
        per_query = {int(k): v for k, v in baseline["per_query"].items()}

        all_pids = sorted({p for pq in per_query.values() for p in pq["ranked_pids"]})
        needed = set(all_pids)
        pid_to_text = {}
        with open(ROOT / "train" / "genIR.collection.tsv", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i in needed:
                    pid_to_text[i] = row[1]
                if len(pid_to_text) == len(needed):
                    break
        texts = [pid_to_text.get(p, "") for p in all_pids]
        print(f"[{split}] {len(all_pids)} unique retrieved pids, text recovered for "
              f"{sum(1 for t in texts if t)}", flush=True)

        # ---------- sklearn classifiers (CPU) ----------
        art = joblib.load(ROOT / "baseline" / "artifacts" / "baseline_models.joblib")
        sys.path.insert(0, str(ROOT / "baseline"))
        from features import extract_features

        Xw = art["word_vec"].transform(texts)
        Xc = art["char_vec"].transform(texts)
        X_tfidf = sparse.hstack([Xw, Xc], format="csr")
        Xs = art["scaler"].transform(extract_features(texts))
        X_comb = sparse.hstack([X_tfidf, sparse.csr_matrix(Xs)], format="csr")

        classifier_scores = {
            "stylometric": art["clf_style"].predict_proba(Xs)[:, 1],
            "tfidf": art["clf_tfidf"].predict_proba(X_tfidf)[:, 1],
            "tfidf_stylometric": art["clf_comb"].predict_proba(X_comb)[:, 1],
        }

        conditions = {"baseline": {}, "oracle": {}}
        for name in classifier_scores:
            conditions[f"filtered_{name}"] = {}

        for qid, pq in per_query.items():
            relevant = relevant_by_qid.get(qid, set())
            gen_pids = gen_by_qid.get(qid, set())
            ranked = pq["ranked_pids"]

            conditions["baseline"][qid] = reciprocal_rank(ranked, relevant)
            oracle_ranked = [p for p in ranked if p not in gen_pids]
            conditions["oracle"][qid] = reciprocal_rank(oracle_ranked, relevant)

            for name, scores in classifier_scores.items():
                pid_to_score = dict(zip(all_pids, scores))
                filtered = [p for p in ranked if pid_to_score.get(p, 0.0) < FILTER_THRESHOLD]
                conditions[f"filtered_{name}"][qid] = reciprocal_rank(filtered, relevant)

        summary = {cond: float(np.mean(list(vals.values()))) for cond, vals in conditions.items()}
        print(f"[{split}] MRR by condition: {json.dumps(summary, indent=2)}", flush=True)

        results_by_split[split] = {
            cond: {str(k): v for k, v in vals.items()} for cond, vals in conditions.items()
        }

    with open(OUT / f"mrr_per_query_{retriever_tag}.json", "w") as f:
        json.dump(results_by_split, f, indent=2)
    print(f"Saved per-query RR arrays to mrr_per_query_{retriever_tag}.json", flush=True)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "bm25"
    main(tag)
