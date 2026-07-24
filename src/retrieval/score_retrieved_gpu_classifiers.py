"""
Add GPU-based classifiers (BGE embeddings, GPT-2 perplexity features, both fine-tuned
DeBERTa-v3-large transformers) as additional filtering conditions to the per-query
reciprocal-rank box-plot data already computed by evaluate_mrr_boxplot.py (sklearn
classifiers only, CPU). Loads the existing mrr_per_query_bm25.json and adds new
"filtered_<name>" conditions in place, so the sklearn-derived conditions aren't recomputed.
"""
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
MRR_CUTOFF = 10
FILTER_THRESHOLD = 0.5
DEVICE = "cuda:0"


def reciprocal_rank(ranked_pids, relevant_set, cutoff=MRR_CUTOFF):
    for rank, pid in enumerate(ranked_pids[:cutoff], start=1):
        if pid in relevant_set:
            return 1.0 / rank
    return 0.0


def load_qrels(split):
    base = ROOT / "test" if split == "test" else ROOT / "train"
    qrels = pd.read_csv(base / f"qrels.{split}.tsv", sep="\t", header=None,
                         names=["qid", "it", "pid", "rel"])
    return qrels.groupby("qid")["pid"].apply(set).to_dict()


def collect_pids_texts(per_query_by_split):
    all_pids = set()
    for per_query in per_query_by_split.values():
        for pq in per_query.values():
            all_pids.update(pq["ranked_pids"])
    all_pids = sorted(all_pids)
    needed = set(all_pids)
    pid_to_text = {}
    with open(ROOT / "train" / "genIR.collection.tsv", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i in needed:
                pid_to_text[i] = row[1]
            if len(pid_to_text) == len(needed):
                break
    return all_pids, [pid_to_text.get(p, "") for p in all_pids]


def score_embeddings(texts):
    print("Scoring with BGE embeddings + LogReg...", flush=True)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)
    embs = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True,
                         normalize_embeddings=True, device=DEVICE)
    clf = joblib.load(ROOT / "baseline" / "artifacts" / "embedding_clf.joblib")
    return clf.predict_proba(embs)[:, 1]


def score_perplexity(texts):
    print("Scoring with GPT2-medium NLL features + LogReg...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(DEVICE).eval()

    # empty passage text -> zero-length token sequence -> crashes the model forward pass;
    # substitute a harmless placeholder so tokenization always yields >=1 real token
    safe_texts = [t if t.strip() else "." for t in texts]

    feats = np.zeros((len(texts), 7), dtype=np.float64)
    BATCH = 32
    with torch.no_grad():
        for start in range(0, len(safe_texts), BATCH):
            batch_texts = safe_texts[start:start + BATCH]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True,
                             max_length=256)
            input_ids = enc["input_ids"].to(DEVICE)
            attn = enc["attention_mask"].to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attn).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = attn[:, 1:].float()
            token_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
                reduction="none",
            ).reshape(shift_labels.size())
            for bi in range(len(batch_texts)):
                m = shift_mask[bi].bool()
                vals = token_nll[bi][m].detach().cpu().numpy()
                if vals.size == 0:
                    continue
                feats[start + bi] = [
                    vals.mean(), vals.std(), np.median(vals), vals.max(), vals.min(),
                    (vals > 5.0).mean(), vals.size,
                ]
            if (start // BATCH) % 50 == 0:
                print(f"  {start}/{len(texts)}", flush=True)

    saved = joblib.load(ROOT / "baseline" / "artifacts" / "perplexity_clf.joblib")
    return saved["clf"].predict_proba(saved["scaler"].transform(feats))[:, 1]


def score_transformer(texts, ckpt_name):
    print(f"Scoring with transformer checkpoint {ckpt_name}...", flush=True)
    sys.path.insert(0, str(ROOT / "transformer"))
    from common import ScoreModel, get_tokenizer, TextDataset, score_dataset, OUT as TOUT
    from torch.utils.data import DataLoader

    tokenizer = get_tokenizer()
    model = ScoreModel().to(DEVICE)
    model.load_state_dict(torch.load(TOUT / ckpt_name, map_location=DEVICE))
    ds = TextDataset(texts, [0] * len(texts), tokenizer)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
    return score_dataset(model, loader, DEVICE, desc=ckpt_name)


def main():
    per_query_by_split = {}
    relevant_by_split = {}
    for split in ["val", "test"]:
        p = OUT / f"mrr_baseline_{split}.json"
        if not p.exists():
            continue
        baseline = json.load(open(p))
        per_query_by_split[split] = {int(k): v for k, v in baseline["per_query"].items()}
        relevant_by_split[split] = load_qrels(split)

    all_pids, texts = collect_pids_texts(per_query_by_split)
    print(f"{len(all_pids)} unique retrieved pids across val+test", flush=True)

    boxplot_path = OUT / "mrr_per_query_bm25.json"
    results = json.load(open(boxplot_path)) if boxplot_path.exists() else {s: {} for s in per_query_by_split}

    scorers = {
        "embeddings": score_embeddings,
        "perplexity": score_perplexity,
        "transformer_classifier": lambda t: score_transformer(t, "classifier_best.pt"),
        "transformer_pairwise": lambda t: score_transformer(t, "pairwise_best.pt"),
    }

    for name, fn in scorers.items():
        scores = fn(texts)
        pid_to_score = dict(zip(all_pids, scores))
        for split, per_query in per_query_by_split.items():
            cond = {}
            for qid, pq in per_query.items():
                relevant = relevant_by_split[split].get(qid, set())
                filtered = [p for p in pq["ranked_pids"] if pid_to_score.get(p, 0.0) < FILTER_THRESHOLD]
                cond[str(qid)] = reciprocal_rank(filtered, relevant)
            results[split][f"filtered_{name}"] = cond
            mean_rr = float(np.mean(list(cond.values())))
            print(f"[{split}] filtered_{name} MRR: {mean_rr:.4f}", flush=True)
        with open(boxplot_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved progress after {name}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
