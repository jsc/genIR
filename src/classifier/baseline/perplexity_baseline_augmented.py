"""
Zero-shot, topic-agnostic-by-construction baseline: score each passage's fluency/
predictability under a generic causal LM (GPT-2 medium) and classify human-vs-LLM from
those likelihood statistics (DetectGPT/GLTR-style features), rather than from lexical
content. This directly tests whether generation leaves a detectable "shape" in the
token-level probability distribution that survives across unseen topics -- the TF-IDF
baseline showed lexical content alone does not (see train_baseline.py results).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src" / "data_ranker_augmented"
OUT = ROOT / "src" / "ranker_augmented_intermediates"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt2-medium"
import os
DEVICE = os.environ.get("PPL_DEVICE") or (
    "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
)
BATCH_SIZE = 32
MAX_LEN = 256

FEATURE_NAMES = ["mean_nll", "std_nll", "median_nll", "max_nll", "min_nll", "frac_high_surprisal", "n_tokens"]


def load(split):
    df = pd.read_csv(DATA / f"{split}.tsv", sep="\t")
    df["text"] = df["text"].fillna("")
    return df


@torch.no_grad()
def score_texts(texts, tokenizer, model):
    # sort by length to reduce padding waste, then restore original order
    order = np.argsort([len(t) for t in texts])
    feats = np.zeros((len(texts), len(FEATURE_NAMES)), dtype=np.float64)

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"scoring on {DEVICE}"):
        batch_idx = order[start:start + BATCH_SIZE]
        batch_texts = [texts[i] for i in batch_idx]
        enc = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN,
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc["attention_mask"].to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
            logits = model(input_ids=input_ids, attention_mask=attn).logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_mask = attn[:, 1:].float()

        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.size())

        for bi, orig_i in enumerate(batch_idx):
            m = shift_mask[bi].bool()
            vals = token_nll[bi][m].detach().cpu().numpy()
            if vals.size == 0:
                continue
            feats[orig_i] = [
                vals.mean(), vals.std(), np.median(vals), vals.max(), vals.min(),
                (vals > 5.0).mean(), vals.size,
            ]
    return feats


def best_threshold(y_true, scores):
    thresholds = np.linspace(0.01, 0.99, 197)
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (scores >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate(name, y_true, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)
    auroc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    f1_macro = f1_score(y_true, preds, average="macro")
    f1_human = f1_score(y_true, preds, pos_label=0)
    f1_gen = f1_score(y_true, preds, pos_label=1)
    cm = confusion_matrix(y_true, preds)
    print(f"\n=== {name} (threshold={threshold:.3f}) ===")
    print(f"AUROC={auroc:.4f}  AvgPrecision={ap:.4f}  F1_macro={f1_macro:.4f}  "
          f"F1_human={f1_human:.4f}  F1_generated={f1_gen:.4f}")
    print(cm)
    return {"auroc": auroc, "avg_precision": ap, "f1_macro": f1_macro,
            "f1_human": f1_human, "f1_generated": f1_gen, "threshold": threshold}


def get_features(split, texts, tokenizer, model):
    cache = OUT / f"ppl_feats_{split}.npy"
    if cache.exists():
        print(f"  loading cached ppl features from {cache}")
        return np.load(cache)
    feats = score_texts(texts, tokenizer, model)
    np.save(cache, feats)
    return feats


def main():
    t0 = time.time()
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    train, val, test = load("train"), load("val"), load("test")
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    Xtr = get_features("train", train.text.tolist(), tokenizer, model)
    Xva = get_features("val", val.text.tolist(), tokenizer, model)
    Xte = get_features("test", test.text.tolist(), tokenizer, model)
    ytr, yva, yte = train.label.values, val.label.values, test.label.values

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(Xtr_s, ytr)
    val_scores = clf.predict_proba(Xva_s)[:, 1]
    t_star = best_threshold(yva, val_scores)
    test_scores = clf.predict_proba(Xte_s)[:, 1]
    results = {"official_split": evaluate("GPT2-medium NLL features + LogReg (official split)", yte, test_scores, t_star)}

    import joblib
    joblib.dump({"scaler": scaler, "clf": clf}, OUT / "perplexity_clf.joblib")

    print("\nFeature weights (standardized, +ve => more LLM-like):")
    for name, w in sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {name:22s} {w:+.3f}")

    print("\nRunning pooled query-disjoint GroupKFold sanity check...")
    all_df = pd.concat([train, val, test], ignore_index=True)
    all_X = np.concatenate([Xtr, Xva, Xte], axis=0)
    all_y = all_df.label.values
    groups = all_df.qid.values

    gkf = GroupKFold(n_splits=5)
    fold_aucs = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(all_X, all_y, groups=groups)):
        sc = StandardScaler().fit(all_X[tr_idx])
        clf_f = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf_f.fit(sc.transform(all_X[tr_idx]), all_y[tr_idx])
        scores = clf_f.predict_proba(sc.transform(all_X[te_idx]))[:, 1]
        auc = roc_auc_score(all_y[te_idx], scores)
        fold_aucs.append(auc)
        print(f"  fold {fold}: AUROC={auc:.4f}")
    results["pooled_groupkfold_mean_auroc"] = float(np.mean(fold_aucs))
    results["pooled_groupkfold_fold_aucs"] = [float(a) for a in fold_aucs]
    print(f"\nPooled query-disjoint mean AUROC: {np.mean(fold_aucs):.4f}")

    with open(OUT / "perplexity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
