"""
Embedding-based human-vs-LLM baseline.

Rationale: the TF-IDF/stylometric baseline (train_baseline.py) got AUROC ~0.92 on train
but ~0.50 (chance) on held-out queries -- confirmed via query-disjoint GroupKFold on the
pooled data. That means raw lexical features are just memorizing query-topic vocabulary.
Sentence embeddings from a pretrained general-purpose encoder might capture more abstract
semantic/stylistic regularities that survive across unseen topics -- this script tests that
directly, both on the official train/val/test split and on pooled query-disjoint folds.
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src" / "data"
OUT = ROOT / "src" / "intermediates"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256


def load(split):
    df = pd.read_csv(DATA / f"{split}.tsv", sep="\t")
    df["text"] = df["text"].fillna("")
    return df


def encode(model, texts, cache_path):
    if cache_path.exists():
        print(f"  loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=DEVICE,
    )
    np.save(cache_path, embs)
    return embs


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
    print("Confusion matrix [rows=true human/generated, cols=pred human/generated]:")
    print(cm)
    return {"auroc": auroc, "avg_precision": ap, "f1_macro": f1_macro,
            "f1_human": f1_human, "f1_generated": f1_gen, "threshold": threshold}


def main():
    t0 = time.time()
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    train, val, test = load("train"), load("val"), load("test")
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    print("Encoding train passages...")
    Xtr = encode(model, train.text.tolist(), OUT / "emb_train.npy")
    print("Encoding val passages...")
    Xva = encode(model, val.text.tolist(), OUT / "emb_val.npy")
    print("Encoding test passages...")
    Xte = encode(model, test.text.tolist(), OUT / "emb_test.npy")

    ytr, yva, yte = train.label.values, val.label.values, test.label.values

    # ---------- official split ----------
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, class_weight='balanced')
    clf.fit(Xtr, ytr)
    val_scores = clf.predict_proba(Xva)[:, 1]
    t_star = best_threshold(yva, val_scores)
    test_scores = clf.predict_proba(Xte)[:, 1]
    results = {"official_split": evaluate("bge-base embeddings + LogReg (official split)", yte, test_scores, t_star)}
    joblib.dump(clf, OUT / "embedding_clf.joblib")

    # ---------- pooled, query-disjoint GroupKFold sanity check ----------
    print("\nRunning pooled query-disjoint GroupKFold to check whether ANY signal "
          "transfers to unseen queries...")
    all_df = pd.concat([train, val, test], ignore_index=True)
    all_X = np.concatenate([Xtr, Xva, Xte], axis=0)
    all_y = all_df.label.values
    groups = all_df.qid.values

    gkf = GroupKFold(n_splits=5)
    fold_aucs = []
    for fold, (tr_idx, te_idx) in enumerate(tqdm(list(gkf.split(all_X, all_y, groups=groups)), desc="GroupKFold")):
        clf_f = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, class_weight='balanced')
        clf_f.fit(all_X[tr_idx], all_y[tr_idx])
        scores = clf_f.predict_proba(all_X[te_idx])[:, 1]
        auc = roc_auc_score(all_y[te_idx], scores)
        fold_aucs.append(auc)
        print(f"  fold {fold}: AUROC={auc:.4f}")
    results["pooled_groupkfold_mean_auroc"] = float(np.mean(fold_aucs))
    results["pooled_groupkfold_fold_aucs"] = [float(a) for a in fold_aucs]
    print(f"\nPooled query-disjoint mean AUROC: {np.mean(fold_aucs):.4f}")

    import json
    with open(OUT / "embedding_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
