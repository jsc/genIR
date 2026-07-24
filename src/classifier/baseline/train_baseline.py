"""
Baseline human-vs-LLM passage classifier for the GenIR SIGIR'26 challenge.

Three models are trained and compared:
  1. stylometric  - hand-crafted surface features only (interpretable baseline)
  2. tfidf        - word(1,2) + char(3,5) TF-IDF + Logistic Regression (main baseline)
  3. tfidf+style  - TF-IDF features concatenated with stylometric features

label: 1 = LLM-generated, 0 = human-written.
All three splits are built from qrels (verified query-relevant real passages) + gen files,
so all are heavily imbalanced (few real passages per query, many generated) -- LogReg uses
class_weight='balanced', and we report AUROC / average precision / F1 rather than accuracy.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features, FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src" / "data"
OUT = ROOT / "src" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)


def load(split):
    df = pd.read_csv(DATA / f"{split}.tsv", sep="\t")
    df["text"] = df["text"].fillna("")
    return df


def best_threshold(y_true, scores):
    """Pick the probability threshold on val that maximizes F1."""
    thresholds = np.linspace(0.01, 0.99, 197)
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (scores >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate(name, y_true, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)
    auroc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    f1_pos = f1_score(y_true, preds, pos_label=1)
    f1_neg = f1_score(y_true, preds, pos_label=0)
    f1_macro = f1_score(y_true, preds, average="macro")
    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, target_names=["human", "generated"], digits=3)
    print(f"\n=== {name} (threshold={threshold:.3f}) ===")
    print(f"AUROC={auroc:.4f}  AvgPrecision={ap:.4f}  F1_macro={f1_macro:.4f}  "
          f"F1_human={f1_neg:.4f}  F1_generated={f1_pos:.4f}")
    print("Confusion matrix [rows=true human/generated, cols=pred human/generated]:")
    print(cm)
    print(report)
    return {
        "auroc": auroc, "avg_precision": ap, "f1_macro": f1_macro,
        "f1_human": f1_neg, "f1_generated": f1_pos, "threshold": threshold,
        "confusion_matrix": cm.tolist(),
    }


def main():
    t0 = time.time()
    train = load("train")
    val = load("val")
    test = load("test")
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    y_train, y_val, y_test = train.label.values, val.label.values, test.label.values

    results = {}

    # ---------- 1. Stylometric-only baseline ----------
    print("\nExtracting stylometric features...")
    X_style_train = extract_features(train.text.tolist())
    X_style_val = extract_features(val.text.tolist())
    X_style_test = extract_features(test.text.tolist())

    scaler = StandardScaler().fit(X_style_train)
    Xs_train = scaler.transform(X_style_train)
    Xs_val = scaler.transform(X_style_val)
    Xs_test = scaler.transform(X_style_test)

    clf_style = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight='balanced')
    clf_style.fit(Xs_train, y_train)
    val_scores = clf_style.predict_proba(Xs_val)[:, 1]
    t_star, _ = best_threshold(y_val, val_scores)
    test_scores = clf_style.predict_proba(Xs_test)[:, 1]
    results["stylometric"] = evaluate("stylometric (LogReg)", y_test, test_scores, t_star)

    coefs = sorted(zip(FEATURE_NAMES, clf_style.coef_[0]), key=lambda x: -abs(x[1]))
    print("Top stylometric feature weights (standardized, +ve => more LLM-like):")
    for name, w in coefs:
        print(f"  {name:24s} {w:+.3f}")

    # ---------- 2. TF-IDF baseline ----------
    print("\nFitting TF-IDF vectorizers...")
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100_000,
        sublinear_tf=True, strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=50_000,
        sublinear_tf=True,
    )
    Xw_train = word_vec.fit_transform(train.text)
    Xc_train = char_vec.fit_transform(train.text)
    Xw_val, Xc_val = word_vec.transform(val.text), char_vec.transform(val.text)
    Xw_test, Xc_test = word_vec.transform(test.text), char_vec.transform(test.text)

    X_tfidf_train = sparse.hstack([Xw_train, Xc_train], format="csr")
    X_tfidf_val = sparse.hstack([Xw_val, Xc_val], format="csr")
    X_tfidf_test = sparse.hstack([Xw_test, Xc_test], format="csr")
    print(f"TF-IDF feature dim: {X_tfidf_train.shape[1]}")

    clf_tfidf = LogisticRegression(max_iter=2000, C=10.0, n_jobs=-1, class_weight='balanced')
    clf_tfidf.fit(X_tfidf_train, y_train)
    val_scores = clf_tfidf.predict_proba(X_tfidf_val)[:, 1]
    t_star, _ = best_threshold(y_val, val_scores)
    test_scores = clf_tfidf.predict_proba(X_tfidf_test)[:, 1]
    results["tfidf"] = evaluate("TF-IDF word+char (LogReg)", y_test, test_scores, t_star)

    # ---------- 3. TF-IDF + stylometric combined ----------
    X_comb_train = sparse.hstack([X_tfidf_train, sparse.csr_matrix(Xs_train)], format="csr")
    X_comb_val = sparse.hstack([X_tfidf_val, sparse.csr_matrix(Xs_val)], format="csr")
    X_comb_test = sparse.hstack([X_tfidf_test, sparse.csr_matrix(Xs_test)], format="csr")

    clf_comb = LogisticRegression(max_iter=2000, C=10.0, n_jobs=-1, class_weight='balanced')
    clf_comb.fit(X_comb_train, y_train)
    val_scores = clf_comb.predict_proba(X_comb_val)[:, 1]
    t_star, _ = best_threshold(y_val, val_scores)
    test_scores = clf_comb.predict_proba(X_comb_test)[:, 1]
    results["tfidf+stylometric"] = evaluate("TF-IDF + stylometric (LogReg)", y_test, test_scores, t_star)

    # ---------- save everything ----------
    joblib.dump(
        {"word_vec": word_vec, "char_vec": char_vec, "scaler": scaler,
         "clf_style": clf_style, "clf_tfidf": clf_tfidf, "clf_comb": clf_comb},
        OUT / "baseline_models.joblib",
    )
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts saved to {OUT}")
    print("\nSummary (test set):")
    print(f"{'model':22s} {'AUROC':>8s} {'AvgPrec':>8s} {'F1_macro':>9s} {'F1_human':>9s} {'F1_gen':>8s}")
    for name, r in results.items():
        print(f"{name:22s} {r['auroc']:8.4f} {r['avg_precision']:8.4f} {r['f1_macro']:9.4f} "
              f"{r['f1_human']:9.4f} {r['f1_generated']:8.4f}")


if __name__ == "__main__":
    main()
