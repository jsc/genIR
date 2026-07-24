"""Shared dataset/model/eval code for the DeBERTa-v3-large human-vs-LLM classifier.

Both train_classifier.py (standard BCE fine-tuning) and train_pairwise.py (contrastive
fine-tuning on triples.train.tsv) use the SAME model wrapper (a single scalar "generated-ness"
logit) so their outputs are directly comparable on the same val/test evaluation protocol used
for the baselines in ../baseline/.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, roc_auc_score,
)
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src" / "data_ranker_augmented"
OUT = ROOT / "src" / "ranker_augmented_intermediates"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 192


def load_split(split):
    df = pd.read_csv(DATA / f"{split}.tsv", sep="\t")
    df["text"] = df["text"].fillna("")
    return df


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class ScoreModel(nn.Module):
    """Backbone + single-logit head. sigmoid(logit) = P(generated)."""

    def __init__(self):
        super().__init__()
        # The pretrained checkpoint's config declares torch_dtype=float16; loading it
        # naively leaves backbone weights in fp16 while autocast expects fp32 master
        # weights, and that mismatch produces NaNs within a step or two. Force fp32.
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over real tokens rather than using the [CLS] position: DeBERTa-v3 has
        # no NSP-style pretraining objective, so a fresh linear head on position-0 alone
        # starts from a much less informative representation than mean pooling does.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.head(self.dropout(pooled)).squeeze(-1)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_len, padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class PairDataset(Dataset):
    """(real_text, gen_text) pairs from triples.train.tsv for pairwise/contrastive training."""

    def __init__(self, real_texts, gen_texts, tokenizer, max_len=MAX_LEN):
        self.real_texts = list(real_texts)
        self.gen_texts = list(gen_texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.real_texts)

    def _tok(self, text):
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        r_ids, r_mask = self._tok(self.real_texts[idx])
        g_ids, g_mask = self._tok(self.gen_texts[idx])
        return {
            "real_input_ids": r_ids, "real_attention_mask": r_mask,
            "gen_input_ids": g_ids, "gen_attention_mask": g_mask,
        }


@torch.no_grad()
def score_dataset(model, dataloader, device, desc="scoring"):
    model.eval()
    all_scores = []
    for batch in tqdm(dataloader, desc=desc, mininterval=2.0):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(input_ids, attn)
        all_scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(all_scores)


def best_threshold(y_true, scores):
    thresholds = np.linspace(0.01, 0.99, 197)
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (scores >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate(name, y_true, scores, threshold=0.5, log_fn=print):
    preds = (scores >= threshold).astype(int)
    auroc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    f1_macro = f1_score(y_true, preds, average="macro")
    f1_human = f1_score(y_true, preds, pos_label=0)
    f1_gen = f1_score(y_true, preds, pos_label=1)
    cm = confusion_matrix(y_true, preds)
    log_fn(f"\n=== {name} (threshold={threshold:.3f}) ===")
    log_fn(f"AUROC={auroc:.4f}  AvgPrecision={ap:.4f}  F1_macro={f1_macro:.4f}  "
           f"F1_human={f1_human:.4f}  F1_generated={f1_gen:.4f}")
    log_fn(f"Confusion matrix:\n{cm}")
    return {"auroc": auroc, "avg_precision": ap, "f1_macro": f1_macro,
            "f1_human": f1_human, "f1_generated": f1_gen, "threshold": float(threshold)}


class ProgressLogger:
    """Periodic plain-text (real newline, flushed) progress lines -- safe to `tail -f` /
    grep, unlike tqdm's carriage-return bar which mangles into one giant line when piped
    to a file."""

    def __init__(self, total_steps, log_every=50, tag=""):
        self.total_steps = total_steps
        self.log_every = log_every
        self.tag = tag
        self.t0 = time.time()
        self.losses = []
        self.accs = []

    def step(self, step_idx, loss, acc=None, extra=""):
        self.losses.append(loss)
        if acc is not None:
            self.accs.append(acc)
        if step_idx % self.log_every == 0 or step_idx == self.total_steps:
            elapsed = time.time() - self.t0
            rate = step_idx / elapsed if elapsed > 0 else 0.0
            eta = (self.total_steps - step_idx) / rate if rate > 0 else float("inf")
            avg_loss = float(np.mean(self.losses[-self.log_every:]))
            acc_str = f" train_acc={float(np.mean(self.accs[-self.log_every:])):.3f}" if self.accs else ""
            print(f"[{self.tag}] step {step_idx}/{self.total_steps} "
                  f"loss={avg_loss:.4f}{acc_str} {rate:.2f} it/s elapsed={elapsed:.0f}s eta={eta:.0f}s {extra}",
                  flush=True)
