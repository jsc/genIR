"""Contrastive/pairwise fine-tuning: DeBERTa-v3-large trained on (query-relevant real passage,
generated passage) pairs sharing the same qid, with a pairwise logistic loss pushing
score(generated) > score(real) for every pair. Same model architecture and eval protocol as
train_classifier.py so the two are directly comparable.

NOTE: pairs are built from qrels.train.tsv (real) x gen.train.tsv (generated) sharing a qid --
NOT from triples.train.tsv, whose "real" column was found to be an essentially random passage
unrelated to the query (see build_dataset.py and the genir-dataset-structure /
genir-baseline-null-result memory notes for the full audit). Using triples directly would
reintroduce the same population mismatch that corrupted the earlier baselines.

Motivation: the in-sample-only signal found by TF-IDF (train AUROC ~0.9999 even on the
corrected data, see ../baseline/train_baseline.py) came from query-paired real-vs-generated
contrasts, which then failed to generalize under plain classification. Training the
transformer with that same paired structure explicitly (instead of collapsing it into an
unpaired binary-label dataset) tests whether a higher-capacity model can turn that contrast
into a query-invariant signal.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_augmented import (
    OUT, MODEL_NAME, ScoreModel, PairDataset, TextDataset, ProgressLogger,
    best_threshold, evaluate, get_tokenizer, load_split, score_dataset,
)

DEVICE = "cuda:1"
PAIR_BATCH = 8  # 8 pairs => 16 forward passes per micro-batch, matching train_classifier.py
GRAD_ACCUM = 2
EPOCHS = 2
LR = 2e-5
WARMUP_RATIO = 0.06
LOG_EVERY = 50
TAG = "pairwise"


def load_triples_text(seed=0):
    """Build (real, generated) text pairs from the ranker-augmented train split, sharing a
    qid. The augmented human class averages ~15 real passages/query (up to 24) instead of
    ~1.27 -- a full real x gen cross product would balloon to ~969K pairs (vs. ~94.5K on the
    qrels-only data), an estimated ~19h run instead of ~2h. Instead, pair each generated
    passage with ONE real passage sampled uniformly at random from that query's (enlarged)
    real pool: total pairs stays ~= gen count (~73.6K, same order as the original run), while
    every real passage in the pool still gets a chance to appear across the many generated
    passages for its query, so training sees the full enlarged real distribution."""
    rng = np.random.default_rng(seed)
    train_df = load_split("train")
    real_by_qid, gen_by_qid = {}, {}
    for qid, text, label in zip(train_df.qid, train_df.text, train_df.label):
        (gen_by_qid if label == 1 else real_by_qid).setdefault(qid, []).append(text)

    real_texts, gen_texts = [], []
    for qid, gens in gen_by_qid.items():
        reals = real_by_qid.get(qid)
        if not reals:
            continue
        for gt in gens:
            rt = reals[rng.integers(0, len(reals))]
            real_texts.append(rt)
            gen_texts.append(gt)
    return real_texts, gen_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    device = args.device

    print(f"[{TAG}] loading tokenizer/model {MODEL_NAME} on {device}", flush=True)
    tokenizer = get_tokenizer()
    model = ScoreModel().to(device)

    real_texts, gen_texts = load_triples_text()
    print(f"[{TAG}] loaded {len(real_texts)} real/gen pairs from triples.train.tsv", flush=True)
    val_df = load_split("val")
    test_df = load_split("test")
    print(f"[{TAG}] val={len(val_df)} test={len(test_df)}", flush=True)

    pair_ds = PairDataset(real_texts, gen_texts, tokenizer)
    pair_loader = DataLoader(pair_ds, batch_size=PAIR_BATCH, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    val_ds = TextDataset(val_df.text, val_df.label, tokenizer)
    test_ds = TextDataset(test_df.text, test_df.label, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=PAIR_BATCH * 4, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=PAIR_BATCH * 4, shuffle=False, num_workers=4, pin_memory=True)

    total_steps = (len(pair_loader) // GRAD_ACCUM) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    yval = val_df.label.values
    best_val_auroc = -1
    results = {}
    global_step = 0
    logger = ProgressLogger(total_steps, log_every=LOG_EVERY, tag=TAG)

    for epoch in range(args.epochs):
        print(f"[{TAG}] === epoch {epoch} ===", flush=True)
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(pair_loader):
            r_ids = batch["real_input_ids"].to(device)
            r_mask = batch["real_attention_mask"].to(device)
            g_ids = batch["gen_input_ids"].to(device)
            g_mask = batch["gen_attention_mask"].to(device)

            input_ids = torch.cat([r_ids, g_ids], dim=0)
            attn = torch.cat([r_mask, g_mask], dim=0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = model(input_ids, attn)
                score_real, score_gen = logits.chunk(2, dim=0)
                diff = score_gen - score_real
                target = torch.ones_like(diff)
                loss = loss_fn(diff, target) / GRAD_ACCUM
            loss.backward()
            with torch.no_grad():
                pair_acc = (diff > 0).float().mean().item()

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                logger.step(global_step, loss.item() * GRAD_ACCUM, acc=pair_acc, extra=f"epoch={epoch}")

        val_scores = score_dataset(model, val_loader, device, desc=f"{TAG} val-epoch{epoch}")
        t_star, _ = best_threshold(yval, val_scores)
        r = evaluate(f"{TAG} val epoch {epoch}", yval, val_scores, t_star)
        print(f"[{TAG}] epoch {epoch} val AUROC={r['auroc']:.4f}", flush=True)
        if r["auroc"] > best_val_auroc:
            best_val_auroc = r["auroc"]
            torch.save(model.state_dict(), OUT / f"{TAG}_best.pt")
            with open(OUT / f"{TAG}_threshold.json", "w") as f:
                json.dump({"threshold": t_star}, f)
            print(f"[{TAG}] new best val AUROC={best_val_auroc:.4f}, checkpoint saved", flush=True)

    print(f"[{TAG}] loading best checkpoint for final test evaluation", flush=True)
    model.load_state_dict(torch.load(OUT / f"{TAG}_best.pt", map_location=device))
    t_star = json.load(open(OUT / f"{TAG}_threshold.json"))["threshold"]
    test_scores = score_dataset(model, test_loader, device, desc=f"{TAG} test")
    results["test"] = evaluate(f"{TAG} FINAL TEST", test_df.label.values, test_scores, t_star)
    results["best_val_auroc"] = best_val_auroc

    with open(OUT / f"{TAG}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{TAG}] DONE. results: {json.dumps(results, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
