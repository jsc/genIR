"""Standard fine-tuned classifier: DeBERTa-v3-large + BCE loss on train.tsv labels.

Control condition for train_pairwise.py -- same model architecture, same eval protocol,
different training signal (direct binary label vs. contrastive real/gen pairs).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DATA, OUT, MODEL_NAME, ScoreModel, TextDataset, ProgressLogger,
    best_threshold, evaluate, get_tokenizer, load_split, score_dataset,
)

DEVICE = "cuda:0"
BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS = 2
LR = 2e-5
WARMUP_RATIO = 0.06
LOG_EVERY = 50
TAG = "classifier"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    device = args.device

    print(f"[{TAG}] loading tokenizer/model {MODEL_NAME} on {device}", flush=True)
    tokenizer = get_tokenizer()
    model = ScoreModel().to(device)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    print(f"[{TAG}] train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)

    train_ds = TextDataset(train_df.text, train_df.label, tokenizer)
    val_ds = TextDataset(val_df.text, val_df.label, tokenizer)
    test_ds = TextDataset(test_df.text, test_df.label, tokenizer)

    # train is heavily imbalanced (~2.7% human, see genir-dataset-structure memory note) --
    # oversample the minority class so batches aren't almost-always all-generated.
    class_counts = train_df.label.value_counts().to_dict()
    sample_weights = train_df.label.map(lambda l: 1.0 / class_counts[l]).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)

    total_steps = (len(train_loader) // GRAD_ACCUM) * args.epochs
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
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = model(input_ids, attn)
                loss = loss_fn(logits, labels) / GRAD_ACCUM
            loss.backward()
            with torch.no_grad():
                batch_acc = ((torch.sigmoid(logits) > 0.5).float() == labels).float().mean().item()

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                logger.step(global_step, loss.item() * GRAD_ACCUM, acc=batch_acc, extra=f"epoch={epoch}")

        # end-of-epoch val check
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

    # final test eval using best checkpoint
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
