"""
Fine-tune a bi-encoder dense retriever ON THIS COLLECTION, from a base checkpoint that has NO
MS MARCO-specific pretraining (BAAI/llm-embedder: BERT-base, retrieval-oriented pretraining
for LLM-augmentation tasks, not MS MARCO). Off-the-shelf MS MARCO dense retrievers (e.g.
sentence-transformers/msmarco-*) can't validly be used here: this collection's passage ids and
content have been modified/augmented with LLM-generated passages, so a model that has memorized
or specialized on "clean" MS MARCO would give confounded, unrealistic retrieval behavior.

Training pairs: (query, query-relevant real passage) from qrels.train.tsv, joined with
queries.train.tsv -- i.e. exactly the label==0 rows of baseline/data/train.tsv, joined back to
query text. In-batch negatives (MultipleNegativesRankingLoss) provide the negative signal, NOT
the generated passages -- this keeps the retriever's behavior with respect to generated content
"organic" (learned only from what real relevance looks like) rather than explicitly trained to
discriminate against generated passages, which would confound the later "does a good dense
retriever also get fooled by LLM content" question.
"""
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
OUT_MODEL = str(Path(__file__).resolve().parent / "dense_retriever_model")
BASE_MODEL = "BAAI/llm-embedder"  # BERT-base, retrieval-oriented pretraining, no MS MARCO
DEVICE = "cuda:0"
EPOCHS = 6
BATCH_SIZE = 32


def main():
    train_df = pd.read_csv(ROOT / "baseline" / "data" / "train.tsv", sep="\t")
    train_df["text"] = train_df["text"].fillna("")
    human_rows = train_df[train_df.label == 0]

    queries = pd.read_csv(ROOT / "train" / "queries.train.tsv", sep="\t", header=None,
                           names=["qid", "q"])
    qmap = dict(zip(queries.qid, queries.q))

    examples = []
    for _, row in human_rows.iterrows():
        q = qmap.get(row.qid)
        if q is None or not row.text:
            continue
        examples.append(InputExample(texts=[q, row.text]))
    print(f"Training pairs (query, relevant passage): {len(examples)}", flush=True)

    print(f"Loading base model {BASE_MODEL} (no MS MARCO pretraining) on {DEVICE} ...", flush=True)
    model = SentenceTransformer(BASE_MODEL, device=DEVICE)

    train_loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"Fine-tuning for {EPOCHS} epochs...", flush=True)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(0.1 * len(train_loader) * EPOCHS),
        show_progress_bar=True,
        output_path=OUT_MODEL,
    )
    print(f"Saved fine-tuned retriever to {OUT_MODEL}", flush=True)


if __name__ == "__main__":
    main()
