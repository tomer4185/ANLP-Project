#!/usr/bin/env python
"""
Bulk-prediction utility for a fine-tuned **BERT** verse/chorus model.

Usage
-----
python bert_bulk_predict.py \
       --model_dir Results/fine_tuned_bert/my_run \
       [--no_context] [--batch_size 8] [--max_len 512]

The script:
  • pulls the test split via utils.preprocessing.get_data()
  • applies the same (pair-sentence) tokenisation used at train time
  • writes predictions (label + softmax probs) to <model_dir>/predictions.csv
"""
# ─── Imports ────────────────────────────────────────────────────────────────
from pathlib import Path
import argparse, torch, pandas as pd
from tqdm.auto import tqdm

from utils.preprocessing import get_data  # you already have this

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ─── Dataset helper (mirrors the Longformer script) ─────────────────────────
def build_ds(df, tok, no_ctx, max_len):
    """
    Converts a DataFrame with columns [text, context, label, …] into a
    🤗 datasets.Dataset of tensors that BERT can consume.
    """
    from datasets import Dataset
    ds = Dataset.from_pandas(df)

    def _tok(batch):
        if not no_ctx:
            enc = tok(
                batch["text"],
                batch["context"],
                truncation="only_second",
                padding="max_length",
                max_length=max_len,
            )
        else:
            enc = tok(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )
        # keep the gold label for accuracy calc
        enc["label"] = batch["label"]
        return enc

    ds = ds.map(
        _tok,
        batched=True,
        remove_columns=[
            c
            for c in ds.column_names
            if c
            not in {"input_ids", "attention_mask", "token_type_ids", "label"}
        ],
    )
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )
    return ds


# ─── Prediction loop ────────────────────────────────────────────────────────
@torch.no_grad()
def predict(loader, model, device):
    preds, probs, labels, song_ids = [], [], [], []

    for batch in tqdm(loader, desc="predict"):
        # send tensors to GPU/CPU
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in {"input_ids", "attention_mask", "token_type_ids"}
        }
        logits = model(**inputs).logits
        p = torch.softmax(logits, dim=-1)

        preds.extend(p.argmax(dim=-1).cpu().tolist())
        probs.extend(p.cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
        if "song_id" in batch:  # keep for traceability
            song_ids.extend(batch["song_id"].cpu().tolist())

    return preds, probs, labels, song_ids


# ─── CLI parsing ────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=Path, required=True,
                   help="Folder that holds config.json / model.safetensors / …")
    p.add_argument("--no_context", action="store_true",
                   help="Replicate the *segment-only* setting")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=512,
                   help="BERT’s maximum sequence length")
    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load test data ────────────────────────────────────────────────────
    df = get_data(type="test")  # → DataFrame[text, context, label, song_id?]
    print(f"dataset size: {len(df):,}")

    # 2. Tokeniser & model ────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    #   AutoModelForSequenceClassification will pick up:
    #   • the BERT backbone specified in config.json
    #   • the number of labels (num_labels) from that same config
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    # 3. DataLoader ───────────────────────────────────────────────────────
    ds = build_ds(df, tok, args.no_context, args.max_len)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False
    )

    # 4. Predict ──────────────────────────────────────────────────────────
    pred_labels, prob_array, true_labels, song_ids = predict(
        loader, model, device
    )

    # 5. Assemble & save CSV ─────────────────────────────────────────────
    out_df = df.copy()
    out_df["pred_label"] = pred_labels
    out_df["prob_0"] = [p[0] for p in prob_array]
    out_df["prob_1"] = [p[1] for p in prob_array]

    out_path = args.model_dir / "predictions.csv"
    out_df.to_csv(out_path, index=False)
    acc = (out_df.pred_label == out_df.label).mean()
    print(f"✔ predictions saved to {out_path}")
    print(f"  accuracy = {acc:.3f}")


if __name__ == "__main__":
    main()
