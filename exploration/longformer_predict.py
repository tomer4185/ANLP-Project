#!/usr/bin/env python
"""
Bulk-prediction utility for the fine-tuned Longformer verse/chorus model.

Usage
-----
python longformer_bulk_predict.py \
       --model_dir Results/fine_tuned_longformer/lr2e-05_bs4_ep3_ctx_run-az29vs00 \
       [--no_context] [--batch_size 4] [--max_len 1024]

The script:
  • pulls the data via utils.preprocessing.get_data()
  • replicates the same tokenisation logic used in training
  • saves predictions (with softmax probs) to <model_dir>/predictions.csv
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
from pathlib import Path
import argparse, torch, pandas as pd
from tqdm.auto import tqdm

# your project layout
from utils.preprocessing import get_data, parse_lyrics_sections, prepare_data_for_training  # noqa: F401

from transformers import LongformerTokenizer
from models.longformer_finetune import (     # same file that defines the model
    LongformerSegmentClassifier,
    build_hf_dataset,                       # reuse helper if accessible
)

# ─── Fallback: lightweight replica of build_hf_dataset ────────────────────────
def _build_ds(df, tok, no_ctx, max_len):
    from datasets import Dataset
    ds = Dataset.from_pandas(df)

    def _tok(batch):
        if not no_ctx:
            enc = tok(batch["text"], batch["context"],
                      truncation="only_second",
                      padding="max_length",
                      max_length=max_len)
        else:
            enc = tok(batch["text"],
                      truncation=True,
                      padding="max_length",
                      max_length=max_len)
        enc["label"] = batch["label"]
        return enc

    ds = ds.map(_tok, batched=True,
                remove_columns=[c for c in ds.column_names
                                if c not in {"input_ids", "attention_mask", "label"}])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

# choose whichever helper is available
build_ds = build_hf_dataset if "build_hf_dataset" in globals() else _build_ds


# ─── Prediction routine ───────────────────────────────────────────────────────
@torch.no_grad()
def predict(loader, model, device):
    preds, probs, labels, song_ids = [], [], [], []
    for batch in tqdm(loader, desc="predict"):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attn_mask)
        p = torch.softmax(logits, dim=-1)

        preds.extend(p.argmax(dim=-1).cpu().tolist())
        probs.extend(p.cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
        # song_id isn’t used by the model, but keep it if present
        if "song_id" in batch:
            song_ids.extend(batch["song_id"].cpu().tolist())

    return preds, probs, labels, song_ids


# ─── CLI ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=Path, required=True,
                   help="Folder produced by your training helper")
    p.add_argument("--no_context", action="store_true",
                   help="Replicate the *segment-only* setting")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=1024)
    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1.  Load data ────────────────────────────────────────────────────────
    df = get_data(type="test")                       # returns [song_id,text,context,label]
    print(f"dataset size: {len(df):,}")

    # ── 2.  Tokeniser & model ───────────────────────────────────────────────
    tok = LongformerTokenizer.from_pretrained(args.model_dir)
    model = LongformerSegmentClassifier(
        model_name="allenai/longformer-base-4096",
        num_labels=2,
    ).to(device)
    state_dict = torch.load(args.model_dir / "best_longformer.pt",
                            map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── 3.  DataLoader ───────────────────────────────────────────────────────
    ds = build_ds(df, tok, args.no_context, args.max_len)
    loader = torch.utils.data.DataLoader(ds,
                                         batch_size=args.batch_size,
                                         shuffle=False)

    # ── 4.  Predict ──────────────────────────────────────────────────────────
    pred_labels, prob_array, true_labels, song_ids = predict(loader, model, device)

    # ── 5.  Assemble & save CSV ─────────────────────────────────────────────
    out_df = df.copy()
    out_df["pred_label"] = pred_labels
    out_df["prob_0"] = [p[0] for p in prob_array]
    out_df["prob_1"] = [p[1] for p in prob_array]

    out_path = args.model_dir / "predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✔ predictions saved to {out_path}\n"
          f"  accuracy = {(out_df.pred_label == out_df.label).mean():.3f}")

if __name__ == "__main__":
    main()
