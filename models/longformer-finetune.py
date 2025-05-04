#!/usr/bin/env python
"""Train & evaluate a Longformer-based segment classifier (verse vs. chorus).

Input files (JSON Lines):
    train.jsonl, valid.jsonl
Each line = {"raw": "[CLS] ... [SEP] ...", "labels": [0,1,0,...]}
    * labels must be integers 0 (verse) or 1 (chorus) and align with song parts.

Usage (example):
    python longformer_segment_trainer.py \
        --train_file data/train.jsonl \
        --valid_file data/valid.jsonl \
        --output_dir checkpoints/longformer

Adjust hyper‑parameters with the CLI flags below.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    LongformerTokenizer, LongformerModel, get_linear_schedule_with_warmup
)

# ────────────────────────────────────────────────────────────────────────────────
#  Data utils
# ────────────────────────────────────────────────────────────────────────────────

import re
SPECIAL_RE = re.compile(r"\s*\[(?:CLS|SEP)\]\s*", re.IGNORECASE)


def parse_song(raw: str) -> List[str]:
    """Split BERT‑formatted lyric into parts (drops [CLS]/[SEP])."""
    return [seg.strip().replace("\u2005", " ")  # remove hair‑spaces
            for seg in SPECIAL_RE.split(raw) if seg.strip()]


class SongPartsDataset(Dataset):
    """Iterable dataset that yields one part per item."""

    def __init__(self, path: Path, tokenizer: LongformerTokenizer, max_len: int = 1024):
        self.samples: List[Dict[str, Any]] = []   # flattered parts
        self.tokenizer = tokenizer
        self.max_len = max_len

        with path.open() as fh:
            for line in fh:
                obj = json.loads(line)
                parts = parse_song(obj["raw"])
                seg_txt = parts[0]  # first part is the segment
                label = obj["labels"]
                full_song_txt = "\n".join(parts[1:])
                self.samples.append({
                    "segment": seg_txt,
                    "context": full_song_txt,
                    "label": int(label)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["segment"], s["context"],
            padding="max_length", truncation=True, max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(s["label"], dtype=torch.long)
        }
        return item


# ────────────────────────────────────────────────────────────────────────────────
#  Model definition
# ────────────────────────────────────────────────────────────────────────────────

class LongformerSegmentClassifier(nn.Module):
    def __init__(self, model_name: str = "allenai/longformer-base-4096", num_labels: int = 2):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.longformer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Set global attention on first token (required)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(cls_emb)
        return logits


# ────────────────────────────────────────────────────────────────────────────────
#  Train / Eval helpers
# ────────────────────────────────────────────────────────────────────────────────

def accuracy(preds, labels):
    return (preds == labels).sum().item() / len(labels)


def run_epoch(model, loader, optimizer, scheduler, device, train=True):
    model.train() if train else model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    steps = 0
    bar = tqdm(loader, desc="train" if train else "eval", leave=False)

    for batch in bar:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(input_ids, attn_mask)
            loss = nn.functional.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += accuracy(preds, labels)
        steps += 1
        bar.set_postfix(loss=epoch_loss/steps, acc=epoch_acc/steps)

    return epoch_loss / steps, epoch_acc / steps


# ────────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path, default="../data/train.jsonl")
    parser.add_argument("--valid_file", type=Path, default="../data/valid.jsonl")
    parser.add_argument("--output_dir", type=Path, default="../checkpoints/longformer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer & datasets
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    train_ds = SongPartsDataset(args.train_file, tokenizer, max_len=args.max_len)
    valid_ds = SongPartsDataset(args.valid_file, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongformerSegmentClassifier()
    model.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    # Training loop
    best_valid_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        print(f"  train loss={train_loss:.4f} acc={train_acc:.4f}")

        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, valid_loader, optimizer, scheduler, device, train=False)
        print(f"  valid loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            ckpt_path = args.output_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved new best model → {ckpt_path}")

    print("Training complete. Best validation accuracy:", best_valid_acc)


if __name__ == "__main__":
    main()
