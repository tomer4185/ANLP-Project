# #!/usr/bin/env python
# """Train & evaluate a Longformer-based segment classifier (verse vs. chorus).
#
# Input files (JSON Lines):
#     train.jsonl, valid.jsonl
# Each line = {"raw": "[CLS] ... [SEP] ...", "labels": [0,1,0,...]}
#     * labels must be integers 0 (verse) or 1 (chorus) and align with song parts.
#
# Usage (example):
#     python longformer_segment_trainer.py \
#         --train_file data/train.jsonl \
#         --valid_file data/valid.jsonl \
#         --output_dir checkpoints/longformer
#
# Adjust hyper‑parameters with the CLI flags below.
# """
# import argparse
# import json
# from pathlib import Path
# from typing import List, Dict, Any
#
# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from tqdm.auto import tqdm
# from transformers import (
#     LongformerTokenizer, LongformerModel, get_linear_schedule_with_warmup
# )
#
# # ────────────────────────────────────────────────────────────────────────────────
# #  Data utils
# # ────────────────────────────────────────────────────────────────────────────────
#
# import re
# SPECIAL_RE = re.compile(r"\s*\[(?:CLS|SEP)\]\s*", re.IGNORECASE)
#
#
# def parse_song(raw: str) -> List[str]:
#     """Split BERT‑formatted lyric into parts (drops [CLS]/[SEP])."""
#     return [seg.strip().replace("\u2005", " ")  # remove hair‑spaces
#             for seg in SPECIAL_RE.split(raw) if seg.strip()]
#
#
# class SongPartsDataset(Dataset):
#     """Iterable dataset that yields one part per item."""
#
#     def __init__(self, path: Path, tokenizer: LongformerTokenizer, max_len: int = 1024):
#         self.samples: List[Dict[str, Any]] = []   # flattered parts
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#
#         with path.open() as fh:
#             for line in fh:
#                 obj = json.loads(line)
#                 parts = parse_song(obj["raw"])
#                 seg_txt = parts[0]  # first part is the segment
#                 label = obj["labels"]
#                 full_song_txt = "\n".join(parts[1:])
#                 self.samples.append({
#                     "segment": seg_txt,
#                     # "context": full_song_txt,
#                     "context": "",
#                     "label": int(label)
#                 })
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         s = self.samples[idx]
#         enc = self.tokenizer(
#             s["segment"], s["context"],
#             padding="max_length", truncation=True, max_length=self.max_len,
#             return_tensors="pt"
#         )
#         item = {
#             "input_ids": enc["input_ids"].squeeze(0),
#             "attention_mask": enc["attention_mask"].squeeze(0),
#             "label": torch.tensor(s["label"], dtype=torch.long)
#         }
#         return item
#
#
# # ────────────────────────────────────────────────────────────────────────────────
# #  Model definition
# # ────────────────────────────────────────────────────────────────────────────────
#
# class LongformerSegmentClassifier(nn.Module):
#     def __init__(self, model_name: str = "allenai/longformer-base-4096", num_labels: int = 2):
#         super().__init__()
#         self.longformer = LongformerModel.from_pretrained(model_name)
#         self.classifier = nn.Linear(self.longformer.config.hidden_size, num_labels)
#
#     def forward(self, input_ids, attention_mask):
#         # Set global attention on first token (required)
#         global_attention_mask = torch.zeros_like(input_ids)
#         global_attention_mask[:, 0] = 1
#
#         outputs = self.longformer(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             global_attention_mask=global_attention_mask,
#         )
#         cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
#         logits = self.classifier(cls_emb)
#         return logits
#
#
# # ────────────────────────────────────────────────────────────────────────────────
# #  Train / Eval helpers
# # ────────────────────────────────────────────────────────────────────────────────
#
# def accuracy(preds, labels):
#     return (preds == labels).sum().item() / len(labels)
#
#
# def run_epoch(model, loader, optimizer, scheduler, device, train=True):
#     model.train() if train else model.eval()
#     epoch_loss = 0.0
#     epoch_acc = 0.0
#     steps = 0
#     bar = tqdm(loader, desc="train" if train else "eval", leave=False)
#
#     for batch in bar:
#         input_ids = batch["input_ids"].to(device)
#         attn_mask = batch["attention_mask"].to(device)
#         labels = batch["label"].to(device)
#
#         with torch.set_grad_enabled(train):
#             logits = model(input_ids, attn_mask)
#             loss = nn.functional.cross_entropy(logits, labels)
#             preds = logits.argmax(dim=-1)
#
#             if train:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()
#
#         epoch_loss += loss.item()
#         epoch_acc += accuracy(preds, labels)
#         steps += 1
#         bar.set_postfix(loss=epoch_loss/steps, acc=epoch_acc/steps)
#
#     return epoch_loss / steps, epoch_acc / steps
#
#
# # ────────────────────────────────────────────────────────────────────────────────
# #  Main
# # ────────────────────────────────────────────────────────────────────────────────
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_file", type=Path, default="../data/train.jsonl")
#     parser.add_argument("--valid_file", type=Path, default="../data/valid.jsonl")
#     parser.add_argument("--output_dir", type=Path, default="../checkpoints/longformer")
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--batch_size", type=int, default=2)
#     parser.add_argument("--lr", type=float, default=2e-5)
#     parser.add_argument("--max_len", type=int, default=1024)
#     parser.add_argument("--warmup_ratio", type=float, default=0.1)
#     args = parser.parse_args()
#
#     args.output_dir.mkdir(parents=True, exist_ok=True)
#
#     # Tokenizer & datasets
#     tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
#     train_ds = SongPartsDataset(args.train_file, tokenizer, max_len=args.max_len)
#     valid_ds = SongPartsDataset(args.valid_file, tokenizer, max_len=args.max_len)
#
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
#     valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)
#
#     # Model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LongformerSegmentClassifier()
#     model.to(device)
#
#     # Optimizer & scheduler
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
#     total_steps = len(train_loader) * args.epochs
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#
#         num_warmup_steps=int(total_steps * args.warmup_ratio),
#         num_training_steps=total_steps,
#     )
#
#     # Training loop
#     best_valid_acc = 0.0
#     for epoch in range(1, args.epochs + 1):
#         print(f"\nEpoch {epoch}/{args.epochs}")
#         train_loss, train_acc = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
#         print(f"  train loss={train_loss:.4f} acc={train_acc:.4f}")
#
#         with torch.no_grad():
#             val_loss, val_acc = run_epoch(model, valid_loader, optimizer, scheduler, device, train=False)
#         print(f"  valid loss={val_loss:.4f} acc={val_acc:.4f}")
#
#         if val_acc > best_valid_acc:
#             best_valid_acc = val_acc
#             ckpt_path = args.output_dir / "best_model_no_context.pt"
#             torch.save(model.state_dict(), ckpt_path)
#             print(f"  ✓ Saved new best model → {ckpt_path}")
#
#     print("Training complete. Best validation accuracy:", best_valid_acc)
#
#
# if __name__ == "__main__":
#     main()


"""
Longformer fine‑tuning script for verse/chorus classification **with Weights & Biases tracking**.
The script now draws its training data directly from ``utils.preprocessing.get_data`` so that
both the BERT and Longformer pipelines share an identical preprocessing path.

Key differences from the original version
──────────────────────────────────────────
1. **W&B integration** – Metrics (loss/accuracy) and hyper‑parameters are logged to the project
   specified by ``--wandb_project``. Disable by omitting the flag.
2. **Data source** – Uses ``get_data()`` (returns a DataFrame with columns
   ``[text, context, label]``). The CLI flag ``--no_context`` controls whether the *context*
   column is concatenated to each segment when tokenising (mirrors the BERT script).
3. **Dataset implementation** – Replaced the custom JSONL reader with a lightweight
   ``HuggingFace datasets`` wrapper so we avoid writing intermediate files.
4. **Structure** – Factored out a ``LongformerFineTuner`` class for parity with the BERT
   implementation. The training loop is almost identical, easing side‑by‑side comparisons.

Example
───────
::

    python longformer-finetune-wandb.py \
        --epochs 3 \
        --batch_size 2 \
        --lr 2e-5 \
        --wandb_project lyrics-parts \
        --run_name longformer_ctx

Add ``--no_context`` to train the *segment‑only* baseline.
"""
import sys
sys.path.append('/cs/labs/daphna/tomer_yaacoby/pythonProjects/ANLP-Project')
import argparse
import os
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import pandas as pd
from datasets import Dataset
from transformers import (
    LongformerTokenizer,
    LongformerModel,
    get_linear_schedule_with_warmup,
)

# ────────────────────────────────────────────────────────────────────────────────
#  Data utilities
# ────────────────────────────────────────────────────────────────────────────────
from datetime import datetime
from pathlib import Path

def build_save_dir(base="/cs/labs/daphna/tomer_yaacoby/pythonProjects/ANLP-Project/Results/fine_tuned_longformer",
                   lr=None, bs=None, epochs=None, ctx_flag=None,
                   wandb_run=None) -> Path:
    """
    Returns a unique directory such as
    Results/fine_tuned_longformer/lr2e-05_bs4_ep3_ctx_run-az29vs00
    """
    name = f"lr{lr}_bs{bs}_ep{epochs}_{ctx_flag}"
    rid  = wandb_run.id if wandb_run else datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(base) / f"{name}_run-{rid}"

try:
    # if project structure has utils.preprocessing
    from utils.preprocessing import get_data
except ImportError:
    # fallback: same directory or PYTHONPATH adjusted outside
    from preprocessing import get_data  # type: ignore


def build_hf_dataset(df: pd.DataFrame, tokenizer, no_context: bool, max_len: int):
    """Convert a pandas DataFrame produced by ``utils.preprocessing.get_data`` into a
    tokenised 🤗 ``datasets.Dataset`` ready for a PyTorch ``DataLoader``.

    The key fix is that **we now explicitly copy the ``label`` column into the
    tokenised output**, so `set_format()` can expose it.  Previously the column
    disappeared when we dropped the original string features, triggering the
    ``ValueError: Columns ['label'] not in the dataset`` the user observed.
    """
    ds = Dataset.from_pandas(df)  # keeps the 'label' column

    def _tokenise(batch):
        if not no_context:
            enc = tokenizer(
                batch["text"],
                batch["context"],
                truncation="only_second",
                padding="max_length",
                max_length=max_len,
            )
        else:
            enc = tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )
        # 🔑  Preserve the ground‑truth
        enc["label"] = batch["label"]
        return enc


    ds = ds.map(_tokenise, batched=True, remove_columns=[c for c in ds.column_names if c not in {"input_ids", "attention_mask", "label"}])

    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

# ────────────────────────────────────────────────────────────────────────────────
#  Model definition
# ────────────────────────────────────────────────────────────────────────────────

class LongformerSegmentClassifier(nn.Module):
    def __init__(self, model_name: str = "allenai/longformer-base-4096", num_labels: int = 2):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.longformer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Global attention on [CLS]
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_emb)


# ────────────────────────────────────────────────────────────────────────────────
#  Trainer class (mirrors BERT version)
# ────────────────────────────────────────────────────────────────────────────────

class LongformerFineTuner:
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        max_len: int,
        wandb_project: str = "longformer",
        run_name: str = None,
    ):
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerSegmentClassifier(model_name, num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_len = max_len
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.run = wandb.init(
            project=wandb_project,
            name=run_name,
            config=dict(
                model_name=model_name,
                num_labels=num_labels,
                lr=self.learning_rate,
                batch_size=self.batch_size,
                epochs=self.epochs,
                max_len=self.max_len,
            ),
            mode="online" if wandb_project else "disabled",
        )

    # ── data prep ─────────────────────────────────────────────────────────────
    def prepare_data(self, train_df, valid_df, no_context: bool):
        train_ds = build_hf_dataset(train_df, self.tokenizer, no_context, self.max_len)
        valid_ds = build_hf_dataset(valid_df, self.tokenizer, no_context, self.max_len)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size)
        return train_loader, valid_loader

    # ── epoch helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
        return (preds == labels).sum().item() / len(labels)

    def _run_epoch(self, loader, scheduler, train: bool, epoch: int):
        self.model.train() if train else self.model.eval()
        running_loss, running_acc, steps = 0.0, 0.0, 0
        pbar = tqdm(loader, desc="train" if train else "eval", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.set_grad_enabled(train):
                logits = self.model(input_ids, attn_mask)
                loss = nn.functional.cross_entropy(logits, labels)
                preds = logits.argmax(dim=-1)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    scheduler.step()

            running_loss += loss.item()
            running_acc += self._accuracy(preds, labels)
            steps += 1
            pbar.set_postfix(loss=running_loss / steps, acc=running_acc / steps)

        epoch_loss = running_loss / steps
        epoch_acc = running_acc / steps
        split = "train" if train else "val"
        self.run.log({f"{split}/loss": epoch_loss, f"{split}/accuracy": epoch_acc}, step=epoch)
        return epoch_loss, epoch_acc

    # ── training loop ─────────────────────────────────────────────────────────
    def fit(self, train_loader, valid_loader, output_dir: Path):
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        best_val_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            self._run_epoch(train_loader, scheduler, train=True, epoch=epoch)
            with torch.no_grad():
                _, val_acc = self._run_epoch(valid_loader, scheduler, train=False, epoch=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_pt  = output_dir / "best_longformer.pt"
                torch.save(self.model.state_dict(), ckpt_pt)
                # keep tokenizer alongside the weights
                self.tokenizer.save_pretrained(output_dir)
                print(f"  ✓ Saved new best model → {ckpt_pt}")
        print("Training complete. Best validation accuracy:", best_val_acc)
        self.run.finish()


# ────────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine‑tune Longformer for verse/chorus classification")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--output_dir", type=Path, default=Path("../checkpoints/longformer"))
    p.add_argument("--wandb_project", type=str, default="longformer")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--no_context", action="store_true", help="Ignore song context (segment only)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Starting Longformer fine-tuning...")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # ── fetch & split data ────────────────────────────────────────────────────
    df = get_data()
    train_df = df.sample(frac=0.8, random_state=42)
    valid_df = df.drop(train_df.index)

    # ── trainer ───────────────────────────────────────────────────────────────
    run_name = f"lr:{args.lr} bs:{args.batch_size} epochs:{args.epochs} train-size:{len(train_df)}"
    ctx_suffix = " ctx" if not args.no_context else " no_ctx"
    trainer = LongformerFineTuner(
        model_name="allenai/longformer-base-4096",
        num_labels=2,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_len=args.max_len,
        wandb_project=args.wandb_project,
        run_name= f"{run_name}{ctx_suffix}",
    )

    train_loader, valid_loader = trainer.prepare_data(
        train_df,
        valid_df,
        no_context=args.no_context,
    )
    # ── build a unique checkpoint folder ───────────────────
    ctx_flag = "ctx" if not args.no_context else "no_ctx"
    save_dir = build_save_dir(lr = args.lr, bs = args.batch_size, epochs = args.epochs,
    ctx_flag = ctx_flag, wandb_run = trainer.run)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✔ checkpoints will be saved to → {save_dir}")

    trainer.fit(train_loader, valid_loader, save_dir)


if __name__ == "__main__":
    main()
