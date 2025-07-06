import sys
sys.path.append('/cs/labs/daphna/tomer_yaacoby/pythonProjects/ANLP-Project')

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
from datasets import Dataset
from utils.preprocessing import get_data
import os
import argparse
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import math
from datetime import datetime
import uuid

def build_save_dir(base="Results/fine_tuned_bert",
                   lr=None, bs=None, epochs=None, ctx_flag=None,
                   wandb_run=None):
    """
    Return a unique directory path like:
    Results/fine_tuned_bert/lr2e-05_bs8_ep3_ctx_run-a1b2c3
    """
    # ① fixed part
    name = f"lr{lr}_bs{bs}_ep{epochs}_{ctx_flag}"

    # ② unique run-ID: prefer W&B run.id, otherwise YYYYmmdd-HHMMSS
    run_id = wandb_run.id if wandb_run else datetime.now().strftime("%Y%m%d-%H%M%S")
    name += f"_run-{run_id}"

    return Path(base) / name

# Set your seed
seed = 42
g = torch.Generator()
g.manual_seed(seed)

# Dummy dataset
x = torch.randn(100, 2)
dataset = TensorDataset(x)

def pick_gpu(required_gib=10):
    required = required_gib * 1024 ** 3           # bytes
    best_id, best_free = None, -1
    for idx in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(idx)  # PyTorch ≥1.9
        if free >= required and free > best_free:
            best_id, best_free = idx, free
    if best_id is None:
        raise RuntimeError(f"No GPU has {required_gib} GiB free")
    return best_id

# gpu_id = pick_gpu(required_gib=12)      # your helper
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # must be *before* torch loads CUDA
# torch.cuda.set_device(0)

class CustomBertModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomBertModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.to(self.device)
        # Set up the global variables for wandb and training

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def save_pretrained(self, save_directory):
        self.bert.save_pretrained(save_directory)

class BertFineTuner:
    def __init__(self, model_name, num_labels, learning_rate=5e-5, batch_size=8, epochs=10, wandb_project="bert", run_name=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = CustomBertModel(model_name, num_labels)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.global_step = 0  # counts *batches*
        self.log_every_n = 100  # change to taste
        self.gradient_update_steps = 32  # for gradient accumulation
        self.run = wandb.init(
            project=wandb_project,
            name=run_name,
            config=dict(
                model_name=model_name,
                num_labels=num_labels,
                lr=self.learning_rate,
                batch_size=self.batch_size,
                epochs=self.epochs
            )
        ) if wandb_project else None
        wandb.define_metric("epoch")  # master axis
        for m in ["train/loss", "train/accuracy","val/loss", "val/accuracy"]:
            wandb.define_metric(m, step_metric="epoch")

    def prepare_data(self, train_data, test_data, no_context=False):
        #make example dataset:
        train_dataset = Dataset.from_pandas(train_data)
        test_dataset = Dataset.from_pandas(test_data)

        if not no_context:
            train_dataset = train_dataset.map(lambda x: self.tokenizer(x['text'], x['context'], truncation='only_second', padding='max_length', max_length=512), batched=True)
            test_dataset = test_dataset.map(lambda x: self.tokenizer(x['text'], x['context'], truncation='only_second', padding='max_length', max_length=512), batched=True)
        else:
            train_dataset = train_dataset.map(
                lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length',
                                         max_length=512), batched=True)
            test_dataset = test_dataset.map(
                lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length',
                                         max_length=512), batched=True)

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        return DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, generator=g), DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True)

    def train(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            labels = batch['label'].to(self.model.device)

            if (step) % self.gradient_update_steps == 0:
                self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss / self.gradient_update_steps  # ★ scale ★
            loss.backward()

            update_now = (step + 1) % self.gradient_update_steps == 0 or (step + 1) == len(train_loader)
            if update_now:
                # Perform gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # Update the model parameters
                self.optimizer.step()


            running_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            self.global_step += 1
            # wandb.log({"train/loss": loss.item()}, step=self.global_step)

            # (optional) throttle console prints
            if self.global_step % self.log_every_n == 0:
                print(
                    f"epoch {epoch} step {self.global_step} loss {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        wandb.log({"epoch": epoch}, commit=False)  # MUST be first
        wandb.log({"train/loss": epoch_loss,
                   "train/accuracy": epoch_acc})

    def evaluate(self, test_loader, epoch):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['label'].to(self.model.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.append(predictions.cpu())
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Test Loss: {avg_loss}, Accuracy: {accuracy}")
        wandb.log({"epoch": epoch}, commit=False)
        wandb.log({"val/loss": avg_loss,
                   "val/accuracy": accuracy})
        # save the predictions to df
        # predictions = torch.cat(all_predictions).numpy()
        # test_df = pd.DataFrame(predictions, columns=["predictions"])
        # test_df.to_parquet("./data/test_predictions.parquet")

def get_predictions(model, dataloader, device="cuda",
                    save_path="predictions.csv"):
    """
    Run the model on `dataloader`, return a DataFrame and (optionally) save it.

    Returned columns
    ---------------
    id              : text identifier (whatever was in batch["id"])
    pred_label      : arg-max class
    prob_0, prob_1  : class-level softmax probabilities  (extend if >2 classes)
    """
    model.eval()
    ids, preds, prob_0, prob_1 = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_ids      = batch["id"]                     # stay on CPU

            # ─── forward pass ────────────────────────────────────────────
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits
            probs  = torch.softmax(logits, dim=-1)

            # ─── collect ────────────────────────────────────────────────
            preds.extend(torch.argmax(probs, dim=-1).cpu().tolist())
            prob_0.extend(probs[:, 0].cpu().tolist())
            prob_1.extend(probs[:, 1].cpu().tolist())
            ids.extend(batch_ids)

    df = pd.DataFrame({
        "id": ids,
        "pred_label": preds,
        "prob_0": prob_0,
        "prob_1": prob_1,
    })

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"✔ Saved predictions to {save_path}")

    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_context", action="store_true", help="Enable context using")
    # Add other arguments here as needed
    parser.add_argument("--output_dir", type=Path, default="../checkpoints/bert")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting BERT fine-tuning...")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    no_context = args.no_context
    model_name = 'bert-base-uncased'
    num_labels = 2
    dataset_name = 'mrYou/Lyrics_eng_dataset'

    run_name = f"lr:{args.lr} bs:{args.batch_size} epochs:{args.epochs}"
    ctx_suffix = " ctx" if not args.no_context else " no_ctx"

    bert_fine_tuner = BertFineTuner(model_name, num_labels, run_name=run_name + ctx_suffix)
    data = get_data()
    # split the data into train and test sets
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)
    # save the data to parquet
    # if not os.path.exists('./data'):
    #     os.makedirs('./data')
    # train_data.to_parquet("./data/train_data.parquet")
    # test_data.to_parquet("./data/test_data.parquet")
    train_loader, test_loader = bert_fine_tuner.prepare_data(train_data, test_data, no_context=no_context)

    for epoch in range(args.epochs):
        bert_fine_tuner.train(train_loader, epoch)
        bert_fine_tuner.evaluate(test_loader, epoch)

    # Save the model
    # ---------------------------------------------
    # Build and create unique save directory
    ctx_flag = "ctx" if not args.no_context else "no_ctx"
    save_dir = build_save_dir(
        lr=args.lr,
        bs=args.batch_size,
        epochs=args.epochs,
        ctx_flag=ctx_flag,
        wandb_run=bert_fine_tuner.run  # may be None if wandb_project=""
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✔ Checkpoints will be saved to: {save_dir}")
    # ---------------------------------------------

    # Save model & tokenizer
    bert_fine_tuner.model.save_pretrained(save_dir)
    bert_fine_tuner.tokenizer.save_pretrained(save_dir)

    ############################################################################

    # Load the model and tokenizer
    # Specify the directory where the model and tokenizer were saved
    # save_directory = './Results/fine_tuned_bert'
    #
    # # Load the tokenizer
    # tokenizer = BertTokenizer.from_pretrained(save_directory)
    #
    # # Load the model
    # model = BertForSequenceClassification.from_pretrained(save_directory)
    #
    # # Example usage
    # text = "This is a test sentence."
    # inputs = tokenizer(text, return_tensors="pt", truncation=True,
    #                    padding="max_length", max_length=512)
    # outputs = model(**inputs)
    # print(outputs)

