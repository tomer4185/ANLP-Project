# Lyrics Verse / Chorus Classification – BERT **&** Longformer

This project provides two sibling training scripts that fine‑tune Transformer encoders for deciding whether a lyric *segment* belongs to a **verse** (`0`) or **chorus** (`1`).

| Script                         | Backbone                       | Max Tokens | Recommended Batch |
| ------------------------------ | ------------------------------ | ---------- | ----------------- |
| `fine_tune_bert.py`            | `bert‑base‑uncased`            | 512        | 8                 |
| `longformer_finetune_wandb.py` | `allenai/longformer‑base‑4096` | 4 096      | 2                 |

Both scripts **share the exact same data pipeline, CLI interface, logging, and naming conventions**; the only real difference is the underlying model.

---

## 1  Set‑up

```bash
# clone (or just `cd` if you already are in the repo)
git clone <repo‑url> && cd ANLP-Project

# create an environment (conda or venv)
conda create -n lyrics python=3.10 -y
conda activate lyrics

# install requirements
pip install -r requirements.txt    # torch, transformers, wandb, datasets, tqdm, …
```

---

## 2  Data Format

`utils.preprocessing.get_data()` returns one `pandas.DataFrame` with columns:

| column    | dtype | description                         |
| --------- | ----- | ----------------------------------- |
| `text`    | str   | the segment (*target* to classify)  |
| `context` | str   | the rest of the song (can be blank) |
| `label`   | int   | 0 = verse, 1 = chorus               |

Each run performs an **80 / 20 random split** into train / validation – **no external files required**.

---

## 3  Common CLI Flags

| Flag              | Meaning                                             | Default                    |
| ----------------- | --------------------------------------------------- | -------------------------- |
| `--epochs`        | training epochs                                     | 1 (BERT) / 10 (Longformer) |
| `--batch_size`    | mini‑batch size                                     | 8 / 2                      |
| `--lr`            | AdamW learning‑rate                                 | 2e‑5                       |
| `--output_dir`    | where checkpoints are written                       | `../checkpoints/<model>`   |
| `--no_context`    | ignore the `context` column (segment‑only baseline) | *disabled*                 |
| `--wandb_project` | Weights & Biases project name (omit to disable)     | `"bert"` / `"longformer"`  |
| `--run_name`      | custom run name in W\&B                             | auto‑generated             |

Anything not exposed through the CLI (warm‑up ratio, grad‑clip, accumulation, etc.) is hard‑coded in the scripts—feel free to tweak.

---

## 4  Running BERT

```bash
python fine_tune_bert.py \
       --epochs 3 \
       --batch_size 8 \
       --lr 1e-5 \
       --wandb_project lyrics-parts \
       --run_name bert_ctx
# add --no_context for the segment‑only baseline
```

Generates a directory such as:

```
Results/fine_tuned_bert/lr1e-05_bs8_ep3_ctx_run-<id>/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
└── …
```

---

## 5  Running Longformer

```bash
python longformer_finetune_wandb.py \
       --epochs 3 \
       --batch_size 2 \
       --lr 2e-5 \
       --wandb_project lyrics-parts \
       --run_name longformer_ctx
```

Produces:

```
Results/fine_tuned_longformer/lr2e-05_bs2_ep3_ctx_run-<id>/
└── best_longformer.pt
```

---

## 6  Monitoring with Weights & Biases

If `--wandb_project` is provided the scripts will automatically:

* log **train / val loss & accuracy** per epoch
* snapshot all hyper‑parameters and the current Git SHA
* attach the best checkpoint as an *artifact*

```bash
wandb login                  # first‑time only
export WANDB_MODE=offline    # optional: run locally without uploading
```

---

## 7  Inference Helper

After training you can score new lyrics with the shared utility:

```python
from utils.inference import get_predictions
preds_df = get_predictions(model, dataloader, device="cuda")
```

The resulting CSV contains `id`, `pred_label`, `prob_0`, `prob_1`.

---

## 8  Re‑using a Checkpoint

```python
from transformers import BertForSequenceClassification, BertTokenizer
model = BertForSequenceClassification.from_pretrained(
            "Results/fine_tuned_bert/…")
tok   = BertTokenizer.from_pretrained("Results/fine_tuned_bert/…")
```

Replace the class with `LongformerSegmentClassifier` if loading the Longformer weights.

---

Happy fine‑tuning & have fun exploring your song data 🎵
