#!/usr/bin/env python
"""
Transformer-based clickbait classifier (BERT-style), mirroring naiveBayes.py structure.

- Uses the same data loader: utility.dataLoader.load_texts_labels
- Deterministic 80/20 split (train/test) with seed 42
- Optional train/validation split inside the 80% train slice
- Fine-tunes a Hugging Face transformer (default: bert-base-uncased)
- Reuses a NaiveBayes-style evaluate() function on the held-out test set
"""

import sys
import argparse
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# -----------------------------------------------------------------------------
# Repo root and dataset loader (same pattern as naiveBayes.py)
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.dataLoader import load_texts_labels as load_texts_labels_unified  # noqa: E402


def get_texts_labels_for(dataset: str):
    """Wrapper to mirror naiveBayes.py."""
    return load_texts_labels_unified(dataset)


# -----------------------------------------------------------------------------
# Metrics helper (adapted from naiveBayes.evaluate)
# -----------------------------------------------------------------------------

def evaluate(gold, pred, tag: str, y_score=None):
    """Print accuracy, precision/recall, confusion matrix, and extended metrics."""
    acc = accuracy_score(gold, pred)
    prec = precision_score(gold, pred, pos_label=1, zero_division=0)
    rec = recall_score(gold, pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=[0, 1])
    print(f"[{tag}] Acc={acc:.3f}  Prec(pos=1)={prec:.3f}  Rec(pos=1)={rec:.3f}")
    print(f"[{tag}] Confusion Matrix:\n{cm}")

    if y_score is not None:
        try:
            p_c, r_c, f1_c, supp_c = precision_recall_fscore_support(
                gold, pred, labels=[0, 1], zero_division=0
            )
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                gold, pred, average="macro", zero_division=0
            )
            p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
                gold, pred, average="micro", zero_division=0
            )
            try:
                roc_auc = roc_auc_score(gold, y_score)
            except Exception:
                roc_auc = None
            try:
                pr_auc = average_precision_score(gold, y_score)
            except Exception:
                pr_auc = None

            print(f"[{tag}] Per-class:")
            print(
                f"  class 0: prec={p_c[0]:.3f} rec={r_c[0]:.3f} "
                f"f1={f1_c[0]:.3f} support={supp_c[0]}"
            )
            print(
                f"  class 1: prec={p_c[1]:.3f} rec={r_c[1]:.3f} "
                f"f1={f1_c[1]:.3f} support={supp_c[1]}"
            )
            print(
                f"[{tag}] Macro: prec={p_macro:.3f} "
                f"rec={r_macro:.3f} f1={f1_macro:.3f}"
            )
            print(
                f"[{tag}] Micro: prec={p_micro:.3f} "
                f"rec={r_micro:.3f} f1={f1_micro:.3f}"
            )
            print(f"[{tag}] ROC-AUC: {'N/A' if roc_auc is None else f'{roc_auc:.3f}'}")
            print(f"[{tag}] PR-AUC:  {'N/A' if pr_auc is None else f'{pr_auc:.3f}'}")
        except Exception as e:
            print(f"[{tag}] Extended metrics error: {e}")


# -----------------------------------------------------------------------------
# Torch Dataset and tokenization
# -----------------------------------------------------------------------------

class ClickbaitDataset(Dataset):
    """Simple torch Dataset wrapping tokenized texts and labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize_texts(tokenizer, texts, max_length: int):
    """Batch-tokenize a list of texts."""
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


# -----------------------------------------------------------------------------
# Training / evaluation pipeline (transformer analogue of train_and_evaluate_naive_bayes)
# -----------------------------------------------------------------------------

def train_and_evaluate_transformer(
    dataset: str,
    X_texts,
    y,
    model_name: str = "bert-base-uncased",
    max_length: int = 64,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    output_dir: str = "transformer_clickbait",
):
    """
    Transformer analogue of train_and_evaluate_naive_bayes:

    - Deterministic shuffle with seed
    - 80/20 train/test split
    - Optional small validation split from train
    - Fine-tune a transformer model
    - Evaluate on held-out test set with the same metrics as evaluate()
    """
    if not X_texts or not y:
        print(f"[{dataset}] No data loaded; skipping.")
        return

    set_seed(seed)

    # Deterministic shuffle (same spirit as naiveBayes.py)
    docs = list(zip(X_texts, y))
    rnd = random.Random(seed)
    rnd.shuffle(docs)
    X_texts, y = zip(*docs)

    n = len(X_texts)
    if n == 0:
        print(f"[{dataset}] No data after shuffle; skipping.")
        return

    # 80/20 split for train+val / test
    k = max(1, int(0.8 * n))
    train_texts_full = X_texts[:k]
    train_labels_full = y[:k]
    test_texts = X_texts[k:]
    test_labels = y[k:]

    # Log test distribution (to mirror naiveBayes.py)
    pos_test = sum(1 for v in test_labels if v == 1)
    neg_test = len(test_labels) - pos_test
    print(
        f"[{dataset}] Test split distribution: "
        f"pos={pos_test} neg={neg_test} (n={len(test_labels)})"
    )

    # Create a small validation set from the training slice (e.g., 10% of train)
    if len(train_texts_full) > 10:
        val_size = max(1, int(0.1 * len(train_texts_full)))
    else:
        val_size = 1 if len(train_texts_full) > 1 else 0

    if val_size > 0:
        train_texts = train_texts_full[:-val_size]
        train_labels = train_labels_full[:-val_size]
        val_texts = train_texts_full[-val_size:]
        val_labels = train_labels_full[-val_size:]
    else:
        train_texts = train_texts_full
        train_labels = train_labels_full
        val_texts, val_labels = [], []

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # Tokenization
    train_encodings = tokenize_texts(tokenizer, train_texts, max_length=max_length)
    if val_texts:
        val_encodings = tokenize_texts(tokenizer, val_texts, max_length=max_length)
    else:
        val_encodings = None
    test_encodings = tokenize_texts(tokenizer, test_texts, max_length=max_length)

    # Datasets
    train_dataset = ClickbaitDataset(train_encodings, train_labels)
    eval_dataset = (
        ClickbaitDataset(val_encodings, val_labels) if val_encodings is not None else None
    )
    test_dataset = ClickbaitDataset(test_encodings, test_labels)

    # Compute metrics for Trainer (simple accuracy + macro F1)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        # Macro F1 over (0,1)
        _, _, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {"accuracy": acc, "f1_macro": f1_macro}

    # Training arguments
    tag_safe = dataset.replace(" ", "_")
    run_output_dir = f"{output_dir}/{dataset}_{model_name.replace('/', '_')}"

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="f1_macro",
        logging_dir=f"{run_output_dir}/logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    # Train
    print(f"[{dataset}] Starting transformer fine-tuning on {model_name}")
    trainer.train()

    # Evaluate on validation (if exists)
    if eval_dataset is not None:
        val_metrics = trainer.evaluate()
        print(f"[{dataset}] Validation metrics: {val_metrics}")

    # Final evaluation on test set (with full metrics)
    print(f"[{dataset}] Evaluating on held-out test set...")
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    test_labels_arr = np.array(test_labels, dtype=int)
    preds = np.argmax(logits, axis=-1)

    # Convert logits to probabilities for class 1
    # logits shape: (N, 2) for binary classification
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_score = probs[:, 1]

    tag = (
        f"{dataset}][Transformer][model={model_name}]"
        f"[max_len={max_length}][epochs={num_epochs}]"
    )
    evaluate(test_labels_arr, preds, tag, y_score=y_score)


# -----------------------------------------------------------------------------
# Driver (mirrors run_for_dataset / main in naiveBayes.py)
# -----------------------------------------------------------------------------

def run_for_dataset(dataset: str, args):
    print(f"\n=== {dataset.capitalize()} (Transformer) ===")
    X_texts, y = get_texts_labels_for(dataset)
    train_and_evaluate_transformer(
        dataset=dataset,
        X_texts=X_texts,
        y=y,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transformer-based clickbait classifier (BERT-style)"
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "kaggle", "train2", "webis"],
        default="all",
        help="Dataset to run",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model name (e.g., bert-base-uncased)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="transformer_clickbait",
        help="Base output directory for checkpoints and logs",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["kaggle", "train2", "webis"]:
            run_for_dataset(ds, args)
    else:
        run_for_dataset(args.dataset, args)


if __name__ == "__main__":
    main()