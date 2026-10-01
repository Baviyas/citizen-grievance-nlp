"""Fine-tune the local sentiment transformer on Indian grievance examples.

The existing sentiment artifact has no positive validation examples. This
script adds balanced authored examples for positive, neutral, negative, and
critical sentiment, evaluates on a stratified holdout, and saves the model
under ``models/india_sentiment_model``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation" / "india_sentiment_examples.csv"
LEGACY_DATA = ROOT / "data" / "processed" / "grievance_processed.csv"
BASE_MODEL = ROOT / "models" / "final_models" / "sentiment_model"
OUTPUT_DIR = ROOT / "models" / "india_sentiment_model"
METRICS = ROOT / "evaluation" / "india_sentiment_metrics.json"
LABELS = ["positive", "neutral", "negative", "critical"]


class TokenizedSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer):
        self.labels = frame["label_id"].astype(int).tolist()
        self.encodings = tokenizer(
            frame["text"].astype(str).tolist(),
            truncation=True,
            max_length=128,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def main() -> None:
    curated = pd.read_csv(DATA).dropna(subset=["text", "label"])
    legacy = pd.read_csv(LEGACY_DATA, low_memory=False)
    legacy = legacy[["clean_text", "sentiment_label"]].rename(
        columns={"clean_text": "text", "sentiment_label": "label"}
    ).dropna()
    legacy["text"] = legacy["text"].astype(str)
    legacy["label"] = legacy["label"].astype(str)
    legacy = legacy[legacy["label"].isin(LABELS)]
    legacy = pd.concat(
        [
            group.sample(min(len(group), 3000), random_state=42)
            for _, group in legacy.groupby("label")
        ],
        ignore_index=True,
    )
    df = pd.concat([curated, legacy], ignore_index=True)
    df["label_id"] = df["label"].map({label: i for i, label in enumerate(LABELS)})
    curated_train, curated_test = train_test_split(
        curated,
        test_size=0.25,
        random_state=42,
        stratify=curated["label"],
    )
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df["label_id"],
    )
    # Guarantee that every authored sentiment is represented in validation.
    test_df = pd.concat([test_df, curated_test], ignore_index=True).drop_duplicates(
        subset=["text", "label"]
    )
    train_df = pd.concat([train_df, curated_train], ignore_index=True).drop_duplicates(
        subset=["text", "label"]
    )
    train_df["label_id"] = train_df["label"].map(
        {label: i for i, label in enumerate(LABELS)}
    )
    test_df["label_id"] = test_df["label"].map(
        {label: i for i, label in enumerate(LABELS)}
    )
    positive_train = train_df[train_df["label"].eq("positive")]
    if len(positive_train) < 100:
        train_df = pd.concat(
            [train_df, positive_train.sample(100, replace=True, random_state=42)],
            ignore_index=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE_MODEL),
        local_files_only=True,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    train_ds = TokenizedSentimentDataset(train_df, tokenizer)
    test_ds = TokenizedSentimentDataset(test_df, tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro"),
        }

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=10,
        report_to=[],
        seed=42,
        no_cuda=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    result = {
        "base_model": str(BASE_MODEL),
        "dataset_type": "curated India-oriented sentiment examples",
        "labels": LABELS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "eval_accuracy": float(metrics["eval_accuracy"]),
        "eval_f1_macro": float(metrics["eval_f1_macro"]),
    }
    METRICS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
