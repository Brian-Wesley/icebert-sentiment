# %%
# Continued Fine-Tuning of IceBERT (from previous checkpoint) on Expanded Dataset

# Requirements same as before + scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import evaluate
from typing import Dict
from datasets import ClassLabel

# -----------------------------
# 1. Configuration
# -----------------------------
MODEL_NAME: str = "mideind/IceBERT"  # Only used for tokenizer
PREVIOUS_MODEL_DIR: str = "./icebert-sentiment-finetuned"  # ← Your old fine-tuned model
CSV_PATH: str = "icelandic_sentiment_v1.1.csv"  # Combined old + new data
RANDOM_SEED: int = 42

TRAIN_BATCH_SIZE: int = 32
EVAL_BATCH_SIZE: int = 32
LEARNING_RATE: float = 3e-5
NUM_EPOCHS: int = 20  # Often fewer epochs needed when continuing
WARMUP_STEPS: int = 50
WEIGHT_DECAY: float = 0.01
OUTPUT_DIR: str = "./icebert-sentiment-v1.1"

# -----------------------------
# 2. Load and split (same as before)
# -----------------------------
def load_and_split_dataset(csv_path: str) -> DatasetDict:
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df["label"] = df["label"].astype(int)

    dataset = Dataset.from_pandas(df[["text", "label"]])
    dataset = dataset.cast_column("label", ClassLabel(names=["negative", "neutral", "positive"]))

    dataset = dataset.train_test_split(test_size=0.15, stratify_by_column="label", seed=RANDOM_SEED)
    test_ds = dataset["test"]
    train_val = dataset["train"].train_test_split(test_size=0.17647, stratify_by_column="label", seed=RANDOM_SEED)

    return DatasetDict({
        "train": train_val["train"],
        "val": train_val["test"],
        "test": test_ds
    })

raw_datasets = load_and_split_dataset(CSV_PATH)

print(f"Sizes → Train: {len(raw_datasets['train'])}, Val: {len(raw_datasets['val'])}, Test: {len(raw_datasets['test'])}")
for split in ["train", "val", "test"]:
    counts = Counter(raw_datasets[split]["label"])
    print(f"{split.capitalize()} distribution: neg={counts[0]}, neu={counts[1]}, pos={counts[2]}")

train_labels = raw_datasets["train"]["label"]
class_counts = np.bincount(train_labels)
class_weights = len(train_labels) / (len(class_counts) * class_counts.astype(float))
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"Class weights: {class_weights.tolist()}")

# -----------------------------
# 3. Tokenization (hashing-safe)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples: Dict[str, list], tokenizer) -> Dict[str, list]:
    return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    fn_kwargs={"tokenizer": tokenizer},
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------
# 4. Metrics (unchanged)
# -----------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
    }

# -----------------------------
# 5. Weighted Trainer (unchanged)
# -----------------------------

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs["labels"]
        # avoid mutating Trainer's input dict
        inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**inputs)
        logits = outputs.logits

        loss = F.cross_entropy(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
        )
        return (loss, outputs) if return_outputs else loss

# -----------------------------
# 6. Load previous model & training setup
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(PREVIOUS_MODEL_DIR)  # ← Key line

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    report_to="none",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    fp16=True,
    lr_scheduler_type="cosine",
    logging_steps=10,
    seed=RANDOM_SEED,
)

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)],
)

# -----------------------------
# 7–10. Train, evaluate, plot, save (identical to previous script)
# -----------------------------
print("Starting continued training...")
trainer.train()

# ... (rest of evaluation, plots, save_model exactly as in the previous script)

trainer.save_model(OUTPUT_DIR)
print(f"Improved model saved to {OUTPUT_DIR}")