# %%
# Detailed Post-Training Evaluation of Fine-Tuned IceBERT Sentiment Model
# Author: Brian Wesley (adapted from training script)
# Runs on the saved model to provide thorough accuracy analysis

# Additional requirement (if not already installed):
# !pip install -q scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import evaluate
from typing import Dict
from datasets import ClassLabel

# -----------------------------
# Configuration (match your training script)
# -----------------------------
MODEL_NAME: str = "mideind/IceBERT"
FINETUNED_DIR: str = "./icebert-sentiment-finetuned"   # Directory where trainer.save_model() saved the model
CSV_PATH: str = "icelandic_sentiment.csv"             # Same CSV used for training
TRAIN_TEST_SPLIT: float = 0.2
RANDOM_SEED: int = 42
EVAL_BATCH_SIZE: int = 32

# -----------------------------
# 1. Load and split dataset (identical to training)
# -----------------------------
def load_custom_dataset(csv_path: str) -> DatasetDict:
    """Load CSV, validate, cast label to ClassLabel, and perform stratified split."""
    df = pd.read_csv(csv_path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    df["label"] = df["label"].astype(int)

    unique_labels = sorted(df["label"].unique())
    if set(unique_labels) != {0, 1, 2}:
        print(f"Warning: Expected labels 0, 1, 2. Found: {unique_labels}")

    dataset = Dataset.from_pandas(df[["text", "label"]])

    # Cast to ClassLabel for proper stratification
    dataset = dataset.cast_column(
        "label",
        ClassLabel(names=["negative", "neutral", "positive"])
    )

    dataset = dataset.train_test_split(
        test_size=TRAIN_TEST_SPLIT,
        seed=RANDOM_SEED,
        stratify_by_column="label"
    )

    return DatasetDict({
        "train": dataset["train"],
        "test": dataset["test"]
    })

raw_datasets = load_custom_dataset(CSV_PATH)
print(f"Loaded {len(raw_datasets['train'])} train | {len(raw_datasets['test'])} test examples.")

# Test set class distribution
print("\nTest set class distribution:")
label_counts = Counter(raw_datasets["test"]["label"])
target_names = ["negative", "neutral", "positive"]
for i in range(3):
    print(f"  {target_names[i]}: {label_counts.get(i, 0)}")

# -----------------------------
# 3. Tokenization (fixed version — no more hashing warning)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Pure function: tokenizer is now an explicit parameter → fully pickleable
def tokenize_function(examples: Dict[str, list], tokenizer) -> Dict[str, list]:
    """
    Tokenize a batch of examples.

    Args:
        examples: Batch dict with "text" key.
        tokenizer: The pretrained tokenizer instance.

    Returns:
        Dict with tokenized fields.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,  # padding handled by DataCollatorWithPadding later
    )


# Apply the map with explicit tokenizer argument
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    fn_kwargs={"tokenizer": tokenizer},  # ← this is the crucial fix
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------
# 3. Load fine-tuned model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR)

# -----------------------------
# 4. Metrics (same as training)
# -----------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
    }

# -----------------------------
# 5. Trainer for evaluation only
# -----------------------------
training_args = TrainingArguments(
    output_dir="./eval_temp",      # temporary, not used for saving
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------------
# 6. Overall metrics on test set
# -----------------------------
print("\n=== Overall Test Metrics ===")
eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# -----------------------------
# 7. Detailed classification report & confusion matrices
# -----------------------------
print("\n=== Generating Detailed Report & Confusion Matrices ===")
predict_results = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predict_results.predictions, axis=-1)
true_labels = predict_results.label_ids

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=target_names, digits=4))

# Confusion matrix (counts)
cm = confusion_matrix(true_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix (Counts)")
plt.tight_layout()
plt.savefig("confusion_matrix_counts.png", dpi=150, bbox_inches='tight')
plt.show()

# Normalized confusion matrix (row-wise = recall per class)
cm_norm = confusion_matrix(true_labels, preds, normalize="true")
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=target_names)
disp_norm.plot(cmap=plt.cm.Blues, values_format='.2f')
plt.title("Normalized Confusion Matrix (Row = True Class)")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nEvaluation complete! Plots saved as:")
print("  - confusion_matrix_counts.png")
print("  - confusion_matrix_normalized.png")