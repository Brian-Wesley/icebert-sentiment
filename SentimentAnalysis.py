# %%
# Fine-Tuning IceBERT for 3-Way Sentiment Analysis on Icelandic Texts
# Author: Brian Wesley
# Description: Complete, optimized script with post-training visualization using matplotlib
# Visualizes training progress after trainer.train()

# Requirements:
# !pip install -q transformers datasets evaluate accelerate sentencepiece pandas torch matplotlib
# --index-url https://download.pytorch.org/whl/cu121  (for CUDA 12.1 if needed)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate
from typing import Dict
from datasets import ClassLabel

# -----------------------------
# 1. Configuration
# -----------------------------
MODEL_NAME: str = "mideind/IceBERT"
CSV_PATH: str = "icelandic_sentiment.csv"  # Columns: "text" (str), "label" (int 0=neg, 1=neu, 2=pos)
TRAIN_TEST_SPLIT: float = 0.2
RANDOM_SEED: int = 42

# Optimized hyperparameters for best results on ~2,700 examples
TRAIN_BATCH_SIZE: int = 32
EVAL_BATCH_SIZE: int = 32
LEARNING_RATE: float = 3e-5
NUM_EPOCHS: int = 30  # High max; early stopping will trigger earlier
WARMUP_STEPS: int = 100
WEIGHT_DECAY: float = 0.01
OUTPUT_DIR: str = "./icebert-sentiment-finetuned"


# -----------------------------
# 2. Load and prepare dataset
# -----------------------------

def load_custom_dataset(csv_path: str) -> DatasetDict:
    """
    Load CSV → Hugging Face Dataset → cast label to ClassLabel → stratified train/test split.
    """
    df = pd.read_csv(csv_path)

    # Basic validation
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Ensure label is integer (defensive)
    df["label"] = df["label"].astype(int)

    # Optional but helpful: quick sanity check on label values
    unique_labels = sorted(df["label"].unique())
    if set(unique_labels) != {0, 1, 2}:
        print(f"Warning: Labels should be 0, 1, 2. Found: {unique_labels}")

    # Create Dataset
    dataset = Dataset.from_pandas(df[["text", "label"]])

    # ─── The crucial fix ───
    # Convert plain integer column → ClassLabel (required for stratification)
    dataset = dataset.cast_column(
        "label",
        ClassLabel(names=["negative", "neutral", "positive"])
    )

    # Now stratified split works
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

# -----------------------------
# 3. Tokenization
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples: Dict[str, list]) -> Dict[str, list]:
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------
# 4. Metrics
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
# 5. Model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
)

# -----------------------------
# 6. Training setup
# -----------------------------
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
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    bf16=False,
    torch_compile=False,
    torch_compile_backend="eager",
    lr_scheduler_type="cosine",
    dataloader_num_workers=0,
    logging_steps=10,
    seed=RANDOM_SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)],
)

# -----------------------------
# 7. Train & Evaluate
# -----------------------------
print("Starting training...")
trainer.train()

# -----------------------------
# 8. Save model
# -----------------------------
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# -----------------------------
# 9. Visualize training progress
# -----------------------------
print("\nGenerating training progress plots...")

logs = trainer.state.log_history
df_logs = pd.DataFrame(logs)

# Separate train and eval logs
train_logs = df_logs[df_logs['loss'].notna()]
eval_logs = df_logs[df_logs['eval_loss'].notna()]

plt.figure(figsize=(14, 5))

# Plot 1: Loss curves
plt.subplot(1, 2, 1)
plt.plot(train_logs['step'], train_logs['loss'], label='Training Loss', color='blue', alpha=0.7, linewidth=1.2)
if not eval_logs.empty and 'eval_loss' in eval_logs:
    plt.plot(eval_logs['step'], eval_logs['eval_loss'], label='Validation Loss', color='orange', marker='o',
             linewidth=1.8)
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Validation metrics
plt.subplot(1, 2, 2)
if not eval_logs.empty:
    if 'eval_accuracy' in eval_logs:
        plt.plot(eval_logs['step'], eval_logs['eval_accuracy'], label='Validation Accuracy', color='green', marker='o',
                 linewidth=1.8)
    if 'eval_f1_macro' in eval_logs:
        plt.plot(eval_logs['step'], eval_logs['eval_f1_macro'], label='Validation Macro F1', color='purple', marker='o',
                 linewidth=1.8)
plt.xlabel('Training Steps')
plt.ylabel('Score')
plt.title('Validation Metrics over Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save plot to disk
plt.savefig("training_progress.png", dpi=150, bbox_inches='tight')
print("Plot saved as 'training_progress.png'")

# -----------------------------
# 10. Quick inference example
# -----------------------------
from transformers import pipeline

sentiment_pipeline = pipeline(
    "text-classification",
    model=OUTPUT_DIR,
    tokenizer=tokenizer,
    return_all_scores=True,
)

example_texts = [
    "Þetta er frábært veður í dag!",
    "Venjulegur dagur, ekkert sérstakt.",
    "Allt í lagi en gæti verið betra.",
    "Hræðilegt, algjörlega vonlaust.",
]

print("\nInference examples:")
for text in example_texts:
    scores = sentiment_pipeline(text)[0]

    # Force scores to always be a list of dicts (handles single-example case)
    if isinstance(scores, dict):
        scores = [scores]
    elif not isinstance(scores, list):
        scores = []  # or handle error, but empty is safe

    # Now sort safely
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

    print(f"\nText: {text}")
    for pred in sorted_scores:
        label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        print(f"  {label_map[pred['label']]}: {pred['score']:.3f}")