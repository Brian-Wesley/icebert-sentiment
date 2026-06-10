# SentimentAnalysis_final.py
# Fine-tune IceBERT for 3-way Icelandic sentiment classification (neg/neu/pos)
#

import os
import math
import json
import random
import inspect
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed as hf_set_seed,
)

# -----------------------------
# 1) Configuration (edit these)
# -----------------------------
MODEL_NAME: str = "mideind/IceBERT"
CSV_PATH: str = "icelandic_sentiment_v1.2_shuffled.csv"
OUTPUT_DIR: str = "./icebert-sentiment-v1.4.2"

RANDOM_SEED: int = 42

TRAIN_SIZE: float = 0.80
VAL_SIZE: float = 0.10
TEST_SIZE: float = 0.10

MAX_LENGTH_PERCENTILE: float = 0.95
MAX_LENGTH_CAP: int = 512

PER_DEVICE_TRAIN_BATCH_SIZE: int = 16
PER_DEVICE_EVAL_BATCH_SIZE: int = 32
GRADIENT_ACCUMULATION_STEPS: int = 2
MAX_EPOCHS: int = 20

EARLY_STOPPING_PATIENCE: int = 4
EARLY_STOPPING_THRESHOLD: float = 0.001

LABEL_SMOOTHING_FACTOR: float = 0.05
WEIGHT_DECAY: float = 0.01
LEARNING_RATE: float = 3e-5

WARMUP_RATIO: float = 0.06

DO_HPARAM_SWEEP: bool = True
HPARAM_GRID = {
    "learning_rate": [2e-5, 3e-5, 5e-5],
    "weight_decay": [0.0, 0.01],
    "label_smoothing_factor": [0.0, 0.05],
}

# Auto-disable torch.compile on Windows (Triton not supported)
TORCH_COMPILE: bool = (
    torch.cuda.is_available()
    and sys.platform != "win32"
    and torch.__version__ >= "2.0.0"
)
MAX_GRAD_NORM: float = 1.0

LABEL_NAMES: List[str] = ["negative", "neutral", "positive"]


# -----------------------------
# 2) Utilities
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    df["label"] = df["label"].astype(int)

    bad = df[~df["label"].isin([0, 1, 2])]
    if len(bad) > 0:
        raise ValueError(f"Found labels outside {0,1,2} in CSV. Examples:\n{bad.head(5)}")

    df = (
        df.groupby("text", as_index=False)["label"]
        .agg(lambda s: int(s.mode().iloc[0]))
        .reset_index(drop=True)
    )
    return df


def load_dataset_splits(
    csv_path: str,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> DatasetDict:
    if not math.isclose(train_size + val_size + test_size, 1.0, abs_tol=1e-6):
        raise ValueError("TRAIN_SIZE + VAL_SIZE + TEST_SIZE must sum to 1.0")

    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)

    dataset = Dataset.from_pandas(df[["text", "label"]])
    dataset = dataset.cast_column("label", ClassLabel(names=LABEL_NAMES))

    split_1 = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
        stratify_by_column="label",
    )
    train_val = split_1["train"]
    test = split_1["test"]

    val_rel = val_size / (train_size + val_size)
    split_2 = train_val.train_test_split(
        test_size=val_rel,
        seed=seed,
        stratify_by_column="label",
    )
    train = split_2["train"]
    val = split_2["test"]

    return DatasetDict({"train": train, "validation": val, "test": test})


def pick_max_length(
    texts: List[str],
    tokenizer: AutoTokenizer,
    percentile: float,
    cap: int,
    batch_size: int = 512,
    min_len: int = 32,
) -> int:
    lengths: List[int] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, add_special_tokens=True, truncation=False, padding=False)
        lengths.extend([len(ids) for ids in enc["input_ids"]])

    p = int(np.percentile(lengths, percentile * 100))
    hard_cap = min(cap, getattr(tokenizer, "model_max_length", cap) or cap)
    max_len = max(min_len, min(p, hard_cap))
    return int(max_len)


def compute_class_weights(labels: List[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_labels).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"At least one class is missing: counts={counts.tolist()}")
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_warmup_steps(train_len: int, epochs: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    eff_bs = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size
    steps_per_epoch = math.ceil(train_len / max(1, eff_bs))
    total_steps = max(1, steps_per_epoch * max(1, int(epochs)))
    warmup_steps = int(total_steps * WARMUP_RATIO)
    return max(0, min(warmup_steps, total_steps - 1))


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1m = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    pm = precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
    rm = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
    return {
        "accuracy": acc,
        "f1_macro": f1m,
        "precision_macro": pm,
        "recall_macro": rm,
    }


def make_training_args(**kwargs) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self._ce_supports_label_smoothing = "label_smoothing" in inspect.signature(torch.nn.CrossEntropyLoss).parameters

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ls = float(getattr(self.args, "label_smoothing_factor", 0.0) or 0.0)

        loss_kwargs = {}
        if ls > 0 and self._ce_supports_label_smoothing:
            loss_kwargs["label_smoothing"] = ls

        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, **loss_kwargs)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def get_best_epoch_from_log_history(log_history: List[Dict], metric_key: str = "eval_f1_macro") -> Optional[float]:
    best_val = None
    best_epoch = None
    for row in log_history:
        if metric_key in row:
            v = row[metric_key]
            if best_val is None or v > best_val:
                best_val = v
                best_epoch = row.get("epoch")
    return best_epoch


@dataclass(frozen=True)
class RunConfig:
    learning_rate: float
    weight_decay: float
    label_smoothing_factor: float


def train_once(
    cfg: RunConfig,
    epochs: int,
    tok: DatasetDict,
    tokenizer: AutoTokenizer,
    class_weights: torch.Tensor,
    use_bf16: bool,
    use_fp16: bool,
) -> Tuple[float, float, float, str]:
    """Train one hyper-parameter configuration with early stopping on validation set."""
    print(f"  → Trying cfg={cfg} for max {epochs} epochs...")

    id2label = {i: n for i, n in enumerate(LABEL_NAMES)}
    label2id = {n: i for i, n in id2label.items()}
    config = AutoConfig.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES), id2label=id2label, label2id=label2id
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    warmup_steps = compute_warmup_steps(len(tok["train"]), epochs)

    tmp_dir = f"./tmp_hparam_{hash(frozenset(asdict(cfg).items())) % 1000000}"

    training_args = make_training_args(
        output_dir=tmp_dir,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        label_smoothing_factor=cfg.label_smoothing_factor,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=25,
        seed=RANDOM_SEED,
        bf16=use_bf16,
        fp16=use_fp16,
        torch_compile=TORCH_COMPILE,
        max_grad_norm=MAX_GRAD_NORM,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tok["train"],
        eval_dataset=tok["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
        ],
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    best_f1 = eval_metrics["eval_f1_macro"]
    best_acc = eval_metrics["eval_accuracy"]
    best_loss = eval_metrics["eval_loss"]

    best_epoch = get_best_epoch_from_log_history(trainer.state.log_history) or epochs
    info_str = f"lr={cfg.learning_rate}, wd={cfg.weight_decay}, ls={cfg.label_smoothing_factor}, best_epoch={best_epoch}"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"    → Best F1 = {best_f1:.4f}  (epoch {best_epoch})")
    return best_f1, best_acc, best_loss, info_str


def main() -> None:
    seed_everything(RANDOM_SEED)

    raw = load_dataset_splits(CSV_PATH, RANDOM_SEED, TRAIN_SIZE, VAL_SIZE, TEST_SIZE)
    print(f"Loaded splits: train={len(raw['train'])}  val={len(raw['validation'])}  test={len(raw['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_len = pick_max_length(raw["train"]["text"], tokenizer, MAX_LENGTH_PERCENTILE, MAX_LENGTH_CAP)
    print(f"Auto-selected max_length={max_len} (p{int(MAX_LENGTH_PERCENTILE*100)} capped at {MAX_LENGTH_CAP}).")

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)

    tok = raw.map(tok_fn, batched=True, remove_columns=["text"])
    tok = tok.rename_column("label", "labels")
    tok.set_format("torch")

    train_labels = list(raw["train"]["label"])
    class_weights = compute_class_weights(train_labels, len(LABEL_NAMES))
    print(f"Class weights (neg/neu/pos) = {class_weights.tolist()}")

    use_bf16 = bool(torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)
    print(f"Mixed precision: bf16={use_bf16} fp16={use_fp16}")
    print(f"torch.compile enabled: {TORCH_COMPILE} (auto-disabled on Windows)")

    global accuracy_metric, f1_metric, precision_metric, recall_metric
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    # Hyperparameter sweep (or default)
    if DO_HPARAM_SWEEP:
        print("\n=== Starting hyperparameter sweep (12 combinations) ===")
        best_score = -float("inf")
        chosen_cfg = RunConfig(LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING_FACTOR)
        chosen_epochs = MAX_EPOCHS

        for lr in HPARAM_GRID["learning_rate"]:
            for wd in HPARAM_GRID["weight_decay"]:
                for lsf in HPARAM_GRID["label_smoothing_factor"]:
                    cfg = RunConfig(learning_rate=lr, weight_decay=wd, label_smoothing_factor=lsf)
                    f1, acc, loss, info = train_once(
                        cfg=cfg,
                        epochs=MAX_EPOCHS,
                        tok=tok,
                        tokenizer=tokenizer,
                        class_weights=class_weights,
                        use_bf16=use_bf16,
                        use_fp16=use_fp16,
                    )
                    if f1 > best_score:
                        best_score = f1
                        chosen_cfg = cfg
                        print(f"  NEW BEST: {info} → F1={best_score:.4f}")

        print(f"\nHyperparameter sweep finished. Best config: {chosen_cfg} (F1={best_score:.4f})")
    else:
        chosen_cfg = RunConfig(float(LEARNING_RATE), float(WEIGHT_DECAY), float(LABEL_SMOOTHING_FACTOR))
        chosen_epochs = MAX_EPOCHS
        print("\nHyperparameter sweep disabled; using defaults:")
        print(f"  cfg={chosen_cfg}")
        print(f"  epochs={chosen_epochs}")

    # Final model on full train+val
    print("\nTraining final model on train+validation...")

    train_full_raw = concatenate_datasets([raw["train"], raw["validation"]])
    train_full_labels = list(raw["train"]["label"]) + list(raw["validation"]["label"])
    class_weights_full = compute_class_weights(train_full_labels, len(LABEL_NAMES))

    final_max_len = pick_max_length(train_full_raw["text"], tokenizer, MAX_LENGTH_PERCENTILE, MAX_LENGTH_CAP)
    print(f"Final max_length (on train+val) = {final_max_len}")

    def final_tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=final_max_len, padding=False)

    train_full = train_full_raw.map(final_tok_fn, batched=True, remove_columns=["text"])
    train_full = train_full.rename_column("label", "labels")
    train_full.set_format("torch")

    id2label = {i: n for i, n in enumerate(LABEL_NAMES)}
    label2id = {n: i for i, n in id2label.items()}

    final_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(LABEL_NAMES), id2label=id2label, label2id=label2id)
    final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=final_config)

    final_warmup_steps = compute_warmup_steps(len(train_full), chosen_epochs)

    final_args = make_training_args(
        output_dir=OUTPUT_DIR,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=chosen_cfg.learning_rate,
        weight_decay=chosen_cfg.weight_decay,
        label_smoothing_factor=chosen_cfg.label_smoothing_factor,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=chosen_epochs,
        lr_scheduler_type="cosine",
        warmup_steps=final_warmup_steps,
        logging_steps=25,
        seed=RANDOM_SEED,
        bf16=use_bf16,
        fp16=use_fp16,
        torch_compile=TORCH_COMPILE,
        max_grad_norm=MAX_GRAD_NORM,
    )

    final_trainer = WeightedTrainer(
        model=final_model,
        args=final_args,
        train_dataset=train_full,
        eval_dataset=tok["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights_full,
    )

    final_trainer.train()

    print("\nFinal evaluation on held-out test set:")
    test_metrics = final_trainer.evaluate(eval_dataset=tok["test"])
    for k, v in sorted(test_metrics.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    final_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Publication-quality visualization
    print("\nGenerating publication-quality training progress plots...")
    logs = final_trainer.state.log_history
    df_logs = pd.DataFrame(logs)
    train_logs = df_logs[df_logs['loss'].notna()]
    eval_logs = df_logs[df_logs['eval_loss'].notna()]

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].plot(train_logs['step'], train_logs['loss'], label='Training Loss', color='#1f77b4', linewidth=2)
    if not eval_logs.empty:
        axs[0].plot(eval_logs['step'], eval_logs['eval_loss'], label='Test Loss (monitoring)', color='#ff7f0e', marker='o')
    axs[0].set_xlabel('Training Steps')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training & Test Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    if not eval_logs.empty:
        axs[1].plot(eval_logs['epoch'], eval_logs['eval_accuracy'], label='Accuracy', color='#2ca02c', marker='s')
        axs[1].plot(eval_logs['epoch'], eval_logs['eval_f1_macro'], label='Macro F1', color='#9467bd', marker='^')
        axs[1].plot(eval_logs['epoch'], eval_logs['eval_precision_macro'], label='Macro Precision', color='#17becf')
        axs[1].plot(eval_logs['epoch'], eval_logs['eval_recall_macro'], label='Macro Recall', color='#d62728')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Score')
    axs[1].set_title('Test Metrics (monitoring)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.suptitle('IceBERT Final Training Progress — Icelandic Sentiment', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("training_progress_final.png", dpi=300, bbox_inches='tight')
    print("Publication plot saved as 'training_progress_final.png'")

    # Quick inference example - ROBUST handling for pipeline output format
    print("\nQuick inference examples on final model:")
    sentiment_pipeline = pipeline(
        "text-classification",
        model=OUTPUT_DIR,
        tokenizer=OUTPUT_DIR,
        device=0 if torch.cuda.is_available() else -1,
        top_k=3,
    )

    example_texts = [
        "Þetta er frábært veður í dag!",
        "Venjulegur dagur, ekkert sérstakt.",
        "Allt í lagi en gæti verið betra.",
        "Hræðilegt, algjörlega vonlaust.",
    ]
    for text in example_texts:
        results = sentiment_pipeline(text)

        # CRITICAL FIX: unwrap nested list that some transformers versions return
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            results = results[0]

        print(f"\nText: {text}")
        for pred in results:
            print(f"  {pred['label']}: {pred['score']:.4f}")

    # Enhanced metadata
    meta = {
        "model_name": MODEL_NAME,
        "csv_path": CSV_PATH,
        "seed": RANDOM_SEED,
        "splits": {"train": TRAIN_SIZE, "validation": VAL_SIZE, "test": TEST_SIZE},
        "max_length": final_max_len,
        "chosen_config": asdict(chosen_cfg),
        "chosen_epochs": chosen_epochs,
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps_final": int(final_warmup_steps),
        "class_weights_full": class_weights_full.tolist(),
        "class_distribution_train_val": {name: int(count) for name, count in zip(LABEL_NAMES, np.bincount(train_full_labels, minlength=3))},
        "test_metrics": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
        "visualization": "training_progress_final.png",
        "torch_compile_used": TORCH_COMPILE,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved final model + tokenizer to: {OUTPUT_DIR}")
    print("Saved enhanced run metadata to: run_meta.json")


if __name__ == "__main__":
    main()