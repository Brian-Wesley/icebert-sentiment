"""
icebert_inference.py

A complete, standalone script for loading and using your fine-tuned IceBERT
sentiment analysis model on Icelandic text.

Features:
- Loads the model and tokenizer from the saved directory.
- Handles common LayerNorm key mismatches (beta/gamma ↔ weight/bias).
- Manually sets human-readable label names (customize to your training labels).
- Uses Hugging Face pipeline for simple, robust inference.
- Supports both single-text and batch inference.
- Returns scores for *all* labels (return_all_scores=True).
- Runs on GPU if CUDA is available.
- Safe output handling for single/batch cases.

Author: Brian Wesley
Date: January 2026
"""

from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_PATH: str = "./icebert-sentiment-finetuned"


LABEL_NAMES: List[str] = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def load_sentiment_pipeline(model_path: str = MODEL_PATH) -> pipeline:
    """
    Load the fine-tuned IceBERT model and tokenizer, then create a sentiment-analysis pipeline.

    Args:
        model_path (str): Directory containing the saved model and tokenizer.

    Returns:
        pipeline: Ready-to-use Hugging Face pipeline for sentiment analysis.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model, ignoring harmless LayerNorm key mismatches
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True,  # Suppresses beta/gamma warnings
    )

    # Manually set human-readable labels (critical for correct output)
    model.config.id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    model.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    model.config.num_labels = len(LABEL_NAMES)

    # Determine device (GPU if available)
    device = 0 if torch.cuda.is_available() else -1

    # Create pipeline
    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,        # Always return scores for all labels
        function_to_apply="softmax",   # Proper probabilities
    )

    return sentiment_pipeline


# ----------------------------------------------------------------------
# Load the pipeline once
# ----------------------------------------------------------------------
classifier = load_sentiment_pipeline()

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Single text example
    single_text: str = "Þessi veitingastaður er frábær!"
    print("Single prediction:")
    results = classifier(single_text)

    # Normalize output format (handles single vs batch differences)
    preds = results[0] if isinstance(results[0], list) else results
    for pred in preds:
        print(f"  {pred['label']}: {pred['score']:.4f}")

    # Batch example
    texts: List[str] = [
        "Ég elska þessa mynd, hún er ótrúlega góð!",
        "Þetta var hræðilegt, aldrei aftur.",
        "Allt í lagi, ekkert sérstakt.",
    ]

    print("\nBatch predictions:")
    batch_results = classifier(texts)

    for text, result in zip(texts, batch_results):
        print(f"\nText: {text}")
        preds = result if isinstance(result, list) else [result]
        for pred in preds:
            print(f"  {pred['label']}: {pred['score']:.4f}")