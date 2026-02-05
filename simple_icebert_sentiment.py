"""
simple_icebert_sentiment.py

A simple, interactive script for running sentiment analysis on Icelandic text
using the fine-tuned IceBERT model.

Features:
- Easy text input from the console.
- Clear, formatted output with probabilities for all labels.
- Loops until you type 'quit'.
- Automatically uses GPU if available.
- Handles common loading quirks from the model.

"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ----------------------------------------------------------------------
# Configuration - customize only if needed
# ----------------------------------------------------------------------
MODEL_PATH: str = "./icebert-sentiment-v1.1"  # Path to the saved model

LABEL_NAMES: List[str] = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def load_pipeline() -> pipeline:
    """Load tokenizer, model, fix labels, and create the sentiment pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        ignore_mismatched_sizes=True,  # Handles beta/gamma ↔ weight/bias warnings
    )

    # Fix human-readable labels
    model.config.id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    model.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    model.config.num_labels = len(LABEL_NAMES)

    device = 0 if torch.cuda.is_available() else -1

    sentiment_pipe = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
        function_to_apply="softmax",
    )
    return sentiment_pipe

def format_predictions(preds: List[dict]) -> str:
    """Format a list of label-score dicts into a nice string."""
    lines = []
    for pred in sorted(preds, key=lambda x: x["score"], reverse=True):
        lines.append(f"  {pred['label']:8} : {pred['score']:.4f}")
    return "\n".join(lines)

def main() -> None:
    """Main interactive loop."""
    print("IceBERT Icelandic Sentiment Analyzer")
    print("Enter text in Icelandic (or 'quit' to exit)\n")

    classifier = load_pipeline()

    while True:
        user_input = input("Your text: ").strip()

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter some text.\n")
            continue

        # Run prediction
        results = classifier(user_input)

        # Normalize output (handles single vs batch format)
        preds = results[0] if isinstance(results[0], list) else results

        print("\nSentiment prediction:")
        print(format_predictions(preds))
        print()  # blank line for readability

if __name__ == "__main__":

    main()

