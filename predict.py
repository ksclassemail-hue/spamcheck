"""
SpamGuard — Inference Script
Load the trained model and predict on new email/SMS text.

Usage:
    python predict.py "Your text here"
    python predict.py   # interactive mode
"""

import os
import pickle
import sys

MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_model.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Please run `python model/train.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(model, text: str) -> dict:
    proba = model.predict_proba([text])[0]
    label = model.predict([text])[0]
    return {
        "text"       : text[:80] + ("…" if len(text) > 80 else ""),
        "prediction" : "SPAM" if label == 1 else "HAM",
        "confidence" : round(max(proba) * 100, 1),
        "spam_prob"  : round(proba[1] * 100, 1),
        "ham_prob"   : round(proba[0] * 100, 1),
    }


def pretty_print(result: dict):
    tag = "🚨 SPAM" if result["prediction"] == "SPAM" else "✅  HAM"
    print(f"\n  {tag}  — {result['confidence']}% confidence")
    print(f"  Spam prob: {result['spam_prob']}%  |  Ham prob: {result['ham_prob']}%")
    print(f"  Text: \"{result['text']}\"\n")


if __name__ == "__main__":
    model = load_model()

    if len(sys.argv) > 1:
        # single prediction from command-line argument
        text   = " ".join(sys.argv[1:])
        result = predict(model, text)
        pretty_print(result)
    else:
        # interactive loop
        print("SpamGuard — Interactive Mode  (type 'exit' to quit)\n")
        while True:
            text = input("Enter email text: ").strip()
            if text.lower() in ("exit", "quit", "q"):
                break
            if text:
                pretty_print(predict(model, text))
