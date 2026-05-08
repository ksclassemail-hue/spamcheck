"""
SpamGuard — Email Spam Classifier
Based on: https://www.kaggle.com/code/yacermeftah/email-spam-detection/notebook
Dataset : "Spam email classification" on Kaggle
         (kaggle datasets download -d shantanudhakadd/email-spam-detection-dataset-classification)

The dataset CSV has two columns:
    Category  →  "spam" or "ham"
    Message   →  email body text

Model: TF-IDF + Multinomial Naive Bayes (same approach as the notebook)
"""

import os
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)

# ── 1. Locate dataset ─────────────────────────────────────────────────────────
# Expected path after downloading via Kaggle CLI:
#   kaggle datasets download -d shantanudhakadd/email-spam-detection-dataset-classification
#   unzip *.zip -d data/
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Try common filenames used by this dataset
CANDIDATE_FILES = ["spam.csv", "emails.csv", "mail_data.csv", "dataset.csv"]
CSV_PATH = None
for fname in CANDIDATE_FILES:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        CSV_PATH = path
        break

if CSV_PATH is None:
    print("=" * 60)
    print("Dataset not found in ./data/")
    print()
    print("Please download it from Kaggle:")
    print("  1. Install Kaggle CLI:  pip install kaggle")
    print("  2. Place your kaggle.json API key in ~/.kaggle/")
    print("  3. Run:")
    print("     kaggle datasets download -d shantanudhakadd/email-spam-detection-dataset-classification")
    print("     mkdir -p data && unzip *.zip -d data/")
    print()
    print("Or download manually from:")
    print("  https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification")
    print("  and place the CSV inside the ./data/ folder.")
    print("=" * 60)
    sys.exit(1)

print(f"Using dataset: {CSV_PATH}")

# ── 2. Load & inspect ─────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"\nColumns found : {list(df.columns)}")

# Normalise column names (the Kaggle notebook uses 'Category' and 'Message')
df.columns = [c.strip().lower() for c in df.columns]

# Support alternate column names
if "category" in df.columns:
    label_col = "category"
elif "label" in df.columns:
    label_col = "label"
else:
    label_col = df.columns[0]

if "message" in df.columns:
    text_col = "message"
elif "text" in df.columns:
    text_col = "text"
else:
    text_col = df.columns[1]

print(f"Label column  : '{label_col}'")
print(f"Text  column  : '{text_col}'")

df = df[[label_col, text_col]].dropna()
df.columns = ["label", "text"]
df["label"] = df["label"].str.strip().str.lower()
df["label_bin"] = (df["label"] == "spam").astype(int)

print(f"\nDataset shape : {df.shape}")
print(f"Spam messages : {df['label_bin'].sum()} ({df['label_bin'].mean()*100:.1f}%)")
print(f"Ham  messages : {(1-df['label_bin']).sum()} ({(1-df['label_bin'].mean())*100:.1f}%)\n")

# ── 3. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label_bin"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_bin"],
)

# ── 4. Pipeline: TF-IDF → Naive Bayes ────────────────────────────────────────
# Matches the approach used in the yacermeftah notebook
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=10_000,
        sublinear_tf=True,
    )),
    ("clf", MultinomialNB(alpha=0.1)),
])

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("Training model...")
pipeline.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"Accuracy       : {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
print("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, y_pred)
print(f"               Ham   Spam")
print(f"  Actual Ham : {cm[0][0]:4d}  {cm[0][1]:4d}")
print(f"  Actual Spam: {cm[1][0]:4d}  {cm[1][1]:4d}")

# ── 7. Save model ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"\nModel saved → {MODEL_PATH}")
