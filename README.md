# 📧 SpamGuard — Email Spam Detector

A machine learning project that detects spam emails using a **Naive Bayes classifier** with **TF-IDF features**, plus a slick dark-themed web UI for interactive demo.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-~98%25-brightgreen)

---

## 📁 Project Structure

```
spamguard/
├── model/
│   ├── train.py          # Train the ML model
│   ├── predict.py        # Run predictions (CLI / interactive)
│   └── spam_model.pkl    # Saved model (generated after training)
├── demo/
│   └── spam_detector_ui.html   # Interactive web UI (open in browser)
├── data/                 # Auto-downloaded during training
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/spamguard.git
cd spamguard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset from Kaggle

**Option A — Kaggle CLI (recommended):**
```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/
kaggle datasets download -d shantanudhakadd/email-spam-detection-dataset-classification
mkdir -p data && unzip *.zip -d data/
```

**Option B — Manual download:**
1. Go to → https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification
2. Download the CSV
3. Place it inside the `data/` folder

> 📓 This project is based on the Kaggle notebook:
> https://www.kaggle.com/code/yacermeftah/email-spam-detection/notebook

### 4. Train the model
```bash
python model/train.py
```
This will:
- Load the Kaggle **Spam Email Classification** dataset (`Category`, `Message` columns)
- Train a Multinomial Naive Bayes model with TF-IDF features (unigrams + bigrams)
- Print evaluation metrics
- Save the model to `model/spam_model.pkl`

**Expected output:**
```
Accuracy       : ~98.4%
              precision  recall  f1-score
ham              0.99     0.99     0.99
spam             0.95     0.96     0.96
```

### 4. Make predictions

**CLI:**
```bash
python model/predict.py "CONGRATULATIONS! You've won £1,000,000. Reply NOW!"
```

**Interactive mode:**
```bash
python model/predict.py
```

---

## 🌐 Web UI Demo

Open `demo/spam_detector_ui.html` directly in your browser — no server needed.

Features:
- Paste any email text and get an instant spam/ham verdict
- Confidence score with animated progress bar
- Signal analysis explaining the decision
- Sample emails (spam & ham) to test with
- Session history tracking

> Note: The web UI uses a lightweight rule-based classifier for the browser demo. The Python model (`train.py`) is the full ML implementation.

---

## 🧠 Model Details

| Component | Choice |
|-----------|--------|
| Algorithm | Multinomial Naive Bayes |
| Features  | TF-IDF (unigrams + bigrams, top 10k) |
| Dataset   | UCI SMS Spam Collection (5,572 msgs) |
| Train/Test split | 80/20 stratified |
| Accuracy  | ~98.4% |
| F1 (spam) | ~0.96 |

### Why Naive Bayes?
- Fast to train, fast to predict
- Works extremely well on text classification
- Handles high-dimensional sparse feature spaces (TF-IDF) naturally
- Interpretable and lightweight

---

## 📊 Dataset

The [Spam Email Classification Dataset](https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification) from Kaggle contains tagged email messages:

| Column | Description |
|--------|-------------|
| `Category` | `"spam"` or `"ham"` |
| `Message` | Email body text |

Based on the notebook: https://www.kaggle.com/code/yacermeftah/email-spam-detection/notebook

> The dataset must be downloaded manually from Kaggle (see Quick Start above). It is not included in this repo due to Kaggle's terms of service.

---

## 🛠 Future Improvements

- [ ] Flask/FastAPI REST API wrapper
- [ ] Deep learning model (LSTM / BERT fine-tune)
- [ ] Connect web UI to Python backend
- [ ] Streamlit dashboard
- [ ] Docker container

---

## 📄 License

MIT License — feel free to use, modify, and share.
