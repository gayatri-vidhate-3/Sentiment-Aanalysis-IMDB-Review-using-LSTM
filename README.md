# 🎬 IMDB Movie Review Sentiment Analysis — LSTM + FastAPI

An end-to-end sentiment classification system that predicts whether a movie review is **Positive** or **Negative**, built from scratch using an LSTM neural network and served through a lightweight FastAPI backend.

This project covers the complete lifecycle of an NLP model — from raw text preprocessing and vocabulary building, through training and evaluation, to packaging the trained model as a working, queryable REST API.

---

## 📌 Project Overview

Most tutorials stop at "here's the model, here's the accuracy." This project goes a step further: the trained LSTM model is decoupled from its notebook environment and wrapped into a standalone, production-style inference service — the same pattern used when shipping ML models behind a real API.

- **Dataset:** IMDB movie reviews (seperate Train and Test files with labeled reviews — positive/negative)
- **Model:** Embedding → LSTM → Dense (binary sigmoid output)
- **Serving layer:** FastAPI, with the trained model, tokenizer, and preprocessing pipeline loaded once at startup for fast repeated inference
- **Interface:** REST endpoint returning sentiment + confidence score for any input review

---

## 🧠 Model Architecture

| Layer      | Configuration                          |
|------------|------------------------------------------|
| Embedding  | vocab_size ≈ 92,546 → 32-dimensional vectors |
| LSTM       | 64 units                                  |
| Dense      | 1 unit, sigmoid activation (binary output)|

**Input:** padded/truncated sequences of length 130
**Output:** probability score between 0–1 → thresholded at 0.5 for Positive/Negative

---

## 🗂️ Repository Structure

```
imdb-sentiment-analysis/
├── notebook/
│   └── IMDB_sentiment_Analysis_project.ipynb   # data prep, EDA, tokenizer fit, training
├── models/
│   ├── model.weights.h5        # trained LSTM weights
│   ├── tokenizer.pkl           # fitted Keras Tokenizer (word to index mapping)
│   ├── max_length.pkl          # padding length used during training
│   └── english_stops.pkl       # stopword set used in preprocessing
├── loader.py                   # rebuilds model architecture + loads trained weights/tokenizer
├── preprocessing.py            # text cleaning: strip non-alphabets, remove stopwords, lowercase
├── main.py                     # FastAPI app exposing the /predict endpoint
├── requirements.txt
└── README.md
```

## ⚙️ How It Works

1. **Training (notebook):** Reviews are cleaned, tokenized, and converted into padded integer sequences. An LSTM model learns to map these sequences to a sentiment probability.
2. **Artifact export:** Rather than saving only the model, the fitted `Tokenizer`, `max_length`, and stopword set are also persisted — since inference requires the *exact same* word-to-index vocabulary used at training time. Skipping this is a common mistake that silently breaks predictions.
3. **Model reconstruction:** Instead of relying on Keras's full `.keras` config-based reload (which is sensitive to Keras version drift between training and deployment environments), the architecture is rebuilt explicitly in code and only the trained **weights** are loaded. This makes deployment robust even when the serving environment's Keras version differs from the training environment's.
4. **Serving:** FastAPI loads all artifacts once at startup (not per-request) and exposes a `/predict` endpoint that runs the same cleaning → tokenize → pad → predict pipeline used during training.

---

## 🚀 Getting Started

### 1. Clone & set up environment
```bash
git clone <repo-url>
cd imdb-sentiment-analysis
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API
```bash
uvicorn main:app
```
> Note: `--reload` is intentionally avoided here — TensorFlow's memory footprint combined with uvicorn's file-watcher subprocess can cause `MemoryError` on machines with limited RAM.

### 4. Test it
Open **http://127.0.0.1:8000/docs** for interactive Swagger UI, or call directly:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely fantastic, I loved every minute of it!"}'
```

**Response:**
```json
{
  "review": "This movie was absolutely fantastic, I loved every minute of it!",
  "sentiment": "Positive",
  "score": 0.9234
}
```

---

## 🛠️ Tech Stack

- **Python 3.10**
- **TensorFlow / Keras 3** — model training & inference
- **FastAPI** — REST API layer
- **Uvicorn** — ASGI server
- **Pydantic** — request validation

---

## 💡 Key Engineering Notes

- Preprocessing at inference time is **kept identical** to training-time preprocessing (same cleaning function, same stopword set, same tokenizer) — a mismatch here is one of the most common silent failure modes in deployed NLP models.
- Model weights are loaded into a **manually rebuilt architecture** rather than via full model deserialization, avoiding brittleness from Keras config format changes across versions.
- Artifacts (model weights, tokenizer, max_length, stopwords) are loaded **once at application startup**, not per-request, for low-latency predictions.

---

## 📈 Possible Extensions

- Batch prediction endpoint for scoring multiple reviews in one call
- Model versioning / A-B testing between architectures
- Dockerizing the service for deployment
- Swapping LSTM for a transformer-based encoder (e.g., DistilBERT) as a comparison baseline

---

## 👤 Author

**Gayatri Vidhate**
Data Scientist | NLP Engineer | ML Engineer | GenAI Engineer 
