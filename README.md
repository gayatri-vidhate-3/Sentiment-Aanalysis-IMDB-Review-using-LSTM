# 🎬 IMDB Movie Review Sentiment Analysis — LSTM + FastAPI

An end-to-end sentiment classification system that predicts whether a movie review is **Positive** or **Negative**, built from scratch using an LSTM neural network and served through a lightweight FastAPI backend.

This project covers the complete lifecycle of an NLP model — from raw text preprocessing and vocabulary building, through training and evaluation, to packaging the trained model as a working, queryable REST API.

---

## 📌 Project Overview

Most tutorials stop at "here's the model, here's the accuracy." This project goes a step further: the trained LSTM model is decoupled from its notebook environment and wrapped into a standalone, production-style inference service — the same pattern used when shipping ML models behind a real API.

- **Dataset:** IMDB movie reviews (50,000 labeled reviews — positive/negative)
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
