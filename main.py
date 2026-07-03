"""
main.py
FastAPI app for IMDB sentiment prediction.

Run with:
    uvicorn main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

from loader import load_artifacts
from preprocessing import clean_review

app = FastAPI(title="IMDB Sentiment Analysis API")

# Load everything once at startup
model, token, max_length, english_stops = load_artifacts()


class ReviewInput(BaseModel):
    review: str


@app.get("/")
def home():
    return {"message": "IMDB Sentiment Analysis API is running"}


@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    review = data.review

    # Pre-process input (same as notebook)
    filtered = clean_review(review, english_stops)

    # Tokenize + pad
    tokenize_words = token.texts_to_sequences(filtered)
    tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')

    # Predict
    result = model.predict(tokenize_words)

    if result[0][0] >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return {
        "review": review,
        "sentiment": sentiment,
        "score": float(result[0][0])
    }
