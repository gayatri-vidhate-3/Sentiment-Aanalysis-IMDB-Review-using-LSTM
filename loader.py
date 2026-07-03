# """
# loader.py
# Loads the 4 saved artifacts (model, tokenizer, max_length, english_stops)
# from the models/ folder. Call load_artifacts() once when the app starts.
# """

# import pickle
# from tensorflow.keras.models import load_model

# MODEL_PATH = "models/best_lstm_model.h5"
# TOKENIZER_PATH = "models/tokenizer.pkl"
# MAX_LENGTH_PATH = "models/max_length.pkl"
# STOPWORDS_PATH = "models/english_stops.pkl"


# def load_artifacts():
#     model = load_model(MODEL_PATH)

#     with open(TOKENIZER_PATH, "rb") as f:
#         token = pickle.load(f)
#     print('Tokenizer loaded successfully.')    

#     with open(MAX_LENGTH_PATH, "rb") as f:
#         max_length = pickle.load(f)
#     print('Max length loaded successfully.')    

#     with open(STOPWORDS_PATH, "rb") as f:
#         english_stops = pickle.load(f)
#     print('English stopwords loaded successfully.')

#     print("Model, tokenizer, max_length, english_stops loaded successfully.")
#     return model, token, max_length, english_stops


############################################################

"""
loader.py
Rebuilds the model architecture in code and loads only the trained weights.
This avoids Keras version mismatches that break full-model (.keras) loading,
since weight loading by shape is far more stable across versions than
config-based reconstruction.
"""

import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

MODEL_WEIGHTS_PATH = "models/best_lstm_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
MAX_LENGTH_PATH = "models/max_length.pkl"
STOPWORDS_PATH = "models/english_stops.pkl"

# Architecture constants, read off from the original model config
EMBEDDING_DIM = 32
LSTM_UNITS = 64


def build_model(vocab_size, max_length):
    model = Sequential([
        Input(shape=(max_length,)),
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
        LSTM(LSTM_UNITS),
        Dense(1, activation='sigmoid')
    ])
    return model


def load_artifacts():
    with open(TOKENIZER_PATH, "rb") as f:
        token = pickle.load(f)
    print('Tokenizer loaded successfully.')

    with open(MAX_LENGTH_PATH, "rb") as f:
        max_length = pickle.load(f)
    print('Max length loaded successfully.')

    with open(STOPWORDS_PATH, "rb") as f:
        english_stops = pickle.load(f)
    print('English stopwords loaded successfully.')

    # vocab_size must match total_words from training: len(token.word_index) + 1
    vocab_size = len(token.word_index) + 1

    model = build_model(vocab_size, max_length)
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("Model architecture rebuilt and weights loaded successfully.")

    return model, token, max_length, english_stops