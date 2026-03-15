from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def train_dl_model(df):
    texts = df["clean_review"].values
    labels = df["label"].values
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    vocab_size = len(tokenizer.word_index) + 1 
    X = pad_sequences(sequences)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(LSTM(150))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    loss, acc = model.evaluate(X_test, y_test)  
    print("DL Accuracy:", acc)
    model.save(os.path.join(BASE_DIR, "models", "lstm_model.h5"))