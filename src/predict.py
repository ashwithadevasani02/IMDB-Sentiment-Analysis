import joblib
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "ml_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)
    if pred[0] == 1:
        return "Positive Review"
    else:
        return "Negative Review"