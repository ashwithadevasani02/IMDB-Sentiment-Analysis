import streamlit as st
import joblib
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "ml_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))
st.title("IMDB Sentiment Analysis")
review = st.text_area("Enter Movie Review")
if st.button("Predict"):
    review_vec = vectorizer.transform([review])
    pred = model.predict(review_vec)
    if pred[0] == 1:
        st.success("Positive Review")
    else:
        st.error("Negative Review")