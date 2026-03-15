import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def clean_text(text):
    text = text.casefold()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
def load_data(path=None):
    if path is None:
      path = os.path.join(BASE_DIR, "dataset", "IMDB_Dataset.csv")
    df = pd.read_csv(path)
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = df["sentiment"].map({"positive":1,"negative":0})
    return df