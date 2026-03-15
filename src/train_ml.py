from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def train_ml_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_review"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("ML Accuracy:", acc)
    joblib.dump(model, os.path.join(BASE_DIR, "models", "ml_model.pkl"))
    joblib.dump(vectorizer, os.path.join(BASE_DIR, "models", "vectorizer.pkl"))
    return model