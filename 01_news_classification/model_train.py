# train_model.py
import os
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from text_preprocessor import TextPreprocessor


# Ensure resources are available
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# 1. Load Data
print("Loading data...")
df = pd.read_csv("01_news_classification/data/train.csv", names=["label", "title", "text"])
df["text"] = df["title"] + " " + df["text"]
df = df[["text", "label"]]


# 3. Create pipeline
pipeline = Pipeline([
    ("preprocess", TextPreprocessor()),
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# 4. Train/Test Split
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# 5. Fit pipeline
pipeline.fit(X_train, y_train)

# 6. Save pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/news_classifier_pipeline.pkl")
print("Model + Preprocessing saved as news_classifier_pipeline.pkl")

# 7. Evaluate
accuracy = pipeline.score(X_test, y_test)
print(f"Model trained! Test accuracy: {accuracy:.2%}")
