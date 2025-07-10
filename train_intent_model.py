# train_intent_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load your training data
df = pd.read_parquet("train.parquet")

# Convert winner column to binary label: A → 1, B → 0
df["label"] = df["winner"].apply(lambda x: 1 if x == "A" else 0)

# Vectorize only the prompt
intent_vectorizer = TfidfVectorizer(max_features=3000)
prompt_vectors = intent_vectorizer.fit_transform(df["prompt"])

# Train Logistic Regression on prompt vectors
intent_model = LogisticRegression()
intent_model.fit(prompt_vectors, df["label"])

# Save both model and vectorizer
joblib.dump(intent_model, "intent_model.pkl")
joblib.dump(intent_vectorizer, "intent_vectorizer.pkl")

print("✅ intent_model.pkl and intent_vectorizer.pkl saved.")
