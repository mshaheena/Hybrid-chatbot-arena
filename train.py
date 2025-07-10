# train.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Load your dataset
train_df = pd.read_parquet("train.parquet")

# Label encoding: A = 1, B = 0
train_df["label"] = train_df["winner"].apply(lambda x: 1 if x == "A" else 0)

# --- Train Intent Model (Logistic Regression on Prompt) ---
intent_vectorizer = TfidfVectorizer(max_features=3000)
prompt_vectors = intent_vectorizer.fit_transform(train_df["prompt"])
intent_model = LogisticRegression()
intent_model.fit(prompt_vectors, train_df["label"])

# Save intent model + vectorizer
joblib.dump(intent_model, "intent_model.pkl")
joblib.dump(intent_vectorizer, "intent_vectorizer.pkl")

# --- Train Response Scorer (XGBoost on Prompt + Responses) ---
combined_texts = train_df["prompt"] + " " + train_df["response_a"] + " " + train_df["response_b"]
response_vectorizer = TfidfVectorizer(max_features=5000)
X = response_vectorizer.fit_transform(combined_texts)
y = train_df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# Save response scorer + vectorizer
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(response_vectorizer, "tfidf_vectorizer.pkl")

# Evaluation
val_preds = xgb_model.predict(X_val)
val_acc = (val_preds == y_val).mean()
print(f"✅ Validation Accuracy: {val_acc:.4f}")

