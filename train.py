# train.py

import pandas as pd
import fasttext
import xgboost as xgb
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Load Dataset
train_df = pd.read_parquet("train.parquet")  # Ensure you have this file from WSDM Cup

# Step 2: Encode Labels
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["winner"])

# Step 3: Prepare FastText Training File
with open("fasttext_train.txt", "w") as f:
    for _, row in train_df.iterrows():
        label = f"__label__{row['label']}"
        prompt = row["prompt"].replace("\n", " ").strip()
        f.write(f"{label} {prompt}\n")

# Step 4: Train FastText Classifier
ft_model = fasttext.train_supervised("fasttext_train.txt", epoch=100, lr=0.5, wordNgrams=3)
ft_model.save_model("fasttext_model.bin")

# Step 5: Prepare TF-IDF + XGBoost Inputs
X = train_df["prompt"] + " " + train_df["response_a"] + " " + train_df["response_b"]
y = train_df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# Step 6: Train XGBoost Classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_tfidf, y_train)

# Step 7: Save All Models
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")

# Step 8: Evaluate
val_preds = xgb_model.predict(X_val_tfidf)
val_accuracy = (val_preds == y_val).mean()
print(f"✅ XGBoost Validation Accuracy: {val_accuracy:.4f}")
