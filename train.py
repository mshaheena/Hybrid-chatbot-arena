import pandas as pd
import fasttext
import xgboost as xgb
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data (replace with your local path)
train_df = pd.read_parquet("train.parquet")  # Download from Kaggle and place locally

# Encode labels
le = LabelEncoder()
train_df["label"] = le.fit_transform(train_df["winner"])

# Prepare FastText training file
with open("fasttext_train.txt", "w") as f:
    for _, row in train_df.iterrows():
        label = f"__label__{row['label']}"
        prompt = row["prompt"].replace("\n", " ").strip()
        f.write(f"{label} {prompt}\n")

# Train FastText
ft_model = fasttext.train_supervised("fasttext_train.txt", epoch=100, lr=0.5, wordNgrams=3)
ft_model.save_model("fasttext_model.bin")

# Train TF-IDF + XGBoost
X = train_df["prompt"] + " " + train_df["response_a"] + " " + train_df["response_b"]
y = train_df["label"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_valid_tfidf = tfidf.transform(X_valid)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_tfidf, y_train)

# Save models
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")

# Evaluate
preds = xgb_model.predict(X_valid_tfidf)
accuracy = (preds == y_valid).mean()
print(f"Validation Accuracy: {accuracy:.4f}")