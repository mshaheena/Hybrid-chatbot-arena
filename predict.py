# predict.py

import fasttext
import joblib
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
try:
    ft_model = fasttext.load_model("fasttext_model.bin")
    xgb_model = joblib.load("xgboost_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# Prediction function
def predict_response(prompt, response_a, response_b):
    # FastText prediction
    text = prompt.replace("\n", " ").strip()
    ft_label = ft_model.predict(text)[0][0]  # '__label__0' or '__label__1'
    fasttext_pred = 1 if ft_label == "__label__1" else 0

    # TF-IDF + XGBoost prediction
    combined_text = f"{prompt} {response_a} {response_b}"
    tfidf_vector = tfidf.transform([combined_text])
    xgb_pred = round(xgb_model.predict(tfidf_vector)[0])

    # Combine logic (you can customize this)
    final = xgb_pred if xgb_pred != fasttext_pred else fasttext_pred
    return "Response A" if final == 0 else "Response B"

# Demo mode (can replace with CLI input)
if __name__ == "__main__":
    prompt = input("Enter the prompt: ")
    response_a = input("Enter response A: ")
    response_b = input("Enter response B: ")

    winner = predict_response(prompt, response_a, response_b)
    print(f"🏆 Predicted Better Response: {winner}")
