# app.py

import streamlit as st
import joblib
import xgboost as xgb  # ✅ add this
from sklearn.feature_extraction.text import TfidfVectorizer


# Load models
@st.cache_resource
def load_models():
    try:
        intent_model = joblib.load("intent_model.pkl")
        intent_vectorizer = joblib.load("intent_vectorizer.pkl")
        xgb_model = joblib.load("xgboost_model.pkl")
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return intent_model, intent_vectorizer, xgb_model, tfidf_vectorizer
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

intent_model, intent_vectorizer, xgb_model, tfidf_vectorizer = load_models()

# Streamlit UI
st.set_page_config(page_title="Hybrid Chatbot", layout="centered")
st.title("🤖 Hybrid Chatbot Predictor")
st.write("Enter a prompt and two responses to predict which one is preferred.")

prompt = st.text_area("📝 Prompt", "What is AI?")
response_a = st.text_area("💬 Response A", "AI is artificial intelligence.")
response_b = st.text_area("💬 Response B", "AI stands for artificial intelligence, a field of computing.")

def predict_winner(prompt, response_a, response_b):
    if not all([intent_model, intent_vectorizer, xgb_model, tfidf_vectorizer]):
        return "❌ Model not loaded."

    # Predict intent from prompt
    prompt_vec = intent_vectorizer.transform([prompt])
    intent_pred = intent_model.predict(prompt_vec)[0]

    # Predict best response
    combined_text = f"{prompt} {response_a} {response_b}"
    tfidf_vec = tfidf_vectorizer.transform([combined_text])
    xgb_pred = round(xgb_model.predict(tfidf_vec)[0])

    # Combine predictions
    final = xgb_pred if intent_pred != xgb_pred else intent_pred
    return "Response A" if final == 1 else "Response B"

# Prediction trigger
if st.button("🔍 Predict Winner"):
    if prompt and response_a and response_b:
        winner = predict_winner(prompt, response_a, response_b)
        st.success(f"🏆 Predicted Winner: **{winner}**")
    else:
        st.warning("⚠️ Please fill in all fields!")

st.sidebar.markdown("## ℹ️ About")
st.sidebar.info("This chatbot uses a hybrid ML model (Logistic Regression + XGBoost) trained on multilingual dialogue.")

