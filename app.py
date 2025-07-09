# app.py

import streamlit as st
import pandas as pd
import numpy as np
import fasttext
import xgboost as xgb
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

st.set_page_config(page_title="Hybrid Chatbot", layout="centered")
st.title("🤖 Hybrid Chatbot: WSDM Cup Evaluator")
st.write("Enter a prompt and two responses to predict which one is preferred!")

@st.cache_resource
def load_models():
    try:
        ft_model = fasttext.load_model("fasttext_model.bin")
        xgb_model = joblib.load("xgboost_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        return ft_model, xgb_model, tfidf
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

ft_model, xgb_model, tfidf = load_models()

# --- Input Fields ---
prompt = st.text_area("📝 Prompt", "What is AI?")
response_a = st.text_area("💬 Response A", "AI is artificial intelligence.")
response_b = st.text_area("💬 Response B", "AI stands for artificial intelligence, a field in computing.")

# --- Prediction Logic ---
def predict_winner(prompt, response_a, response_b):
    if not ft_model or not xgb_model or not tfidf:
        return "❌ Models not loaded."

    # FastText: predict label from prompt
    text = prompt.replace("\n", " ").strip()
    label = ft_model.predict(text)[0][0]
    fasttext_pred = 1 if label == "__label__1" else 0

    # TF-IDF + XGBoost: predict from full input
    combined_text = f"{prompt} {response_a} {response_b}"
    tfidf_vec = tfidf.transform([combined_text])
    xgb_pred = round(xgb_model.predict(tfidf_vec)[0])

    # Combine both predictions (you can customize)
    final = xgb_pred if fasttext_pred != xgb_pred else fasttext_pred
    return "Response A" if final == 0 else "Response B"

# --- Predict Button ---
if st.button("🔍 Predict Preferred Response"):
    if prompt and response_a and response_b:
        winner = predict_winner(prompt, response_a, response_b)
        st.success(f"🏆 Predicted Winner: **{winner}**")
    else:
        st.warning("⚠️ Please fill in all fields!")

st.sidebar.markdown("## 📘 About")
st.sidebar.info("This hybrid chatbot uses FastText + XGBoost trained on WSDM Cup multilingual chat data.")

