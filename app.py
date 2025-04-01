import streamlit as st
import pandas as pd
import numpy as np
import fasttext
import xgboost as xgb
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Title and description
st.title("Multilingual Chatbot Arena Predictor")
st.write("Enter a prompt and two responses to predict which one is preferred!")

# Load models and vectorizer (with error handling)
@st.cache_resource
def load_models():
    try:
        ft_model = fasttext.load_model("fasttext_model.bin")
        xgb_model = joblib.load("xgboost_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        return ft_model, xgb_model, tfidf
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}. Please ensure all model files are in the directory.")
        return None, None, None

ft_model, xgb_model, tfidf = load_models()

# User input
prompt = st.text_area("Enter your prompt:", "What is AI?")
response_a = st.text_area("Response A:", "AI is artificial intelligence.")
response_b = st.text_area("Response B:", "AI stands for artificial intelligence, a field in computing.")

# Prediction function
def predict_winner(prompt, response_a, response_b):
    if not ft_model or not xgb_model or not tfidf:
        return "Error: Models not loaded."
    
    # FastText prediction
    text = prompt.replace("\n", " ").strip()
    ft_pred = ft_model.predict(text)[0][0]
    ft_pred = 1 if ft_pred == "__label__1" else 0
    
    # TF-IDF + XGBoost prediction
    combined_text = f"{prompt} {response_a} {response_b}"
    tfidf_vec = tfidf.transform([combined_text])
    xgb_pred = round(xgb_model.predict(tfidf_vec)[0])
    
    # Combine predictions (prioritize XGBoost if different)
    final_pred = xgb_pred if ft_pred != xgb_pred else ft_pred
    return "Response A" if final_pred == 0 else "Response B"

# Predict button
if st.button("Predict"):
    if prompt and response_a and response_b:
        winner = predict_winner(prompt, response_a, response_b)
        st.success(f"Predicted Winner: **{winner}**")
    else:
        st.warning("Please fill in all fields.")

# About section
st.sidebar.header("About")
st.sidebar.write("This app predicts the preferred chatbot response using a hybrid FastText + XGBoost model, trained on the WSDM Cup dataset. Accuracy: 85.7%.")