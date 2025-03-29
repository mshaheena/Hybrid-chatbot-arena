import streamlit as st
import joblib
import os

# ✅ Function to safely load the model
@st.cache_resource
def load_model():
    model_path = "hybrid_model.pkl"  # Make sure this file is correctly uploaded
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"⚠️ Error loading model: {e}")
            return None
    else:
        st.error("⚠️ Model file not found! Please upload 'hybrid_model.pkl' to GitHub.")
        return None

# Load model
model = load_model()

# Streamlit UI
st.title("Hybrid Model Prediction Chatbot")
st.write("Ask me anything, and I'll predict the response based on my training!")

# User input
user_input = st.text_input("Enter your message:")

# Function to predict response
def predict_response(user_input):
    if model:
        try:
            return model.predict([user_input])[0]
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
            return "Error in prediction."
    return "Model not loaded."

# Predict button
if st.button("Predict"):
    if user_input:
        response = predict_response(user_input)
        st.success(f"Predicted Response: {response}")
    else:
        st.warning("Please enter a message.")
