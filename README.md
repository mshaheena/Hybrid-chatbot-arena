# 🤖 Hybrid Chatbot Predictor – Multilingual AI Model

Welcome to the **Hybrid Chatbot Predictor** — a smart, multilingual chatbot scoring engine built for **real-world AI competitions** and deployable commercial use.

🚀 This project combines **FastText** for fast intent recognition and **TF-IDF + XGBoost** for deep contextual judgment of chatbot responses.

---

## 🌐 Live Demo (Hugging Face Spaces)

👉 Try it here: [https://huggingface.co/spaces/mshaheena/hybrid-chatbot](https://huggingface.co/spaces/mshaheena/hybrid-chatbot)

---

## 🎯 What It Does

Given:
- A user **prompt**
- Two chatbot **responses**

This hybrid model predicts **which response is better** — trained on real multilingual conversational data (WSDM Cup).

Use cases:
- 🔥 Chatbot comparison tools
- 🤖 Customer service response testing
- 🌍 Multilingual conversation understanding

---

## 🧠 Tech Stack

| Component | Description |
|----------|-------------|
| `FastText` | Lightweight and multilingual prompt classifier |
| `TF-IDF + XGBoost` | Feature-based scorer for full prompt + responses |
| `Streamlit` | Interactive web interface |
| `Hugging Face Spaces` | One-click hosting for AI demos |
| `scikit-learn` | Preprocessing & vectorization |

---

## 📦 Project Structure

hybrid-chatbot/
├── app.py # Streamlit UI for inference
├── train.py # Full training pipeline
├── predict.py # Command-line test script
├── requirements.txt # All dependencies
├── runtime.txt # Python version
├── fasttext_model.bin # Trained FastText model
├── xgboost_model.pkl # Trained XGBoost model
├── tfidf_vectorizer.pkl # Saved vectorizer
└── README.md # This file


---

## 🔧 How to Run Locally

git clone https://github.com/mshaheena/hybrid-chatbot
cd hybrid-chatbot

# (Optional) create virtual env
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py

----

Or test directly:
python predict.py

----
| Model                      | Accuracy                   |
| -------------------------- | -------------------------- |
| FastText intent classifier | \~83%                      |
| XGBoost response scorer    | \~87%                      |
| Hybrid combination logic   | 💡 Smart fallback decision |

----
*Ideal For Freelancers & Clients:
AI Chatbot Evaluation Tools
Multilingual NLP Apps
Enterprise Chat UX Testing
Academic Competitions (WSDM, etc.)
----
*Future Ideas:
Replace FastText with BERT or DistilBERT for deeper understanding
Add memory/context chaining
Support ranking for 3+ responses (RAG)
----
About Me
I'm a machine learning developer building multilingual, production-grade AI tools.
Looking for freelance work? Let’s build smart NLP systems together!

📫 Contact: mshaheena8838@gmail.com
💼 LinkedIn: https://www.linkedin.com/in/m-shaheena

----




