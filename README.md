---
title: Hybrid Chatbot Evaluator
emoji: 🤖
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 4.24.0
app_file: app.py
tags:
  - chatbot
  - ai
  - machine-learning
  - gradio
  - huggingface
  - nlp
  - xgboost
  - logistic-regression
---

# 🤖 Hybrid Chatbot Evaluator

### 📍 **Live App:**  
👉 [Click here to try it on Hugging Face Spaces](https://huggingface.co/spaces/mshaheena/hybrid-chatbot)

---

## 💡 What is This?

This is a **hybrid machine learning chatbot evaluator** that helps you determine which chatbot response is better given a prompt.

It uses **intelligent intent detection** + **response scoring** to predict which response (A or B) is preferred.

---

## 🔍 Example Use Case

> You have two different chatbot models and want to **automatically choose** the better response for a given prompt — **this app does that**.

✅ Great for:
- Evaluating AI assistants  
- Testing multilingual chatbot models  
- Ranking outputs from LLMs  
- Real-time feedback for NLP applications

---

## 🧠 How It Works

| Component           | Description |
|--------------------|-------------|
| 💬 Prompt Input     | User gives a conversational question or command |
| 🅰️ / 🅱️ Response A/B | Two generated chatbot replies |
| 🔍 Model Output     | App picks the most appropriate response |

The prediction is powered by two models:

### 1️⃣ **Intent Classifier**
- Model: `Logistic Regression`
- Features: `TF-IDF` on prompt only
- Purpose: Understand the user's **intent or tone**

### 2️⃣ **Response Ranker**
- Model: `XGBoost`
- Features: Combined `TF-IDF` on prompt + responses
- Purpose: Predict which response is **more suitable**

The final winner is chosen by **hybrid decision logic** 🧠⚖️

---

## 🚀 Technologies Used

| Tech / Tool | Description |
|-------------|-------------|
| 🐍 Python 3.10 | Backend |
| 🤖 scikit-learn | ML: Logistic Regression |
| ⚡ XGBoost | ML: Boosted trees |
| 🔤 TfidfVectorizer | Text embedding |
| 🎨 Gradio | User interface |
| 💾 Joblib | Model saving/loading |
| ☁️ Hugging Face Spaces | Hosting & deployment |

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio interface |
| `train.py` | Training both models |
| `requirements.txt` | Dependencies |
| `runtime.txt` | Python version lock |
| `intent_model.pkl` | Trained intent classifier |
| `intent_vectorizer.pkl` | TF-IDF for prompt |
| `xgboost_model.pkl` | Trained response scorer |
| `tfidf_vectorizer.pkl` | TF-IDF for full inputs |

---

## 💼 Freelancer Portfolio Ready

This project is ideal to show off:
- Hybrid ML pipelines
- Real-world NLP evaluation tools
- Deployment with Hugging Face Spaces
- Fast, interactive UI with Gradio

---

## 🌐 Try It Live!

🔗 [https://huggingface.co/spaces/mshaheena/hybrid-chatbot](https://huggingface.co/spaces/mshaheena/hybrid-chatbot)

---

## 🙋‍♀️ Built With ❤️ By [mshaheena](https://huggingface.co/mshaheena)

*About Me:

I'm a machine learning developer building multilingual, production-grade AI tools.
Looking for freelance work? Let’s build smart NLP systems together!

📫 Contact: mshaheena8838@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/m-shaheena

----




