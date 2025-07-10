---
title: Hybrid Chatbot Evaluator
emoji: 🤖
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: "4.24.0"
app_file: app.py
pinned: false
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

This is a **hybrid machine learning chatbot evaluator** that intelligently determines which chatbot response is better for a given prompt.

It uses **intent classification** + **response scoring** to predict which response (A or B) is preferred.

---

## 🔍 Example Use Case

> You have two chatbot models and want to **automatically select the better response** — this app helps you do just that.

✅ Great for:
- Evaluating AI assistants  
- Testing multilingual chatbots  
- Ranking outputs from LLMs  
- Real-time feedback in NLP pipelines

---

## 🧠 How It Works

| Component           | Description |
|--------------------|-------------|
| 💬 Prompt Input     | User enters a conversational question or instruction |
| 🅰️ / 🅱️ Responses   | Two generated chatbot replies |
| 🔍 Model Output     | The app picks the better response using ML |

### Powered by Two Models:

#### 1️⃣ Intent Classifier
- 🔹 **Model:** Logistic Regression  
- 🔹 **Input:** Prompt only (TF-IDF)  
- 🔹 **Goal:** Understand the user's tone or intent

#### 2️⃣ Response Ranker
- 🔸 **Model:** XGBoost  
- 🔸 **Input:** Combined prompt + both responses (TF-IDF)  
- 🔸 **Goal:** Choose the more relevant/stronger response

The final decision is made using **hybrid logic** 🧠⚖️

---

## 🚀 Technologies Used

| Tech | Role |
|------|------|
| 🐍 Python 3.10 | Core programming |
| 🤖 scikit-learn | Intent classifier |
| ⚡ XGBoost | Response ranker |
| 🔤 TfidfVectorizer | Text feature extraction |
| 💾 Joblib | Model saving/loading |
| 🎨 Gradio | User interface |
| ☁️ Hugging Face Spaces | Deployment |

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio app |
| `train.py` | Training logic |
| `requirements.txt` | Dependencies |
| `runtime.txt` | Python version |
| `intent_model.pkl` | Intent classification model |
| `intent_vectorizer.pkl` | TF-IDF for intent |
| `xgboost_model.pkl` | Response scoring model |
| `tfidf_vectorizer.pkl` | TF-IDF for response |

---

## 💼 Freelancer Portfolio Ready

This project is ideal to showcase your:
- 🔁 Hybrid ML pipelines (LogReg + XGBoost)
- 🌍 Real-world NLP evaluation use case
- ☁️ Cloud-based deployment on Hugging Face
- ⚡ Fast Gradio-based UI for demo or client use

---

## 🌐 Try It Live!

🔗 [Launch on Hugging Face](https://huggingface.co/spaces/mshaheena/hybrid-chatbot)

---

## 🙋‍♀️ Built With ❤️ by [mshaheena](https://huggingface.co/mshaheena)

**About Me**  
I'm a machine learning engineer with a passion for deploying real-time, multilingual, production-grade NLP applications.

📬 **Contact:** mshaheena8838@gmail.com  
💼 **LinkedIn:** [linkedin.com/in/m-shaheena](https://www.linkedin.com/in/m-shaheena)

Let’s work together to build intelligent AI solutions! 🚀





