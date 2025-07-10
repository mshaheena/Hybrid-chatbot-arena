# train_intent_vectorizer.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

prompts = [
    "Hello", "Hi", "Good morning",
    "Bye", "Good night", "See you later",
    "Tell me a joke", "What is AI?", "Explain biology"
]

vectorizer = TfidfVectorizer()
vectorizer.fit(prompts)

joblib.dump(vectorizer, "intent_vectorizer.pkl")
print("✅ Saved: intent_vectorizer.pkl")

