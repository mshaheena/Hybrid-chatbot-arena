# train_intent_model.py

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# 📦 Step 1: Sample training data (you can replace this with real prompts)
# Format: "prompt" → intent label (0 or 1 or any custom intent class)
prompts = [
    "Hello", "Hi there", "Hey",
    "Goodbye", "Bye", "See you later",
    "Tell me a joke", "What is AI?", "Explain photosynthesis",
    "Thanks", "Thank you", "Appreciate it"
]

# 🎯 Step 2: Labels for those prompts (example: 0 = greeting, 1 = exit, 2 = question)
labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

# 🧠 Step 3: TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(prompts)

# 🚀 Step 4: Train the intent classifier
model = LogisticRegression()
model.fit(X, labels)

# 💾 Step 5: Save both model and vectorizer
joblib.dump(model, "intent_model.pkl")
joblib.dump(vectorizer, "intent_vectorizer.pkl")

print("✅ intent_model.pkl and intent_vectorizer.pkl saved successfully!")

