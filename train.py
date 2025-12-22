import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------
# 1️⃣ Load dataset
# Adjust the column names if needed
data = pd.read_csv('data/spam.csv', encoding='latin-1', usecols=['v1', 'v2'])
data = data.rename(columns={'v1': 'label', 'v2': 'message'})

# Optional: clean text
import string
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
data['cleaned_message'] = data['message'].apply(clean_text)

# -----------------------------
# 2️⃣ Train-test split
X = data['cleaned_message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# 4️⃣ Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# 5️⃣ Evaluate
y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 6️⃣ Save model and vectorizer
joblib.dump(model, 'model/spam_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("\n✅ Model and vectorizer saved in 'model/' folder.")
