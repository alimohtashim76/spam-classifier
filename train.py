import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# Load dataset
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


# Train-test split
X = data['cleaned_message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train models

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)



# Predictions
lr_pred = lr_model.predict(X_test_tfidf)
nb_pred = nb_model.predict(X_test_tfidf)

print(" Logistic Regression Results")
print(classification_report(y_test, lr_pred))
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))

print("\n Naive Bayes Results")
print(classification_report(y_test, nb_pred))
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))


# Save best (Logistic Regression) model and vectorizer
joblib.dump(lr_model, 'model/spam_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("\n Logistic Regression model selected and saved.")
print("\n Model and vectorizer saved in 'model/' folder.")
