# Spam Classifier
**Python · Flask · Pandas · scikit-learn · TF-IDF · Joblib**

An end-to-end **AI-powered Spam Classification system** that detects whether a text message is **Spam** or **Ham (Not Spam)**.  
The project covers the complete machine learning lifecycle — **data preprocessing, model training, evaluation, and API deployment using Flask**.

---

## Features

- REST API to classify messages as **Spam or Ham**
- TF-IDF based text vectorization
- Comparison of **Logistic Regression** and **Naive Bayes**
- Model evaluation using precision, recall, F1-score & confusion matrix
- Trained models saved and reused with Joblib
- Tested using **Postman**
- Clean and modular project structure

---

## Dataset

The model is trained on the **SMS Spam Collection Dataset**.  

- Total Messages: **5,572**
- Features:
  - `label` → spam / ham
  - `message` → SMS text
- Class Distribution:
  - Ham: **4,825**
  - Spam: **747**

**Source**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)

---

## Model

## 1. Logistic Regression (Selected for Deployment)
- Accuracy: **96.68%**
- Spam Recall: **75%**
- Best overall performance

### 2. Multinomial Naive Bayes
- Accuracy: **95.42%**
- Spam Recall: **66%**

Logistic Regression performed better and is used in the Flask API.

---

## Results & Evaluation

### Logistic Regression Confusion Matrix
- True Ham: **965**
- False Spam: **0**
- False Ham: **37**
- True Spam: **113**

### Naive Bayes Confusion Matrix
- True Ham: **965**
- False Spam: **0**
- False Ham: **51**
- True Spam: **99**

Evaluation Metrics Used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Technologies Used

- **Python** – Core programming language
- **Pandas** – Data handling and preprocessing
- **scikit-learn** – ML models & evaluation
- **TF-IDF Vectorizer** – Feature extraction
- **Flask** – REST API development
- **Joblib** – Model persistence
- **Postman** – API testing

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/alimohtashim76/spam-classifier.git
cd spam-classifier
```

### 2. Create a virtual environment

`python -m venv venv`

### 3. Activate the virtual environment:

`venv\Scripts\activate`

## Usage

### Train the model

`python train.py`
This will:
  -Train Logistic Regression & Naive Bayes models
  -Print evaluation metrics
  -Save trained model and vectorizer to the model/ folder

### Run the Flask API

`python app.py`

- The server will start at http://127.0.0.1:5000/

- Use **Postman** or **Python requests** to send a **POST** request to `/predict` with JSON body:
```json
{
 "message": "Congratulations! You have won a free ticket."
}
```

- Example response:
```json
{
 "prediction": "spam",
 "confidence": 0.98
}
```



