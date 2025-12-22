# Spam Classifier
**Python Flask Pandas scikit-learn Joblib**

A simple web application built with **Flask** to classify text messages as **Spam** or **Ham (not spam)**. This project uses a **Logistic Regression** model trained on the **SMS Spam Collection Dataset**.

---

## Features

- **Interactive Prediction**: Users can send a POST request with a message and get a real-time prediction of whether it is spam or ham.
- **Model Evaluation**: The model performance can be checked with a classification report and confusion matrix during training.
- **User-Friendly API**: Simple REST API endpoint `/predict` for integration with front-end or other applications.
- **Prototype Ready**: Ideal for learning AI model deployment with Flask.

---

## Dataset

The model is trained on the **SMS Spam Collection Dataset**.  

- **Number of Instances**: 5572 messages  
- **Attributes**: 2 (`label` and `message`)  
- **Class Distribution**: Spam vs Ham  
  - Spam: 747  
  - Ham: 4825  

**Source**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)

---

## Model

The prediction model is a **Logistic Regression** model from scikit-learn.

- **Model Training**: Trained on TF-IDF vectorized messages with an 80/20 train-test split.
- **Evaluation**: Model performance is evaluated using classification report and confusion matrix.
- **Model Serialization**: Trained model and TF-IDF vectorizer are saved as `spam_model.pkl` and `vectorizer.pkl` for use in the Flask API.

---

## Technologies Used

- **Python**: Core programming language.  
- **Flask**: For creating and running the web API.  
- **Pandas**: Data loading and preprocessing.  
- **scikit-learn**: Logistic Regression and TF-IDF vectorization.  
- **Joblib**: Saving and loading the trained model.  
- **Postman**: For testing API endpoints.

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/alimohtashim76/spam-classifier.git
cd spam-classifier

### 2. Create a virtual environment

'python -m venv venv

### 3. Activate the virtual environment:

'venv\Scripts\activate

## Usage

### Train the model

'python train.py 
- This will train the model, evaluate it, and save the .pkl files in the model/ folder.

### Run the Flask API

'python app.py

- The server will start at http://127.0.0.1:5000/

- Use Postman or Python requests to send a POST request to /predict with JSON body:
'''{
     "message": "Congratulations! You have won a free ticket."
   }

- Example response:
'''{
     "prediction": "spam",
     "confidence": 0.98
   }




