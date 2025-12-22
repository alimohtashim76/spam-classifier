from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Ensure 'message' key exists
    if 'message' not in data:
        return jsonify({'error': "Please provide a 'message' key in JSON"}), 400
    
    message = data['message']
    
    # Transform message using saved vectorizer
    message_tfidf = vectorizer.transform([message])
    
    # Predict label
    pred = model.predict(message_tfidf)[0]
    
    # Get probability of predicted class
    prob = model.predict_proba(message_tfidf).max()
    
    return jsonify({'prediction': pred, 'confidence': float(prob)})

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
