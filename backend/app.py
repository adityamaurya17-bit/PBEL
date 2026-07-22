from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load your trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)
CORS(app)  # allow React frontend to connect

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Convert text to vector
    input_vec = vectorizer.transform([text])
    prediction = model.predict(input_vec)[0]
    prob = model.predict_proba(input_vec)[0]
    confidence = max(prob) * 100

    result = "Fake News" if prediction == 0 else "Real News"
    return jsonify({"prediction": result, "confidence": round(confidence, 2)})

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Backend is working!"})

if __name__ == "__main__":
    app.run(debug=True)
