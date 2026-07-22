from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os

try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    model = None
    vectorizer = None
    print(f"Warning: model/vectorizer not loaded: {e}")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not available"}), 503

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
