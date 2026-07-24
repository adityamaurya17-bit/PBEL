from flask import Flask, request, jsonify
from flask_cors import CORS
import os

import verify as verify_module

app = Flask(__name__)
CORS(app)

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Backend is working!"})


@app.route("/verify", methods=["POST"])
def verify_claim():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = verify_module.verify(text)
    except Exception as e:
        return jsonify({"error": f"Verification failed: {e}"}), 502
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
