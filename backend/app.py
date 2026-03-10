import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detector_model.h5")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
CORS(app)

model = None
model_error = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_storage, input_shape):
    # Use model input resolution when available, fallback to 224x224.
    height = input_shape[1] if len(input_shape) > 2 and input_shape[1] else 224
    width = input_shape[2] if len(input_shape) > 2 and input_shape[2] else 224

    image = Image.open(file_storage.stream).convert("RGB")
    image = image.resize((width, height))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def initialize_model():
    global model, model_error
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        model_error = None
    except Exception as exc:
        model_error = str(exc)


@app.route("/health", methods=["GET"])
def health():
    if model_error:
        return jsonify({"status": "error", "message": model_error}), 500
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model_error:
        return jsonify({"error": model_error}), 500

    if "file" not in request.files:
        return jsonify({"error": "Missing file in request"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        x = preprocess_image(file, model.input_shape)
        score = float(model.predict(x, verbose=0)[0][0])

        prediction = "Fake" if score >= 0.5 else "Real"
        confidence = score if score >= 0.5 else 1.0 - score

        return jsonify(
            {
                "prediction": prediction,
                "confidence": confidence,
                "raw_score": score,
            }
        ), 200
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    initialize_model()
    app.run(host="127.0.0.1", port=5000, debug=True)
