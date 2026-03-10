import os
import tempfile
import uuid
import logging

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detector_model.h5")
VIDEO_MODEL_PATH = os.path.join(BASE_DIR, "deepfake_video_detector_model.h5")
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"}

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

image_model = None
video_model = None
model_error = None


def allowed_file(filename: str, allowed_extensions) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def preprocess_image(file_storage, input_shape):
    # Use model input resolution when available, fallback to 224x224.
    height = input_shape[1] if len(input_shape) > 2 and input_shape[1] else 224
    width = input_shape[2] if len(input_shape) > 2 and input_shape[2] else 224

    image = Image.open(file_storage.stream).convert("RGB")
    image = image.resize((width, height))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def preprocess_video(file_storage, input_shape):
    # Expected shape from model: (batch, frames, height, width, channels)
    seq_len = input_shape[1] if len(input_shape) > 4 and input_shape[1] else 5
    height = input_shape[2] if len(input_shape) > 4 and input_shape[2] else 64
    width = input_shape[3] if len(input_shape) > 4 and input_shape[3] else 64

    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.{ext}")
    file_storage.save(temp_path)

    frames = []
    cap = cv2.VideoCapture(temp_path)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Unable to read video frames")

        sample_indices = np.linspace(0, max(total_frames - 1, 0), seq_len, dtype=int)

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            frames.append(frame)

        if not frames:
            raise ValueError("No valid frames extracted from video")

        while len(frames) < seq_len:
            frames.append(frames[-1])

        arr = np.array(frames[:seq_len], dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    finally:
        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)


def initialize_model():
    global image_model, video_model, model_error
    try:
        if not os.path.exists(IMAGE_MODEL_PATH):
            raise FileNotFoundError(f"Image model file not found: {IMAGE_MODEL_PATH}")

        image_model = load_model(IMAGE_MODEL_PATH)

        if os.path.exists(VIDEO_MODEL_PATH):
            video_model = load_model(VIDEO_MODEL_PATH)
        else:
            video_model = None

        model_error = None
        app.logger.info("Models loaded. image_model=%s video_model=%s", bool(image_model), bool(video_model))
    except Exception as exc:
        model_error = str(exc)
        app.logger.exception("Model initialization failed")


@app.before_request
def log_request_info():
    app.logger.info("Request: %s %s", request.method, request.path)


@app.route("/health", methods=["GET"])
def health():
    if model_error:
        return jsonify({"status": "error", "message": model_error}), 500
    return jsonify({"status": "ok", "image_model": bool(image_model), "video_model": bool(video_model)}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model_error:
        return jsonify({"error": model_error}), 500

    if "file" not in request.files:
        return jsonify({"error": "Missing file in request"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        x = preprocess_image(file, image_model.input_shape)
        score = float(image_model.predict(x, verbose=0)[0][0])

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
        app.logger.exception("Image prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/predict/video", methods=["POST"])
def predict_video():
    if model_error:
        return jsonify({"error": model_error}), 500

    if not video_model:
        return jsonify({"error": "Video model not available"}), 500

    if "file" not in request.files:
        return jsonify({"error": "Missing file in request"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        x = preprocess_video(file, video_model.input_shape)
        score = float(video_model.predict(x, verbose=0)[0][0])
        prediction = "Fake" if score >= 0.5 else "Real"
        confidence = score if score >= 0.5 else 1.0 - score

        return jsonify(
            {
                "prediction": prediction,
                "confidence": confidence,
                "raw_score": score,
                "type": "video",
            }
        ), 200
    except Exception as exc:
        app.logger.exception("Video prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    initialize_model()
    app.run(host="127.0.0.1", port=5000, debug=True)
