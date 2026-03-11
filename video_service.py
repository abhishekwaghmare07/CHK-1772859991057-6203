import numpy as np
import cv2

FRAME_COUNT = 5
IMG_SIZE = (224, 224)
REAL_THRESHOLD = 0.5


def _extract_evenly_spaced_frames(video_path, frame_count=FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames > 0:
        frame_indices = np.linspace(0, max(total_frames - 1, 0), frame_count, dtype=int)
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = cap.read()
            if success and frame is not None:
                frames.append(frame)
    else:
        # Fallback for codecs that do not expose frame count.
        while len(frames) < frame_count:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    # Ensure fixed frame_count by repeating the last valid frame when needed.
    while len(frames) < frame_count:
        frames.append(frames[-1].copy())

    return frames[:frame_count]


def _get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def _center_crop(frame_bgr):
    height, width = frame_bgr.shape[:2]
    if height == 0 or width == 0:
        return frame_bgr

    crop_size = min(height, width)
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    cropped = frame_bgr[start_y : start_y + crop_size, start_x : start_x + crop_size]
    if cropped.size == 0:
        return frame_bgr
    return cropped


def _crop_face_or_fallback(frame_bgr, face_cascade):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return _center_crop(frame_bgr)

    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    face = frame_bgr[y : y + h, x : x + w]
    if face.size == 0:
        return _center_crop(frame_bgr)
    return face


def _preprocess_frame(frame_bgr, face_cascade, target_size=IMG_SIZE):
    focused = _crop_face_or_fallback(frame_bgr, face_cascade)
    frame_rgb = cv2.cvtColor(focused, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, target_size)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    return np.expand_dims(frame_rgb, axis=0)


def predict_video_file(video_path, model):
    frames = _extract_evenly_spaced_frames(video_path, frame_count=FRAME_COUNT)
    face_cascade = _get_face_cascade()

    probabilities = []
    frame_predictions = []

    for frame in frames:
        frame_batch = _preprocess_frame(frame, face_cascade=face_cascade, target_size=IMG_SIZE)
        probability = float(model.predict(frame_batch, verbose=0)[0][0])
        probabilities.append(probability)

        # The image model outputs Real probability.
        label = "Real" if probability >= REAL_THRESHOLD else "Fake"
        frame_predictions.append(label)

    real_votes = sum(1 for pred in frame_predictions if pred == "Real")
    fake_votes = len(frame_predictions) - real_votes
    frames_analyzed = len(frame_predictions)
    mean_prob = float(np.mean(probabilities))

    final_prediction = "Fake" if mean_prob < REAL_THRESHOLD else "Real"
    confidence = 1.0 - mean_prob if final_prediction == "Fake" else mean_prob

    if confidence > 0.75:
        confidence_level = "High"
    elif confidence >= 0.6:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    print("Video prediction debug")
    print("Frames extracted:", frames_analyzed)
    print("Sequences evaluated:", frames_analyzed)
    for idx, prob in enumerate(probabilities, start=1):
        print(f"Frame {idx} prediction probability:", prob)
    print("Real votes:", real_votes)
    print("Fake votes:", fake_votes)
    print("Mean probability:", mean_prob)
    print("Prediction probability:", confidence)
    print("Threshold used:", REAL_THRESHOLD)
    print("Confidence level:", confidence_level)

    return {
        "type": "video",
        "prediction": final_prediction,
        "confidence": float(confidence),
        "details": {
            "frames_analyzed": int(frames_analyzed),
            "sequences_evaluated": int(frames_analyzed),
            "real_votes": int(real_votes),
            "fake_votes": int(fake_votes),
            "mean_probability": float(mean_prob),
            "confidence_level": confidence_level,
        },
    }
