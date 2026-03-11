import cv2
import numpy as np


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

    # Select the largest detected face.
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    face = frame_bgr[y : y + h, x : x + w]
    if face.size == 0:
        return _center_crop(frame_bgr)
    return face


def extract_face_frames(video_path, frame_size=(128, 128), max_frames=120):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    face_cascade = _get_face_cascade()
    frames = []
    sampled_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        sampled_frames += 1

        face = _crop_face_or_fallback(frame, face_cascade)
        face = cv2.resize(face, frame_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        frames.append(face)

        if len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    return np.array(frames, dtype=np.float32), sampled_frames


def build_video_sequences(frames, sequence_length=30):
    if len(frames) == 0:
        raise ValueError("Cannot build sequences from empty frames")

    if len(frames) < sequence_length:
        padded = list(frames)
        while len(padded) < sequence_length:
            padded.append(padded[-1].copy())
        return np.expand_dims(np.array(padded, dtype=np.float32), axis=0)

    stride = max(1, sequence_length // 2)
    windows = []
    last_start = len(frames) - sequence_length
    for start in range(0, last_start + 1, stride):
        windows.append(frames[start : start + sequence_length])

    if last_start % stride != 0:
        windows.append(frames[last_start : last_start + sequence_length])

    return np.array(windows, dtype=np.float32)


def preprocess_video(video_path, sequence_length=30, frame_size=(128, 128), max_frames=120):
    frames, _ = extract_face_frames(video_path, frame_size=frame_size, max_frames=max_frames)
    sequences = build_video_sequences(frames, sequence_length=sequence_length)

    # Keep backward compatibility with code paths expecting a single sequence.
    return np.expand_dims(sequences[0], axis=0)
