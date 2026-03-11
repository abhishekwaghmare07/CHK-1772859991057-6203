import os
import random
import shutil

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LSTM, MaxPooling2D, TimeDistributed
from tensorflow.keras.models import Sequential


fake_videos_path = r"C:\Users\sagar\OneDrive\Desktop\deep fake detection\video for traning\videos_fake"
real_videos_path = r"C:\Users\sagar\OneDrive\Desktop\deep fake detection\video for traning\videos_real"

SEQUENCE_LENGTH = 20
IMG_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 25
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

MODEL_SAVE_PATH = "video_deepfake_model.h5"


def list_video_files(folder_path):
    valid_ext = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")
    return [
        os.path.join(folder_path, file_name)
        for file_name in sorted(os.listdir(folder_path))
        if os.path.isfile(os.path.join(folder_path, file_name))
        and file_name.lower().endswith(valid_ext)
    ]


def _get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def _center_crop(frame_bgr):
    h, w = frame_bgr.shape[:2]
    size = min(h, w)
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    cropped = frame_bgr[start_y : start_y + size, start_x : start_x + size]
    if cropped.size == 0:
        return frame_bgr
    return cropped


def _crop_largest_face(frame_bgr, face_cascade):
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


def extract_frames_from_video(video_path, sequence_length=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """Sample frames evenly, detect/crop face, resize, and normalize."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0

    face_cascade = _get_face_cascade()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
    else:
        frame_indices = np.arange(sequence_length)

    frames = []
    sampled_count = 0

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()
        if not success:
            break

        sampled_count += 1
        face = _crop_largest_face(frame, face_cascade)
        face = cv2.resize(face, img_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        frames.append(face)

    cap.release()

    if not frames:
        return None, sampled_count

    while len(frames) < sequence_length:
        frames.append(frames[-1].copy())

    return np.array(frames[:sequence_length], dtype=np.float32), sampled_count


def build_balanced_video_list(fake_paths, real_paths, seed=RANDOM_SEED):
    random.seed(seed)
    fake_count = len(fake_paths)
    real_count = len(real_paths)

    print(f"Number of fake videos: {fake_count}")
    print(f"Number of real videos: {real_count}")

    if fake_count == 0 or real_count == 0:
        raise ValueError("Both fake and real folders must contain at least one video.")

    if fake_count != real_count:
        target_count = min(fake_count, real_count)
        print(f"Dataset unbalanced. Randomly sampling {target_count} videos per class.")
        fake_paths = random.sample(fake_paths, target_count)
        real_paths = random.sample(real_paths, target_count)
    else:
        print("Dataset already balanced. Using all videos.")

    balanced_samples = [(path, 1.0, "fake") for path in fake_paths] + [
        (path, 0.0, "real") for path in real_paths
    ]
    random.shuffle(balanced_samples)
    print(f"Balanced dataset size: {len(balanced_samples)}")
    return balanced_samples


def load_dataset(samples):
    X_data = []
    y_data = []

    print("Starting frame extraction pipeline...")
    for i, (video_path, label, class_name) in enumerate(samples, start=1):
        print(f"Processing video {i}/{len(samples)} [{class_name}]: {os.path.basename(video_path)}")
        frames, sampled_count = extract_frames_from_video(video_path)
        if frames is None:
            print(f"Skipping unreadable or empty video: {video_path}")
            continue

        print(
            f"Frames extracted per video: {SEQUENCE_LENGTH} "
            f"(sampled {sampled_count} frames before padding if needed)"
        )
        X_data.append(frames)
        y_data.append(label)

    if not X_data:
        raise ValueError("No valid videos were loaded after frame extraction.")

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.float32)
    print(f"Final dataset tensor shape: {X.shape}")
    return X, y


def build_model():
    model = Sequential(
        [
            TimeDistributed(
                Conv2D(32, (3, 3), activation="relu"),
                input_shape=(SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3),
            ),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Conv2D(64, (3, 3), activation="relu")),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Conv2D(128, (3, 3), activation="relu")),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Flatten()),
            LSTM(64),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/{EPOCHS} started")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            "Epoch {}/{} finished - loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch + 1,
                EPOCHS,
                float(logs.get("loss", 0.0)),
                float(logs.get("accuracy", 0.0)),
                float(logs.get("val_loss", 0.0)),
                float(logs.get("val_accuracy", 0.0)),
            )
        )


def resolve_backend_model_path():
    candidates = [
        os.path.join("deepfake_detection", "backend", "models", "video_deepfake_model.h5"),
        os.path.join("backend", "models", "video_deepfake_model.h5"),
    ]
    for candidate in candidates:
        parent = os.path.dirname(candidate)
        if os.path.isdir(parent):
            return candidate
    return candidates[0]


def main():
    if not os.path.isdir(fake_videos_path):
        raise FileNotFoundError(f"Fake videos folder not found: {fake_videos_path}")
    if not os.path.isdir(real_videos_path):
        raise FileNotFoundError(f"Real videos folder not found: {real_videos_path}")

    print(f"Fake dataset path: {fake_videos_path}")
    print(f"Real dataset path: {real_videos_path}")
    print(f"Training config -> SEQUENCE_LENGTH={SEQUENCE_LENGTH}, IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")

    fake_videos = list_video_files(fake_videos_path)
    real_videos = list_video_files(real_videos_path)
    samples = build_balanced_video_list(fake_videos, real_videos)

    X_train, y_train = load_dataset(samples)

    print("Building CNN + LSTM model...")
    model = build_model()
    model.summary()

    print("Starting training with validation split...")
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        verbose=1,
        callbacks=[EpochLogger()],
    )

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    backend_model_path = resolve_backend_model_path()
    os.makedirs(os.path.dirname(backend_model_path), exist_ok=True)
    shutil.copy2(MODEL_SAVE_PATH, backend_model_path)
    print(f"Model copied to: {backend_model_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
