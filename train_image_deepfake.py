import os
import sys

# If any library import fails, install dependencies with:
# pip install tensorflow matplotlib opencv-python numpy
try:
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError as exc:
    raise ImportError(
        "Required libraries are missing. Install with: "
        "pip install tensorflow matplotlib opencv-python numpy"
    ) from exc


# Dataset root folder containing:
# img/
#   real/
#   fake/
DATASET_DIR = r"C:\Users\sagar\OneDrive\Desktop\deep fake detection\img"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 30
MODEL_SAVE_PATH = "image_deepfake_model.h5"
MODEL = None


def resolve_dataset_dir(base_dir):
    """Resolve the folder that directly contains the two class subfolders."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    # Preferred structure requested by user: base_dir/real and base_dir/fake
    preferred_real = os.path.join(base_dir, "real")
    preferred_fake = os.path.join(base_dir, "fake")
    if os.path.isdir(preferred_real) and os.path.isdir(preferred_fake):
        return base_dir

    # Known workspace variants.
    candidates = [
        os.path.join(base_dir, "real_and_fake_face"),
        os.path.join(base_dir, "real_and_fake_face_detection", "real_and_fake_face"),
    ]
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        has_real = os.path.isdir(os.path.join(candidate, "training_real"))
        has_fake = os.path.isdir(os.path.join(candidate, "training_fake"))
        if has_real and has_fake:
            return candidate

    # Generic fallback: first child folder that contains exactly two class dirs.
    for child in os.listdir(base_dir):
        child_path = os.path.join(base_dir, child)
        if not os.path.isdir(child_path):
            continue
        class_dirs = [
            name for name in os.listdir(child_path)
            if os.path.isdir(os.path.join(child_path, name))
        ]
        if len(class_dirs) == 2:
            return child_path

    raise FileNotFoundError(
        "Could not find a dataset folder with two class subfolders (real/fake)."
    )


def create_data_generators(dataset_dir, img_size, batch_size):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,      # Normalize pixel values to [0, 1]
        validation_split=0.3      # 70% train, 30% validation split
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_generator, val_generator


def print_environment_diagnostics():
    """Print interpreter and dependency status to help resolve VS Code warnings."""
    print("Python executable:", sys.executable)

    required_modules = {
        "tensorflow": "tensorflow",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "opencv-python": "cv2",
    }

    for package_name, module_name in required_modules.items():
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{package_name}: installed (version {version})")
        except Exception:
            print(f"{package_name}: NOT INSTALLED")

    # Small TensorFlow import/version test block requested by user.
    try:
        import tensorflow as tf_check
        print("TensorFlow version:", tf_check.__version__)
    except Exception as exc:
        print("TensorFlow import test failed:", exc)


def build_cnn_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_or_build_model(input_shape=(224, 224, 3)):
    """Build the CNN only if no model is currently defined in memory."""
    global MODEL
    if MODEL is None:
        MODEL = build_cnn_model(input_shape=input_shape)
    return MODEL


def plot_training_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_single_image(image_path, model, img_size=(224, 224), class_indices=None):
    image = load_img(image_path, target_size=img_size)
    image_array = img_to_array(image) / 255.0
    image_array = tf.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)[0][0]

    # Default mapping for flow_from_directory in alphabetical order: fake=0, real=1
    if class_indices is None:
        class_indices = {"fake": 0, "real": 1}

    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class_idx = 1 if prediction >= 0.5 else 0
    predicted_label = idx_to_class.get(predicted_class_idx, str(predicted_class_idx))

    print(f"Image: {image_path}")
    print(f"Predicted label: {predicted_label}")
    print(f"Confidence score (real probability): {prediction:.4f}")

    return predicted_label, float(prediction)


def main():
    dataset_dir = resolve_dataset_dir(DATASET_DIR)

    print_environment_diagnostics()

    # Sanity check to confirm OpenCV and NumPy are available at runtime.
    print(f"NumPy version: {np.__version__}; OpenCV version: {cv2.__version__}")
    print(f"Using dataset directory: {dataset_dir}")

    train_data, val_data = create_data_generators(dataset_dir, IMG_SIZE, BATCH_SIZE)

    print("Class indices:", train_data.class_indices)

    model = get_or_build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    print("Starting model training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    plot_training_history(history)

    # Example single-image prediction (uncomment and set your image path)
    # test_image_path = r"C:\path\to\your\test_image.jpg"
    # predict_single_image(test_image_path, model, img_size=IMG_SIZE, class_indices=train_data.class_indices)


if __name__ == "__main__":
    main()
