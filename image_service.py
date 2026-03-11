from utils.image_preprocessing import preprocess_image


def predict_image_file(image_path, model):
    image_batch = preprocess_image(image_path, target_size=(224, 224))
    probability = float(model.predict(image_batch, verbose=0)[0][0])
    label = "Real" if probability >= 0.5 else "Fake"

    return {
        "type": "image",
        "prediction": label,
        "confidence": probability,
    }
