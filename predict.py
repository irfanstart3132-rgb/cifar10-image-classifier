"""
predict.py
----------
Loads the saved model and classifies a single new image file.

Colab usage:
    !python predict.py your_image.jpg

(Upload your_image.jpg via the Colab file sidebar first, into the
same root folder as this script.)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

from model import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH

# Flat path for Colab - matches where train.py saved the model
MODEL_PATH = os.path.join("models", "cifar10_cnn.keras")


def load_and_preprocess_image(image_path):
    """
    Prepares an arbitrary image file for the model:
      1. Open and convert to RGB (drops alpha channel / handles grayscale)
      2. Resize to 32x32 to match what the model was trained on
      3. Convert to a float32 array and normalize to [0, 1]
      4. Add a batch dimension: (32, 32, 3) -> (1, 32, 32, 3)
         because Keras models always expect a batch axis, even for
         a single image.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: no trained model found at {MODEL_PATH}.")
        print("Run 'python train.py' first.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Preprocessing {image_path}...")
    img_array = load_and_preprocess_image(image_path)

    # model.predict returns a probability distribution over the 10 classes,
    # shape (1, 10). We take index [0] to get the single image's result.
    probabilities = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx]

    print(f"\nPrediction: {predicted_class} (confidence: {confidence:.2%})\n")
    print("Full probability distribution:")
    for class_name, prob in sorted(
        zip(CLASS_NAMES, probabilities), key=lambda x: x[1], reverse=True
    ):
        print(f"  {class_name:12s}: {prob:.4f}")


if __name__ == "__main__":
    main()
