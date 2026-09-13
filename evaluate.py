"""
evaluate.py
-----------
Loads the saved model and evaluates it on the CIFAR-10 test set,
printing overall accuracy plus a per-class breakdown and a confusion
matrix.

Colab usage:
    !python evaluate.py

(Run this only after train.py has completed and saved a model.)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

from model import CLASS_NAMES

# Flat path for Colab - matches where train.py saved the model
MODEL_PATH = os.path.join("models", "cifar10_cnn.keras")


def main():
    print("Loading test data...")
    (_, _), (X_test, y_test) = cifar10.load_data()
    X_test = X_test.astype("float32") / 255.0
    y_test = y_test.flatten()

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Running evaluation...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nOverall test accuracy: {test_acc:.4f}")
    print(f"Overall test loss:     {test_loss:.4f}\n")

    # Per-class accuracy: how well the model does on each of the 10 classes
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    print("Per-class accuracy:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = y_test == class_idx
        class_acc = np.mean(predicted_labels[mask] == y_test[mask])
        print(f"  {class_name:12s}: {class_acc:.4f}")

    # Simple confusion matrix (rows = true label, cols = predicted label)
    num_classes = len(CLASS_NAMES)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_test, predicted_labels):
        confusion[true_label, pred_label] += 1

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "        " + " ".join(f"{c[:4]:>5s}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(confusion):
        row_str = " ".join(f"{v:5d}" for v in row)
        print(f"{CLASS_NAMES[i][:7]:8s}{row_str}")


if __name__ == "__main__":
    main()
