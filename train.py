"""
train.py
--------
Loads CIFAR-10, preprocesses it, builds the CNN, trains it, and saves
the trained model to disk.

Colab usage:
    !python train.py

(Upload model.py, train.py, evaluate.py, predict.py all together into
the root of your Colab session - no subfolders needed. Turn on a GPU
via Runtime > Change runtime type for much faster training.)
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_model, CLASS_NAMES

# Flat paths for Colab - everything lives in the current working directory
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn.keras")
EPOCHS = 15
BATCH_SIZE = 64


def load_and_preprocess_data():
    """
    Loads CIFAR-10 and normalizes pixel values.

    X_train: (50000, 32, 32, 3) uint8  -> raw training images
    y_train: (50000, 1)         int    -> raw training labels (0-9)
    X_test:  (10000, 32, 32, 3) uint8  -> raw test images
    y_test:  (10000, 1)         int    -> raw test labels (0-9)

    Normalization divides pixel values by 255 so every input feature
    is in the range [0, 1] instead of [0, 255]. This keeps gradients
    well-scaled during training and helps the network converge faster.
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Keras' sparse_categorical_crossentropy expects integer labels
    # shaped (N,), not (N, 1), so we flatten them here.
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return (X_train, y_train), (X_test, y_test)


def get_data_augmentor():
    """
    Data augmentation artificially expands the training set by applying
    small random transformations (flips, shifts, rotations) to each
    image every epoch. The model never sees the exact same image twice,
    which reduces overfitting and improves generalization.
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )


def plot_history(history):
    """Saves accuracy/loss curves so you can visually inspect training."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train_accuracy")
    axes[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[0].set_title("Accuracy over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(MODEL_DIR, "training_history.png")
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading and preprocessing CIFAR-10...")
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()

    # Train/validation split: hold out 20% of the training set so we can
    # measure performance on data the model never trains on directly.
    # This is what catches overfitting during training.
    val_split = int(0.8 * len(X_train))
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = build_model()
    model.summary()

    datagen = get_data_augmentor()
    datagen.fit(X_train)

    print("Starting training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    print("Evaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    plot_history(history)


if __name__ == "__main__":
    main()
