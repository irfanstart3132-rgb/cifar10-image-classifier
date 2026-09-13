"""
model.py
--------
Defines the CNN architecture used for CIFAR-10 classification.

Kept in its own file so train.py, evaluate.py, and predict.py
can all build an identical network without duplicating code.

Colab note: upload this file to the root of your Colab session
(no subfolders needed) alongside train.py, evaluate.py, predict.py.
"""

from tensorflow.keras import layers, models

# CIFAR-10 constants
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3
NUM_CLASSES = 10

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_model():
    """
    Builds and returns a compiled CNN for 32x32x3 RGB images.

    Architecture (shapes assume batch size omitted):

    Input          (32, 32, 3)
      -> Conv2D(32, 3x3) + ReLU   -> (30, 30, 32)
      -> MaxPooling2D(2x2)        -> (15, 15, 32)
      -> Conv2D(64, 3x3) + ReLU   -> (13, 13, 64)
      -> MaxPooling2D(2x2)        -> (6, 6, 64)
      -> Conv2D(64, 3x3) + ReLU   -> (4, 4, 64)
      -> Flatten                  -> (1024,)
      -> Dense(64) + ReLU         -> (64,)
      -> Dropout(0.5)             -> (64,)
      -> Dense(10) + Softmax      -> (10,)
    """
    model = models.Sequential(name="cifar10_cnn")

    # Block 1: learns simple local patterns (edges, color blobs)
    model.add(layers.Conv2D(
        32, (3, 3), activation="relu",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        name="conv1"
    ))
    model.add(layers.MaxPooling2D((2, 2), name="pool1"))

    # Block 2: combines Block 1's patterns into more complex shapes
    model.add(layers.Conv2D(64, (3, 3), activation="relu", name="conv2"))
    model.add(layers.MaxPooling2D((2, 2), name="pool2"))

    # Block 3: learns higher-level features (textures, object parts)
    model.add(layers.Conv2D(64, (3, 3), activation="relu", name="conv3"))

    # Flatten converts the 3D feature map into a 1D vector for Dense layers
    model.add(layers.Flatten(name="flatten"))

    # Dense "reasoning" layer that combines all extracted features
    model.add(layers.Dense(64, activation="relu", name="dense1"))

    # Dropout randomly zeroes 50% of activations during training only,
    # which reduces overfitting by preventing co-dependency between neurons
    model.add(layers.Dropout(0.5, name="dropout"))

    # Output layer: one neuron per class, softmax turns raw scores into
    # a probability distribution that sums to 1
    model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="output"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Quick sanity check: print the architecture and shapes
    m = build_model()
    m.summary()
