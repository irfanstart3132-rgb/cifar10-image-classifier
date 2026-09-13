# cifar10-image-classifier
A CNN image classifier built from scratch with TensorFlow/Keras, trained on CIFAR-10 to recognize 10 object categories, includes training, evaluation, and prediction scripts.
# CIFAR-10 Image Classifier (CNN, TensorFlow/Keras)

A from-scratch, fully runnable image classification project. Trains a
convolutional neural network on CIFAR-10 to recognize 10 categories of
everyday objects, then lets you run predictions on your own images.

## Project structure

```
image-classifier/
├── data/                   # (CIFAR-10 downloads here automatically)
├── models/                 # trained model + training curves saved here
├── src/
│   ├── model.py             # CNN architecture definition
│   ├── train.py             # loads data, trains, saves the model
│   ├── evaluate.py          # detailed evaluation on the test set
│   └── predict.py           # classify a single new image
├── requirements.txt
└── README.md
```

## 1. Install

```bash
cd image-classifier
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train

```bash
python src/train.py
```

This downloads CIFAR-10 automatically (first run only, ~170 MB),
trains for 15 epochs, and saves:
- `models/cifar10_cnn.keras` — the trained model
- `models/training_history.png` — accuracy/loss curves

Expected output (numbers vary by run):
```
Train: (40000, 32, 32, 3), Val: (10000, 32, 32, 3), Test: (10000, 32, 32, 3)
Epoch 1/15
625/625 [==============================] - 12s - loss: 1.62 - accuracy: 0.41 - val_loss: 1.35 - val_accuracy: 0.51
...
Epoch 15/15
625/625 [==============================] - 11s - loss: 0.78 - accuracy: 0.73 - val_loss: 0.82 - val_accuracy: 0.71
Test accuracy: 0.7050 | Test loss: 0.8420
Model saved to ../models/cifar10_cnn.keras
```

A CNN this size typically reaches **68-75% test accuracy** after 15
epochs on CIFAR-10 (random guessing would be 10%). Training on CPU
takes roughly 10-20 minutes; a GPU cuts this to 1-2 minutes.

## 3. Evaluate

```bash
python src/evaluate.py
```

Prints overall accuracy, per-class accuracy (some classes like "cat"
vs "dog" are harder than others like "automobile" or "truck"), and a
confusion matrix showing which classes get mixed up with each other.

## 4. Predict on a new image

```bash
python src/predict.py path/to/your_photo.jpg
```

Works with any JPG/PNG. The script resizes it to 32x32 and runs it
through the model. Example output:

```
Prediction: dog (confidence: 87.32%)

Full probability distribution:
  dog         : 0.8732
  cat         : 0.0891
  horse       : 0.0203
  ...
```

Note: CIFAR-10 images are small (32x32) and low-resolution, so the
model works best on images where the subject is large, centered, and
unobstructed. It was trained only on the 10 classes below — anything
else will be forced into the closest-matching category.

## Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Common errors and fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install -r requirements.txt` inside your virtual environment |
| `Error: no trained model found` (in predict.py) | Run `python src/train.py` first — it must complete before predicting |
| Very slow training / no GPU detected | Normal on CPU-only machines; reduce `EPOCHS` in `train.py` for a quicker test run |
| `OOM` / out-of-memory during training | Lower `BATCH_SIZE` in `train.py` (e.g., from 64 to 32) |
| CIFAR-10 download fails | Check your internet connection; Keras downloads it from `https://www.cs.toronto.edu/~kriz/cifar.html` on first run |
| Predictions look random/wrong for real photos | Make sure the image is one of the 10 trained classes and the subject fills most of the frame |

## Why these design choices?

- **Adam optimizer**: adapts the learning rate per parameter automatically, converges faster than plain SGD with less tuning.
- **sparse_categorical_crossentropy**: used instead of `categorical_crossentropy` because our labels are plain integers (0-9), not one-hot vectors — this saves a manual conversion step.
- **Dropout(0.5)**: only active during training; forces the network to not rely too heavily on any single neuron, which reduces overfitting.
- **Data augmentation**: CIFAR-10 has only 50,000 training images, which is small for a CNN. Random flips/shifts/rotations act as "more data for free."
