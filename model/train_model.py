import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# Create folder to save model artifacts
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Load MNIST dataset from TensorFlow/Keras
# No manual download is required
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Original x_train shape:", x_train.shape)
print("Original x_test shape:", x_test.shape)

# 2. Preprocessing
# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to (28, 28, 1) because CNN expects height, width, channels
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoded vectors
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("Processed x_train shape:", x_train.shape)
print("Processed x_test shape:", x_test.shape)

# 3. Build CNN model
model = Sequential([
    Input(shape=(28, 28, 1)),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Train model
history = model.fit(
    x_train, y_train_cat,
    validation_split=0.1,
    epochs=5,
    batch_size=64,
    verbose=1
)

# 5. Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 6. Predictions for confusion matrix
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7. Save confusion matrix
np.save(os.path.join(SAVE_DIR, "confusion_matrix.npy"), cm)

# 8. Save class-wise accuracy
class_accuracy = {}
for digit in range(10):
    digit_indices = np.where(y_test == digit)[0]
    digit_correct = np.sum(y_pred[digit_indices] == y_test[digit_indices])
    digit_acc = digit_correct / len(digit_indices)
    class_accuracy[str(digit)] = round(float(digit_acc), 4)

with open(os.path.join(SAVE_DIR, "class_accuracy.json"), "w") as f:
    json.dump({
        "overall_accuracy": round(float(test_accuracy), 4),
        "overall_loss": round(float(test_loss), 4),
        "class_accuracy": class_accuracy
    }, f, indent=4)

# 9. Save training history
history_data = {
    "accuracy": [float(x) for x in history.history["accuracy"]],
    "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
    "loss": [float(x) for x in history.history["loss"]],
    "val_loss": [float(x) for x in history.history["val_loss"]]
}

with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
    json.dump(history_data, f, indent=4)

# 10. Save trained model
model.save(os.path.join(SAVE_DIR, "digit_cnn.keras"))
print("\nModel saved successfully!")

# 11. Optional: Save training graphs as images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"))
plt.show()


