
# --- Imports ---
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import gc
import psutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Constants ---
IMAGE_SIZE = (128, 128)
BASE_DIR = Path(__file__).resolve().parent.parent 
CSV_PATH = BASE_DIR / "Data" / "data_sheet.csv"
DATASET_PATH = BASE_DIR / "Data" / "data_images"

# --- Load CSV ---
df = pd.read_csv(CSV_PATH)
print("CSV shape:", df.shape)

# --- Preprocessing: Filter for Binary Classes (L, R) ---
binary_df = df[df["Tumor laterality"].isin(["L", "R"])].copy()
binary_df["Binary Label"] = binary_df["Tumor laterality"].map({"L": 0, "R": 1})
binary_df.set_index("dummy_id", inplace=True)

# --- Image Loader (No Resize Needed) ---
def load_image(path):
    image = Image.open(path).convert("L")  # Grayscale
    return np.array(image) / 255.0         # Normalize

# --- Match Data ---
X, y = [], []

for _, row in tqdm(binary_df.reset_index().iterrows(), total=binary_df.shape[0], desc="Matching Images"):
    dummy_id = row["dummy_id"]
    folder_path = DATASET_PATH / f"{dummy_id}_ct_images"

    if folder_path.exists():
        image_files = sorted(folder_path.glob("*.jpg"))
        images = [load_image(p) for p in image_files]

        if images:
            avg_image = np.mean(images, axis=0)
            X.append(avg_image)
            y.append(row["Binary Label"])
    else:
        print(f"Missing folder: {folder_path}")

# --- Convert to Arrays ---
X = np.array(X)
y = np.array(y)

# --- Expand dims for CNN ---
if len(X.shape) == 3:
    X = X[..., np.newaxis]  # (N, H, W, 1)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Safe Augmentation (No flips) ---
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

# --- Model Definition ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# --- Compile ---
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Train Model ---
start = time.time()
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=15,
                    verbose=1)
end = time.time()
print(f"Training complete in {(end - start):.2f} seconds.")

# --- Evaluate ---
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

# --- Save Plot ---
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("CNN Tumor Laterality Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tumor_laterality_cnn_accuracy_plot.png")
plt.show()

# --- Save Model ---
model.save("tumor_laterality_CNN_model.h5")
print("Model saved as: tumor_laterality_CNN_model.h5")

