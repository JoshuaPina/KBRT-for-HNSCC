#Typical Libraries

import pandas as pd #Pandas for DataFrame's
import numpy as np #Numpy for math
from PIL import Image #Pillow for image processing
from pathlib import Path #Pathlib as an OS replacement for paths
import matplotlib.pyplot as plt #Matplotlib for plotting data

#Tracking progress and time

import time
from datetime import datetime #To get the current time for a timestamp
from zoneinfo import ZoneInfo #To set my timezone
from tqdm import tqdm #tdqm for progress bars

#Possible Viz Enhancements

import seaborn as sns #Seaborn for advanced plotting (tbd)
from tabulate import tabulate #Tabulate for pretty tables (tbd)

#Machine Learning with Sci-Kit Learn, TensorFlow, and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
#CNN Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator #For data augmentation

#Constants for the model
IMAGE_SIZE = (128, 128)
DATASET_PATH = Path("../Data/data_images")
CSV_PATH = Path("../Data/data_sheet.csv")

df = pd.read_csv(CSV_PATH)
print("CSV shape:", df.shape)
df.head()

import gc
import psutil
import os
from PIL import Image

def load_and_match_images(image_path, size=IMAGE_SIZE):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size)
    return np.array(image) / 255.0 # Normalize to [0, 1]


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    return f"{mem_mb:.2f} MB"

# Adaptive garbage collection if memory exceeds threshold
def adaptive_gc(threshold_gb=35):
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    if mem_gb > threshold_gb:
        print(f"RAM at {mem_gb:.2f} GB — triggering garbage collection")
        gc.collect()

def process_large_folders(df, dataset_path, batch_size=1000, memory_threshold_gb=35):
    matched_data = []
    start = time.time()
    total_batches = (len(df) - 1) // batch_size + 1

    print(f"🚀 Starting Processing")
    print(f"Total rows: {len(df)} | Batch size: {batch_size} | Total batches: {total_batches}")
    print(f"Initial memory usage: {get_memory_usage()}")
    print("-" * 60)

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        batch_num = batch_start // batch_size + 1
        print(f"\n Batch {batch_num}/{total_batches} (Rows {batch_start}-{batch_end-1})")
        print(f"Memory before batch: {get_memory_usage()}")

        batch_data = []
        for idx, (_, row) in enumerate(tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Processing Batch {batch_num}")):
            folder_name = str(row['dummy_id']) + "_ct_images"
            folder_path = dataset_path / folder_name

            if folder_path.exists():
                image_files = sorted(folder_path.glob("*.jpg"))
                image_batch = []

                for j, img_path in enumerate(image_files):
                    img = load_and_match_images(img_path)
                    image_batch.append(img)

                    if (j + 1) % 1000 == 0:
                        adaptive_gc(threshold_gb=memory_threshold_gb)

                if image_batch:
                    batch_data.append((row, image_batch.copy()))
                    del image_batch
                    gc.collect()
            else:
                if idx < 5:
                    print(f"  ⚠️ Missing folder: {folder_path}")

        matched_data.extend(batch_data)
        del batch_data
        gc.collect()

        print(f"Batch {batch_num} complete | Memory now: {get_memory_usage()} | Total matched: {len(matched_data)}")

    end = time.time()
    print("\nPROCESSING COMPLETE")
    print(f"Total time: {end - start:.2f}s | Avg per batch: {(end - start)/total_batches:.2f}s")
    print(f"Final memory usage: {get_memory_usage()} | Total matched folders: {len(matched_data)}")
    print("=" * 60)

    return matched_data
# Call the function to start the processing
matched_data = process_large_folders(df, DATASET_PATH, batch_size=50)


# Filter only L/R labels and map to binary
binary_df = df[df["Tumor laterality"].isin(["L", "R"])].copy()
binary_df["Binary Label"] = binary_df["Tumor laterality"].map({"L": 0, "R": 1})

print("Preview of binary_df:")
print(binary_df[["dummy_id", "Tumor laterality", "Binary Label"]].head())

X = []
y = []

# Ensure binary_df is indexed by dummy_id for fast lookup
binary_df.set_index("dummy_id", inplace=True)

for row, images in matched_data:
    dummy_id = row["dummy_id"]

    # Skip any samples not labeled as L/R (already filtered in binary_df)
    if dummy_id in binary_df.index:
        label = binary_df.loc[dummy_id, "Binary Label"]
        
        # You can change this to any image reduction strategy you prefer
        avg_image = np.mean(images, axis=0)  # Shape: (H, W) or (H, W, 1)
        
        X.append(avg_image)
        y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Optional: Expand dims if grayscale
if len(X.shape) == 3:  # (samples, height, width)
    X = X[..., np.newaxis]  # → (samples, height, width, 1)

#Train-Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Data Augmentation for CT Scans

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

#Reading CSV to Print DataSet
df = pd.read_csv(CSV_PATH)
print()
print("-"*60)
print("Dataframe shape:", df.shape)
print("-"*60)
print("Selected Parameter:")
print()
df["Tumor laterality"] = df["Tumor laterality"].str.strip().str.upper()
laterality_counts = df["Tumor laterality"].value_counts(dropna=False)
print(laterality_counts.to_string())
print()
print("-"*60)
print()

#CNN Model Defined
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)
])

#Model Compilation with ADAM
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Model Summary Print
model.summary()

#How to Train Your Dragon (or model)
history = model.fit(X_train, y_train, epochs=15, 
                    validation_data=(X_test, y_test))


# Evaluate The Model w/ Test Data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

# Plot training & validation
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('cnn_accuracy_plot.png') 
plt.show()

model.save('HNSCC_CNN_Model.h5')

