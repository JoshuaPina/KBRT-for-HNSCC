import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (128, 128)
BASE_DIR = Path("/workspaces/KBRT-for-HNSCC/Pina Folder/Test Folder 2")
CSV_PATH = BASE_DIR / "Data" / "data_sheet.csv"
IMAGE_PATH = BASE_DIR / "Data" / "data_images"

# Load CSV and filter L/R
df = pd.read_csv(CSV_PATH)
df = df[df["Tumor laterality"].isin(["L", "R"])]
df["Tumor laterality"] = df["Tumor laterality"].str.strip().str.upper()
df["Binary Label"] = df["Tumor laterality"].map({"L": 0, "R": 1})
df.set_index("dummy_id", inplace=True)

# Load one averaged image per folder
X, y = [], []

for dummy_id, row in df.iterrows():
    folder_path = IMAGE_PATH / f"{dummy_id}_ct_images"
    if not folder_path.exists():
        continue
    image_files = sorted(folder_path.glob("*.jpg"))
    if not image_files:
        continue
    images = [np.array(Image.open(img).convert("RGB")) / 255.0 for img in image_files]
    avg_img = np.mean(images, axis=0)
    X.append(avg_img)
    y.append(row["Binary Label"])

X = np.array(X)
y = np.array(y)

# Expand dims if grayscale
if len(X.shape) == 3:
    X = X[..., np.newaxis]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")

# Plot
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend(); plt.title('Accuracy'); plt.grid(); plt.savefig('cnn_accuracy.png'); plt.show()

# Save model
model.save("HNSCC_CNN_Model.h5")
