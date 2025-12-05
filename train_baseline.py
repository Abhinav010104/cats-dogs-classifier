import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# ========= 1) Download curated dataset (no corrupt files) =========
URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = keras.utils.get_file(
    fname="cats_and_dogs_filtered.zip",
    origin=URL
)

# Extract manually (only once)
base_dir = os.path.join(os.path.dirname(zip_path), "cats_and_dogs_filtered")
if not os.path.exists(base_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.dirname(zip_path))

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# ========= 2) Create datasets =========
img_size = (180, 180)
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Improve performance with prefetch
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ========= 3) Data Augmentation =========
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ========= 4) Build CNN Model =========
model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # binary output
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ========= 5) Train Model =========
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ========= 6) Plot & Save Curves =========
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.savefig("training_curves.png")   # ✅ Saves plot as PNG
print("\n✅ Training complete. Curves saved as 'training_curves.png'")