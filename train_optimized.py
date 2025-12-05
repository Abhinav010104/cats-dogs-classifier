import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# ========= 1) Download curated dataset =========
URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = keras.utils.get_file(
    fname="cats_and_dogs_filtered.zip",
    origin=URL
)

base_dir = os.path.join(os.path.dirname(zip_path), "cats_and_dogs_filtered")
if not os.path.exists(base_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.dirname(zip_path))

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# ========= 2) Create datasets =========
img_size = (224, 224)  # MobileNetV2 default
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

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ========= 3) Data Augmentation =========
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ========= 4) Transfer Learning Model =========
base_model = keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze feature extractor

inputs = keras.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)  # regularization
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)



for images, labels in train_ds.take(1):
    print("Labels:", labels.numpy())
    print("Images shape:", images.numpy().shape)

# ========= 5) Train model (head only) =========
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ========= 6) Fine-tune (unfreeze deeper layers) =========
base_model.trainable = True
for layer in base_model.layers[:-60]:  # unfreeze last ~60 layers
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ========= 7) Save Model =========
model.save("best_model.h5")       # HDF5 format
model.save("best_model.keras")    # New Keras format

# ========= 8) Save Accuracy/Loss Curves =========
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_finetune.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + history_finetune.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.savefig("training_curves.png")
print("\n✅ Training done. Model saved as best_model.h5 & best_model.keras")