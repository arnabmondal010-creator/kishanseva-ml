import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---- Config ----
DATA_DIR = "PlantVillage"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16          # Lower for CPU stability
EPOCHS = 5               # Increase later to 15–25 for better accuracy
MODEL_PATH = "model/plant_disease.h5"

# ---- Data Generators ----
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_data.num_classes
print("Classes:", train_data.class_indices)

# ---- Model (Lightweight CNN) ----
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---- Train ----
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ---- Save ----
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
