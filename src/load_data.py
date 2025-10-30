"""
# src/load_data.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --------------------
# CONFIG
# --------------------
IMG_SIZE = (128, 128)      # must match preprocess target size
BATCH_SIZE = 32
DATA_DIR = "data"           # parent folder containing train/val/test

# --------------------
# IMAGE DATA GENERATORS
# --------------------

# Training generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Validation and Test generators (no augmentation, only rescale)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# --------------------
# LOAD DATA
# --------------------

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',   # 0=no_fall, 1=fall
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --------------------
# PRINT INFO
# --------------------
print("Class indices:", train_gen.class_indices)
print(f"Train batches: {len(train_gen)}, Validation batches: {len(val_gen)}, Test batches: {len(test_gen)}")

# Optional: peek one batch
# x_batch, y_batch = next(train_gen)
# print("Batch shape:", x_batch.shape, y_batch.shape)

"""