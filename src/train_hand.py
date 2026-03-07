#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =============================
# CONFIG
# =============================
TRAIN_FILE = "data/splits_hand/train.csv"
VAL_FILE = "data/splits_hand/val.csv"

MODEL_OUT = "models/hand_gesture_mlp.keras"
SCALER_OUT = "models/hand_scaler.pkl"

os.makedirs("models", exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 32


# =============================
# FEATURE ENGINEERING
# =============================

def normalize_hand(kp):
    wrist = kp[0]
    kp = kp - wrist
    scale = np.linalg.norm(kp[9]) + 1e-6
    kp = kp / scale
    return kp


def angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def extract_features_from_flat(kp_flat):
    kp = kp_flat.reshape(21, 2)
    kp = normalize_hand(kp)

    features = []

    fingers = [
        [0,1,2,3,4],
        [0,5,6,7,8],
        [0,9,10,11,12],
        [0,13,14,15,16],
        [0,17,18,19,20]
    ]

    # Joint angles
    for f in fingers:
        features.append(angle(kp[f[0]], kp[f[1]], kp[f[2]]))
        features.append(angle(kp[f[1]], kp[f[2]], kp[f[3]]))

    # Fingertip distances from wrist
    tips = [4,8,12,16,20]
    for t in tips:
        features.append(np.linalg.norm(kp[t]))

    # Distances between fingertips
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            features.append(np.linalg.norm(kp[tips[i]] - kp[tips[j]]))

    # Hand orientation
    orientation = math.atan2(kp[9][1], kp[9][0])
    features.append(orientation)

    return np.array(features, dtype=np.float32)


# =============================
# LOAD DATA
# =============================

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    kp_cols = [c for c in df.columns if c.startswith("kp_")]

    raw = df[kp_cols].values.astype(np.float32)
    X = np.array([extract_features_from_flat(kp) for kp in raw])
    y = df["label"].values.astype(np.int32)

    return X, y


print("Loading dataset...")

X_train, y_train = load_data(TRAIN_FILE)
X_val, y_val = load_data(VAL_FILE)

print("Train:", X_train.shape)
print("Val:", X_val.shape)


# =============================
# NORMALIZATION
# =============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

joblib.dump(scaler, SCALER_OUT)


# =============================
# CLASS WEIGHTS
# =============================

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = dict(zip(classes, class_weights))


# =============================
# MODEL
# =============================

model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =============================
# TRAIN
# =============================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(patience=6, restore_best_weights=True),
        ModelCheckpoint(MODEL_OUT, save_best_only=True)
    ],
    verbose=1
)


# =============================
# EVALUATION
# =============================

print("\nEvaluating model...")

y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("models/confusion_matrix.png")
plt.close()

plt.figure(figsize=(7,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.savefig("models/accuracy_graph.png")
plt.close()

plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.savefig("models/loss_graph.png")
plt.close()

print("\n✅ Training complete")