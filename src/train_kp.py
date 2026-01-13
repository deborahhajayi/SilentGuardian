#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from glob import glob
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

# --------------------
# CONFIG
# --------------------
TRAIN_DIR = "data/splits/train"
VAL_DIR   = "data/splits/val"

SEQ_LEN = 8
GEOM_FEATURES = 13

BATCH_SIZE = 32
EPOCHS = 20

MODEL_OUT = "models/fall_geom_lstm.keras"
os.makedirs("models", exist_ok=True)

# --------------------
# GEOMETRY HELPERS
# --------------------
def angle(a, b):
    return np.arctan2(b[1] - a[1], b[0] - a[0])

def geom_from_kp(kp, median_height):
    """
    kp: (17,3) → (x,y,score)
    """
    try:
        NOSE = kp[0][:2]
        L_SHO, R_SHO = kp[5][:2], kp[6][:2]
        L_HIP, R_HIP = kp[11][:2], kp[12][:2]
    except Exception:
        return np.zeros(GEOM_FEATURES, dtype=np.float32)

    shoulders = (L_SHO + R_SHO) / 2
    hips = (L_HIP + R_HIP) / 2

    body_height = np.linalg.norm(NOSE - hips)
    relative_height = body_height / (median_height + 1e-6)

    xs = kp[:, 0]
    ys = kp[:, 1]
    aspect = (ys.max() - ys.min() + 1e-6) / (xs.max() - xs.min() + 1e-6)

    return np.array([
        angle(L_SHO, R_SHO),      # 0 shoulder_angle
        angle(L_HIP, R_HIP),      # 1 hip_angle
        angle(shoulders, hips),   # 2 torso_angle
        body_height,              # 3
        aspect,                   # 4
        shoulders[0],             # 5
        shoulders[1],             # 6
        hips[0],                  # 7
        hips[1],                  # 8
        NOSE[0],                  # 9
        NOSE[1],                  # 10
        np.linalg.norm(shoulders - hips),  # 11
        relative_height           # 12
    ], dtype=np.float32)

# --------------------
# SEQUENCE BUILDER
# --------------------
def sequences_from_df(df):
    kp_cols = [c for c in df.columns if c.startswith("kp_")]
    if len(kp_cols) != 17 * 3:
        raise ValueError(f"Expected 51 kp columns, got {len(kp_cols)}")

    arr = df[kp_cols].values.reshape(len(df), 17, 3)
    labels = df["label"].astype(int).values

    # per-video median height baseline
    heights = []
    for kp in arr:
        try:
            h = np.linalg.norm(kp[0][:2] - ((kp[11][:2] + kp[12][:2]) / 2))
            if h > 0 and np.isfinite(h):
                heights.append(h)
        except Exception:
            pass
    median_h = float(np.median(heights)) if heights else 1.0

    geom = np.stack([geom_from_kp(kp, median_h) for kp in arr])

    X, y = [], []
    for i in range(len(df) - SEQ_LEN + 1):
        X.append(geom[i:i + SEQ_LEN])
        y.append(int(labels[i:i + SEQ_LEN].any()))

    return np.array(X), np.array(y)

def load_folder(folder):
    Xs, Ys = [], []
    for f in glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(f)
        X, y = sequences_from_df(df)
        if len(X):
            Xs.append(X)
            Ys.append(y)
    if not Xs:
        return np.empty((0, SEQ_LEN, GEOM_FEATURES)), np.empty((0,))
    return np.vstack(Xs), np.hstack(Ys)

# --------------------
# LOAD DATA
# --------------------
X_train, y_train = load_folder(TRAIN_DIR)
X_val, y_val = load_folder(VAL_DIR)

print("Train:", X_train.shape, "Val:", X_val.shape)
print("Fall ratio (train):", y_train.mean())

if X_train.size == 0 or X_val.size == 0:
    raise RuntimeError("❌ Not enough data to train")

# --------------------
# MODEL
# --------------------
model = Sequential([
    Masking(mask_value=0., input_shape=(SEQ_LEN, GEOM_FEATURES)),
    LSTM(64),
    Dropout(0.25),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(patience=6, restore_best_weights=True),
        ModelCheckpoint(MODEL_OUT, save_best_only=True)
    ],
    verbose=1
)

# --------------------
# EVAL
# --------------------
yp = (model.predict(X_val) > 0.5).astype(int)
print(classification_report(y_val, yp))
