#!/usr/bin/env python3
"""
Train an LSTM using geometric features (Option A).

Outputs:
 - models/fall_geom_lstm.keras
 - outputs/plots/geom_acc.png, geom_loss.png, geom_cm.png
 - outputs/report/geom_report.txt

Expects per-video CSVs in data/splits/train and data/splits/val
Each per-video CSV must contain header:
video_id,frame_idx,label,kp_0,...,kp_131
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
from glob import glob

# Config
TRAIN_DIR = "data/splits/train"
VAL_DIR = "data/splits/val"
SEQ_LEN = 8
STRIDE = 1
GEOM_FEATURES = 12
BATCH_SIZE = 32
EPOCHS = 20
MODEL_OUT = "models/fall_geom_lstm.keras"
PLOTS_DIR = "outputs/plots"
REPORT_DIR = "outputs/report"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# -----------------------------
# geometric helpers
# -----------------------------
def angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)

def extract_geom_from_frame(kp):
    """
    kp: (33,4) array in *frame-local normalized coords* (x,y,z,vis)
    returns GEOM_FEATURES array (floats)
    Indices correspond to Mediapipe's PoseLandmark:
      0 = nose, 11 = left_shoulder, 12 = right_shoulder, 23 = left_hip, 24 = right_hip
    """
    # defensive: if any landmark is missing, fill zeros
    try:
        L_SHO = kp[11][:2].astype(float)
        R_SHO = kp[12][:2].astype(float)
        L_HIP = kp[23][:2].astype(float)
        R_HIP = kp[24][:2].astype(float)
        NOSE  = kp[0][:2].astype(float)
    except Exception:
        # fallback zero vector
        return np.zeros(GEOM_FEATURES, dtype=float)

    shoulders_center = (L_SHO + R_SHO) / 2.0
    hips_center = (L_HIP + R_HIP) / 2.0

    shoulder_angle = angle(L_SHO, R_SHO)
    hip_angle = angle(L_HIP, R_HIP)
    torso_angle = angle(shoulders_center, hips_center)

    body_height = np.linalg.norm(NOSE - hips_center)  # euclidean in normalized coords

    xs = kp[:,0]
    ys = kp[:,1]
    w = xs.max() - xs.min() + 1e-6
    h = ys.max() - ys.min() + 1e-6
    aspect = (h / w)

    dist_sh_hip = np.linalg.norm(shoulders_center - hips_center)

    feat = np.array([
        shoulder_angle,
        hip_angle,
        torso_angle,
        body_height,
        aspect,
        shoulders_center[0],
        shoulders_center[1],
        hips_center[0],
        hips_center[1],
        NOSE[0],
        NOSE[1],
        dist_sh_hip
    ], dtype=float)

    return feat

# -----------------------------
# CSV -> sequences
# -----------------------------
def read_csvs(folder):
    files = sorted(glob(os.path.join(folder, "*.csv")))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print("Failed to read", f, e)
    return dfs

def sequences_from_df(df):
    # expects kp_0..kp_131 columns (flattened x,y,z,vis for 33 landmarks)
    kp_cols = [c for c in df.columns if c.startswith("kp_")]
    if len(kp_cols) != 33*4:
        raise ValueError(f"Expected 132 kp columns, found {len(kp_cols)} in {df.shape}")
    arr = df[kp_cols].values.reshape(len(df), 33, 4)
    labels = df["label"].values.astype(int)

    geom = np.stack([extract_geom_from_frame(arr[i]) for i in range(len(arr))])

    X = []
    y = []
    n = len(df)
    if n < SEQ_LEN:
        return np.empty((0, SEQ_LEN, GEOM_FEATURES)), np.empty((0,), dtype=int)
    for start in range(0, n - SEQ_LEN + 1, STRIDE):
        seq = geom[start:start+SEQ_LEN]
        seq_label = 1 if labels[start:start+SEQ_LEN].any() else 0
        X.append(seq)
        y.append(seq_label)
    return np.array(X), np.array(y, dtype=int)

# -----------------------------
# augment per-sequence
# -----------------------------
def augment(X, y, n_aug=1):
    X2 = []
    y2 = []
    for seq, label in zip(X, y):
        X2.append(seq)
        y2.append(label)
        for _ in range(n_aug):
            # small gaussian noise (per-feature)
            noise = np.random.normal(0, 0.03, seq.shape)
            X2.append(seq + noise)
            y2.append(label)
    return np.array(X2), np.array(y2, dtype=int)

# -----------------------------
# build model
# -----------------------------
def build_model():
    model = Sequential([
        Masking(mask_value=0., input_shape=(SEQ_LEN, GEOM_FEATURES)),
        LSTM(64, return_sequences=False),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -----------------------------
# prepare folder
# -----------------------------
def prepare_folder(folder):
    dfs = read_csvs(folder)
    Xs = []
    Ys = []
    for df in dfs:
        Xv, yv = sequences_from_df(df)
        if Xv.size == 0:
            continue
        Xs.append(Xv)
        Ys.append(yv)
    if not Xs:
        return np.empty((0, SEQ_LEN, GEOM_FEATURES)), np.empty((0,), dtype=int)
    return np.vstack(Xs), np.hstack(Ys)

# -----------------------------
# main
# -----------------------------
def main():
    print("Preparing geometric datasets...")
    X_train, y_train = prepare_folder(TRAIN_DIR)
    X_val, y_val = prepare_folder(VAL_DIR)
    print("Train:", X_train.shape, "Val:", X_val.shape)

    if X_train.size == 0 or X_val.size == 0:
        print("Not enough data to train. Check data/splits/train and /val")
        return

    X_train, y_train = augment(X_train, y_train, n_aug=1)
    print("After aug:", X_train.shape)

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint('models/checkpoint_geom.keras', save_best_only=True)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

    model.save(MODEL_OUT)
    print("Saved model to", MODEL_OUT)

    # plots
    plt.figure(); plt.plot(history.history['accuracy'], label='train'); plt.plot(history.history['val_accuracy'], label='val'); plt.title('Accuracy'); plt.legend(); plt.savefig(os.path.join(PLOTS_DIR, 'geom_acc.png')); plt.close()
    plt.figure(); plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='val'); plt.title('Loss'); plt.legend(); plt.savefig(os.path.join(PLOTS_DIR, 'geom_loss.png')); plt.close()

    # eval
    y_prob = model.predict(X_val)
    y_pred = (y_prob > 0.5).astype(int).ravel()
    rpt = classification_report(y_val, y_pred, target_names=['no_fall','fall'])
    with open(os.path.join(REPORT_DIR, 'geom_report.txt'), 'w') as f:
        f.write(rpt)
    print("\nClassification Report:\n", rpt)

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(4,4)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=['no_fall','fall'], yticklabels=['no_fall','fall']); plt.savefig(os.path.join(PLOTS_DIR, 'geom_cm.png')); plt.close()

if __name__ == "__main__":
    main()
