# src/train_keypoint_model_v3.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns

FALL_CSV = "data/keypoints/fall_keypoints.csv"
NO_CSV = "data/keypoints/no_fall_keypoints.csv"

def load_data():
    df_fall = pd.read_csv(FALL_CSV, header=None)
    df_no = pd.read_csv(NO_CSV, header=None)

    df_fall['label'] = 1
    df_no['label'] = 0

    df = pd.concat([df_fall, df_no]).sample(frac=1).reset_index(drop=True)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

# --------------------------
# Simple keypoint augmentation
# --------------------------
def augment_keypoints(X, y, n_aug=2, scale_range=(0.95, 1.05), rotation_range=(-10,10)):
    """
    Augment keypoints by random scaling and rotation (in degrees).
    n_aug: number of augmented copies per sample
    """
    X_aug = []
    y_aug = []

    for xi, yi in zip(X, y):
        keypoints = xi.reshape(-1, 4)  # 33 landmarks with x,y,z,visibility
        for _ in range(n_aug):
            kp_aug = keypoints.copy()
            # scale (x, y, z)
            scale = np.random.uniform(*scale_range)
            kp_aug[:, :3] *= scale

            # rotation around z-axis (yaw) for x,y only
            angle = np.radians(np.random.uniform(*rotation_range))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = kp_aug[:, 0] * cos_a - kp_aug[:, 1] * sin_a
            y_rot = kp_aug[:, 0] * sin_a + kp_aug[:, 1] * cos_a
            kp_aug[:, 0] = x_rot
            kp_aug[:, 1] = y_rot

            X_aug.append(kp_aug.flatten())
            y_aug.append(yi)

    X_combined = np.vstack([X] + [np.array(X_aug)])
    y_combined = np.hstack([y] + [np.array(y_aug)])
    return X_combined, y_combined

def build_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    X, y = load_data()
    # --------------------------
    # Apply augmentation
    # --------------------------
    X, y = augment_keypoints(X, y, n_aug=2)
    print(f"After augmentation: {X.shape[0]} samples")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

    model = build_model(X.shape[1])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_val, y_val))

    # Save model
    model.save("models/fall_keypoint_model.h5")
    print("Model saved to models/fall_keypoint_model.h5")

    # --------------------------
    # Plot training/validation accuracy & loss
    # --------------------------
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # --------------------------
    # Evaluate on validation set
    # --------------------------
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred, target_names=['no_fall', 'fall']))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['no_fall','fall'], yticklabels=['no_fall','fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__=="__main__":
    main()
