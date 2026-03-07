#!/usr/bin/env python3
import cv2
import numpy as np
import joblib
import tensorflow as tf
import math
from ultralytics import YOLO

MODEL_PATH = "models/hand_gesture_mlp.keras"
SCALER_PATH = "models/hand_scaler.pkl"
YOLO_MODEL_PATH = "models/yolov8_hand_pose.pt"

THRESH = 0.6

CLASS_NAMES = ["like", "palm", "neutral"]


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

    for f in fingers:
        features.append(angle(kp[f[0]], kp[f[1]], kp[f[2]]))
        features.append(angle(kp[f[1]], kp[f[2]], kp[f[3]]))

    tips = [4,8,12,16,20]
    for t in tips:
        features.append(np.linalg.norm(kp[t]))

    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            features.append(np.linalg.norm(kp[tips[i]] - kp[tips[j]]))

    orientation = math.atan2(kp[9][1], kp[9][0])
    features.append(orientation)

    return np.array(features, dtype=np.float32)


# =============================
# LOAD MODELS
# =============================

print("Loading models...")
mlp_model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)


# =============================
# KEYPOINT EXTRACTION
# =============================

def extract_keypoints(frame):
    results = yolo_model(frame, verbose=False)
    if len(results) == 0:
        return None

    r = results[0]
    if r.keypoints is None:
        return None

    kp = r.keypoints.xy
    if kp is None or len(kp) == 0:
        return None

    kp = kp[0].cpu().numpy()
    return kp[:, :2].flatten()


def draw_keypoints(frame, kp_flat):
    if kp_flat is None:
        return

    pts = kp_flat.reshape(21, 2)
    for x, y in pts:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)


# =============================
# REALTIME LOOP
# =============================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting realtime detection... (Press Q to quit)")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gesture_text = "WAIT"
    color = (0,255,255)

    kp_flat = extract_keypoints(frame)

    if kp_flat is not None:

        try:
            features = extract_features_from_flat(kp_flat)
            X = scaler.transform([features])

            prob = mlp_model.predict(X, verbose=0)[0]
            pred_idx = int(np.argmax(prob))
            confidence = float(np.max(prob))

            if confidence > THRESH:
                if CLASS_NAMES[pred_idx] != "neutral":
                    gesture_text = f"{CLASS_NAMES[pred_idx]} ({confidence:.2f})"
                    color = (0,255,0)

        except Exception as e:
            print("Prediction error:", e)

        draw_keypoints(frame, kp_flat)

    cv2.putText(frame, gesture_text, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("SilentGuardian Hand Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()