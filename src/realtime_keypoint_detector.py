#!/usr/bin/env python3
"""
Realtime detector using geometric-features LSTM.

Loads model models/fall_geom_lstm.keras (same format as training script).
Computes geometric features per frame from MediaPipe keypoints, forms a sequence
buffer of length SEQ_LEN and runs inference.
"""
import os, time
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque

mp_pose = mp.solutions.pose

YOLO_MODEL = "yolov8n.pt"
MODEL_PATH = "models/fall_geom_lstm.keras"
SAVE_DIR = "detected_falls"
os.makedirs(SAVE_DIR, exist_ok=True)

yolo = YOLO(YOLO_MODEL)
model = load_model(MODEL_PATH)

SEQ_LEN = 8
GEOM_FEATURES = 12

# smoothing & thresholds
REQUIRED_STABLE = 4
prob_queue = deque(maxlen=SEQ_LEN)
FALL_PROB_THRESHOLD = 0.60
fall_memory = deque(maxlen=20)
FALL_PATTERN_REQUIRED = 8

# -------------------------
# geometric helpers (same as training)
# -------------------------
def angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)

def extract_geom_from_kp(kp):
    try:
        L_SHO = kp[11][:2].astype(float)
        R_SHO = kp[12][:2].astype(float)
        L_HIP = kp[23][:2].astype(float)
        R_HIP = kp[24][:2].astype(float)
        NOSE  = kp[0][:2].astype(float)
    except Exception:
        return np.zeros(GEOM_FEATURES, dtype=float)

    shoulders_center = (L_SHO + R_SHO) / 2.0
    hips_center = (L_HIP + R_HIP) / 2.0

    shoulder_angle = angle(L_SHO, R_SHO)
    hip_angle = angle(L_HIP, R_HIP)
    torso_angle = angle(shoulders_center, hips_center)

    body_height = np.linalg.norm(NOSE - hips_center)

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

def extract_keypoints(image, pose):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints)

# -------------------------
# realtime main
# -------------------------
def main():
    cap = cv2.VideoCapture(0)
    last_state = "no_fall"
    stable_counter = 0
    seq_buffer = deque(maxlen=SEQ_LEN)

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, verbose=False)
            bbox = None
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2-x1)*(y2-y1)
                        if bbox is None or area > (bbox[2]-bbox[0])*(bbox[3]-bbox[1]):
                            bbox = (x1,y1,x2,y2)

            if bbox is not None:
                x1,y1,x2,y2 = bbox
                crop = frame[y1:y2, x1:x2]
                kps = extract_keypoints(crop, pose)
                if kps is not None:
                    geom = extract_geom_from_kp(kps)
                    seq_buffer.append(geom)

                    if len(seq_buffer) < SEQ_LEN:
                        label_txt = f"WAITING ({len(seq_buffer)}/{SEQ_LEN})"
                        color = (200,200,0)
                        pred = 0.0
                    else:
                        X = np.array(seq_buffer).reshape(1, SEQ_LEN, GEOM_FEATURES)
                        pred = float(model.predict(X, verbose=0).ravel()[0])

                        prob_queue.append(pred)
                        smooth_pred = float(np.mean(prob_queue))

                        state = "fall" if smooth_pred > FALL_PROB_THRESHOLD else "no_fall"
                        fall_memory.append(state=="fall")
                        if sum(fall_memory) >= FALL_PATTERN_REQUIRED:
                            confirmed_state = "fall"
                        else:
                            confirmed_state = "no_fall"

                        final_state = confirmed_state

                        # smoothing/hysteresis:
                        if final_state == last_state:
                            stable_counter += 1
                        else:
                            stable_counter = 0
                            last_state = final_state
                        displayed_state = last_state if stable_counter >= REQUIRED_STABLE else "no_fall"

                        label_txt = f"{displayed_state.upper()} (p={smooth_pred:.2f})"
                        color = (0,0,255) if displayed_state=="fall" else (0,255,0)

                        if displayed_state == "fall":
                            fname = f"fall_{int(time.time())}_{smooth_pred:.2f}.jpg"
                            cv2.imwrite(os.path.join(SAVE_DIR, fname), frame)
                            print("⚠ Fall saved:", fname)

                    # draw
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, label_txt, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.imshow("Person Crop", crop)
                else:
                    cv2.putText(frame, "Pose not detected", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.putText(frame, "No person detected", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Fall Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
