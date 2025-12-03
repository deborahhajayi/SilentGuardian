#!/usr/bin/env python3
"""
Raspberry Pi 5 Realtime Fall Detector using:
- YOLOv8 (person detection)
- MediaPipe Pose
- 13 geometric features
- TFLite LSTM model inference
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import tensorflow as tf

mp_pose = mp.solutions.pose

# -------------------------
# PATHS
# -------------------------
YOLO_MODEL = "yolov8n.pt"
TFLITE_MODEL_PATH = "models/fall_geom_lstm.tflite"
SAVE_DIR = "detected_falls"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# LOAD MODELS
# -------------------------
yolo = YOLO(YOLO_MODEL)

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------
# SETTINGS
# -------------------------
SEQ_LEN = 8
GEOM_FEATURES = 13

REQUIRED_STABLE = 4
FALL_PROB_THRESHOLD = 0.50

prob_queue = deque(maxlen=SEQ_LEN)
fall_memory = deque(maxlen=20)
FALL_PATTERN_REQUIRED = 8

BASELINE_WINDOW = 300
baseline_heights = deque(maxlen=BASELINE_WINDOW)

MIN_LANDMARK_VIS = 0.45

# -------------------------
# GEOMETRIC HELPERS
# -------------------------
def angle(p1, p2):
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return np.arctan2(dy, dx)

def compute_body_height_and_features(kp):
    try:
        L_SHO = kp[11][:2].astype(float)
        R_SHO = kp[12][:2].astype(float)
        L_HIP = kp[23][:2].astype(float)
        R_HIP = kp[24][:2].astype(float)
        NOSE  = kp[0][:2].astype(float)
    except Exception:
        return None, None, False

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

    feat12 = np.array([
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
    ], dtype=np.float32)

    return feat12, float(body_height), True

def extract_geom_from_kp(kp):
    feat12, body_height, valid = compute_body_height_and_features(kp)
    if not valid:
        return np.zeros(GEOM_FEATURES, dtype=np.float32)

    if len(baseline_heights) > 0:
        median_h = float(np.median(np.array(baseline_heights)))
        if median_h <= 1e-6:
            median_h = 1.0
    else:
        median_h = 1.0

    relative_height = body_height / (median_h + 1e-6)

    feat13 = np.concatenate([feat12, np.array([relative_height], dtype=np.float32)])
    return feat13

def extract_keypoints(image, pose):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints, dtype=np.float32)

# -------------------------
# REALTIME MAIN LOOP
# -------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # SPEED BOOST: Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

            if bbox:
                x1,y1,x2,y2 = bbox
                h_frame, w_frame = frame.shape[:2]
                x1 = max(0, min(x1, w_frame-1))
                x2 = max(0, min(x2, w_frame-1))
                y1 = max(0, min(y1, h_frame-1))
                y2 = max(0, min(y2, h_frame-1))

                crop = frame[y1:y2, x1:x2].copy()
                kps = extract_keypoints(crop, pose) if crop.size > 0 else None

                if kps is not None:
                    try:
                        vis_ok = (
                            kps[11][3] >= MIN_LANDMARK_VIS and
                            kps[12][3] >= MIN_LANDMARK_VIS and
                            kps[23][3] >= MIN_LANDMARK_VIS and
                            kps[24][3] >= MIN_LANDMARK_VIS
                        )
                    except:
                        vis_ok = False

                    feat12, body_height, valid = compute_body_height_and_features(kps)
                    if valid and vis_ok:
                        baseline_heights.append(body_height)

                    geom = extract_geom_from_kp(kps)
                    seq_buffer.append(geom)

                    if len(seq_buffer) == SEQ_LEN:
                        X = np.array(seq_buffer, dtype=np.float32).reshape(1, SEQ_LEN, GEOM_FEATURES)

                        interpreter.set_tensor(input_details[0]['index'], X)
                        interpreter.invoke()
                        pred = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

                        prob_queue.append(pred)
                        smooth_pred = float(np.mean(prob_queue))

                        state = "fall" if smooth_pred > FALL_PROB_THRESHOLD else "no_fall"
                        fall_memory.append(state == "fall")

                        final_state = "fall" if sum(fall_memory) >= FALL_PATTERN_REQUIRED else "no_fall"

                        if final_state == last_state:
                            stable_counter += 1
                        else:
                            stable_counter = 0
                            last_state = final_state

                        displayed_state = last_state if stable_counter >= REQUIRED_STABLE else "no_fall"

                        label_txt = f"{displayed_state.upper()} (p={smooth_pred:.2f})"
                        color = (0,0,255) if displayed_state == "fall" else (0,255,0)

                        if displayed_state == "fall":
                            fname = f"fall_{int(time.time())}_{smooth_pred:.2f}.jpg"
                            cv2.imwrite(os.path.join(SAVE_DIR, fname), frame)
                            print("⚠ FALL SAVED:", fname)

                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(frame, label_txt, (x1, max(20, y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("RPI Fall Detection (TFLite)", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
