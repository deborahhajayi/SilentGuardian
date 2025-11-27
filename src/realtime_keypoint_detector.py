#!/usr/bin/env python3
"""
Realtime detector using geometric-features LSTM (updated for 13 features).

Matches training script feature order:
  0 shoulder_angle
  1 hip_angle
  2 torso_angle
  3 body_height
  4 aspect
  5 shoulders_center.x
  6 shoulders_center.y
  7 hips_center.x
  8 hips_center.y
  9 nose.x
 10 nose.y
 11 dist_sh_hip
 12 relative_height   <- NEW (body_height / running_median_height)

This runtime computes a running median baseline for "median_height" using
recent frames with reliable landmarks, so relative_height is available for inference.
"""
import os
import time
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
GEOM_FEATURES = 13   # <-- updated to 13 to match training

# smoothing & thresholds (tweakable)
REQUIRED_STABLE = 4
prob_queue = deque(maxlen=SEQ_LEN)
FALL_PROB_THRESHOLD = 0.50   # realtime decision threshold (you can lower to 0.5)
fall_memory = deque(maxlen=20)
FALL_PATTERN_REQUIRED = 8

# baseline (median) for relative height
BASELINE_WINDOW = 300   # number of recent good frames to keep (approx several seconds)
baseline_heights = deque(maxlen=BASELINE_WINDOW)

# visibility threshold to accept a frame into baseline
MIN_LANDMARK_VIS = 0.45

# -------------------------
# geometric helpers
# -------------------------
def angle(p1, p2):
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return np.arctan2(dy, dx)

def compute_body_height_and_features(kp):
    """
    Compute body_height and other geometric features from keypoints array (33,4)
    Returns (feat12_without_relative, body_height, valid_flag)
    """
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
    ], dtype=float)

    return feat12, float(body_height), True

def extract_geom_from_kp(kp):
    """
    Return full 13-feature vector. Uses running median baseline to compute relative_height.
    """
    feat12, body_height, valid = compute_body_height_and_features(kp)
    if not valid:
        return np.zeros(GEOM_FEATURES, dtype=float)

    # median baseline (fallback to 1.0 to avoid divide by zero)
    if len(baseline_heights) > 0:
        median_h = float(np.median(np.array(baseline_heights)))
        if median_h <= 1e-6:
            median_h = 1.0
    else:
        median_h = 1.0

    relative_height = body_height / (median_h + 1e-6)

    feat13 = np.concatenate([feat12, np.array([relative_height], dtype=float)])
    return feat13

def extract_keypoints(image, pose):
    """
    Run MediaPipe on the image crop and return 33x4 numpy array or None.
    Coordinates are normalized (0..1) relative to crop size (mediapipe default).
    """
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

    # For baseline stabilization: we only add frames where core landmarks are visible
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
                # clamp coordinates
                h_frame, w_frame = frame.shape[:2]
                x1 = max(0, min(x1, w_frame-1))
                x2 = max(0, min(x2, w_frame-1))
                y1 = max(0, min(y1, h_frame-1))
                y2 = max(0, min(y2, h_frame-1))

                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    kps = None
                else:
                    kps = extract_keypoints(crop, pose)

                if kps is not None:
                    # check core landmarks visibility before adding to baseline
                    try:
                        vis_L = float(kps[11][3])
                        vis_R = float(kps[12][3])
                        vis_LH = float(kps[23][3])
                        vis_RH = float(kps[24][3])
                        core_vis_ok = (vis_L >= MIN_LANDMARK_VIS and vis_R >= MIN_LANDMARK_VIS and
                                       vis_LH >= MIN_LANDMARK_VIS and vis_RH >= MIN_LANDMARK_VIS)
                    except Exception:
                        core_vis_ok = False

                    # compute feat12 and body_height
                    feat12, body_height, valid = compute_body_height_and_features(kps)
                    if valid and core_vis_ok:
                        # add to running baseline heights
                        baseline_heights.append(body_height)

                    # now build full 13-feature vector (uses baseline median, fallback handled inside)
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
                # print some debug info
                median_h = float(np.median(np.array(baseline_heights))) if len(baseline_heights)>0 else 0.0
                print(f"[DEBUG] baseline_len={len(baseline_heights)} median_h={median_h:.4f} probs={list(prob_queue)}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
