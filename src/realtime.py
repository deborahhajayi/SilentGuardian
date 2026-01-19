#!/usr/bin/env python3
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import load_model
import time
import requests
import argparse


SEQ_LEN = 8
THRESH = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--email", required=True)
parser.add_argument("--location", default="Unknown")
parser.add_argument("--api", default="http://127.0.0.1:5000/api/report_fall")
args = parser.parse_args()

COOLDOWN_SEC = 10
last_sent = 0


# --------------------
# LOAD MODELS
# --------------------
movenet = tf.lite.Interpreter(model_path="models/movenet_lightning.tflite")
movenet.allocate_tensors()
mn_in = movenet.get_input_details()
mn_out = movenet.get_output_details()

model = load_model("models/fall_geom_lstm.keras")

buf = deque(maxlen=SEQ_LEN)
height_buffer = deque(maxlen=300)

# --------------------
# GEOMETRY
# --------------------
def angle(a, b):
    return np.arctan2(b[1] - a[1], b[0] - a[0])

def geom_from_kp(kp):
    NOSE = kp[0][:2]
    L_SHO, R_SHO = kp[5][:2], kp[6][:2]
    L_HIP, R_HIP = kp[11][:2], kp[12][:2]

    shoulders = (L_SHO + R_SHO) / 2
    hips = (L_HIP + R_HIP) / 2

    body_height = np.linalg.norm(NOSE - hips)
    if body_height > 0:
        height_buffer.append(body_height)

    median_h = np.median(height_buffer) if height_buffer else 1.0
    rel_h = body_height / (median_h + 1e-6)

    xs, ys = kp[:, 0], kp[:, 1]
    aspect = (ys.max() - ys.min() + 1e-6) / (xs.max() - xs.min() + 1e-6)

    return np.array([
        angle(L_SHO, R_SHO),
        angle(L_HIP, R_HIP),
        angle(shoulders, hips),
        body_height,
        aspect,
        shoulders[0], shoulders[1],
        hips[0], hips[1],
        NOSE[0], NOSE[1],
        np.linalg.norm(shoulders - hips),
        rel_h
    ], dtype=np.float32)

# --------------------
# SKELETON DRAWING
# --------------------
CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),(11,12),
    (5,11),(6,12),(11,13),(13,15),
    (12,14),(14,16)
]

def draw_skeleton(frame, kp, pad_top, pad_left, scale):
    h, w, _ = frame.shape

    def map_point(y, x):
        px = (x * 192 - pad_left) / scale
        py = (y * 192 - pad_top) / scale
        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))
        return px, py

    # Draw joints
    for y, x, score in kp:
        if score < 0.1:
            continue
        px, py = map_point(y, x)
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    # Draw bones
    for a, b in CONNECTIONS:
        if kp[a][2] < 0.1 or kp[b][2] < 0.1:
            continue

        ax, ay = map_point(kp[a][0], kp[a][1])
        bx, by = map_point(kp[b][0], kp[b][1])
        cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)


# --------------------
# REALTIME LOOP
# --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Resize while keeping aspect ratio, pad to 192x192
    in_size = 192
    scale = in_size / max(h, w)
    nh, nw = int(h*scale), int(w*scale)

    img = cv2.resize(frame, (nw, nh))
    pad_top = (in_size - nh)//2
    pad_left = (in_size - nw)//2
    img_padded = np.zeros((in_size, in_size, 3), dtype=np.float32)
    img_padded[pad_top:pad_top+nh, pad_left:pad_left+nw] = img
    img_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)[None,...]

    # Run MoveNet
    movenet.set_tensor(mn_in[0]["index"], img_padded)
    movenet.invoke()
    kp = movenet.get_tensor(mn_out[0]["index"])[0,0]  # 17x3

    # Append geom features for LSTM
    buf.append(geom_from_kp(kp))

    # Draw skeleton properly
    draw_skeleton(frame, kp, pad_top, pad_left, scale)

    # LSTM prediction
    label = "WAIT"
    color = (0,255,255)
    if len(buf) == SEQ_LEN:
        X = np.array(buf)[None,...]
        p = float(model.predict(X, verbose=0)[0][0])
        label = "FALL" if p > THRESH else "NO_FALL"
        color = (0,0,255) if label=="FALL" else (0,255,0)
            # Send notification + screenshot (only when FALL, with cooldown)
        now = time.time()
        if label == "FALL" and (now - last_sent) > COOLDOWN_SEC:
            ok, jpg = cv2.imencode(".jpg", frame)
            if ok:
                files = {"image": ("fall.jpg", jpg.tobytes(), "image/jpeg")}
                data = {"email": args.email, "location": args.location}
                try:
                    r = requests.post(args.api, data=data, files=files, timeout=3)
                    if r.ok:
                        last_sent = now
                    else:
                        print("report_fall failed:", r.status_code, r.text)
                except Exception as e:
                    print("report_fall exception:", e)


    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("MoveNet Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
