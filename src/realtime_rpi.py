#!/usr/bin/env python3

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import deque
import time
import requests
import argparse

SEQ_LEN = 8
THRESH = 0.5
CONFIRMATION_NEEDED = 3

fall_counter = 0

# --------------------
# LOAD MODELS
# --------------------

movenet = tflite.Interpreter(model_path="models/movenet_lightning.tflite")
movenet.allocate_tensors()
mn_in = movenet.get_input_details()
mn_out = movenet.get_output_details()

fall_model = tflite.Interpreter(model_path="models/fall_geom_lstm.tflite")
fall_model.allocate_tensors()
fall_in = fall_model.get_input_details()
fall_out = fall_model.get_output_details()

buf = deque(maxlen=SEQ_LEN)
height_buffer = deque(maxlen=300)

# --------------------
# GEOMETRY
# --------------------

def angle(a, b):
    return np.arctan2(b[1] - a[1], b[0] - a[0])


def geom_from_kp(kp):

    LOW_CONF = 0.15
    l_hip_conf, r_hip_conf = kp[11][2], kp[12][2]

    if l_hip_conf < LOW_CONF or r_hip_conf < LOW_CONF:
        return np.zeros(13, dtype=np.float32)

    try:

        NOSE = kp[0][:2]
        L_SHO, R_SHO = kp[5][:2], kp[6][:2]
        L_HIP, R_HIP = kp[11][:2], kp[12][:2]

        shos_mid = (L_SHO + R_SHO) / 2
        hips_mid = (L_HIP + R_HIP) / 2

        body_h = np.linalg.norm(NOSE - hips_mid)

        if body_h > 0:
            height_buffer.append(body_h)

        med_h = np.median(height_buffer) if height_buffer else 1.0
        rel_h = body_h / (med_h + 1e-6)

        vis = kp[kp[:, 2] > LOW_CONF]

        if len(vis) > 2:
            xs, ys = vis[:, 0], vis[:, 1]
            aspect = (ys.max() - ys.min() + 1e-6) / (xs.max() - xs.min() + 1e-6)
        else:
            aspect = 0.5

        return np.array([
            angle(L_SHO, R_SHO),
            angle(L_HIP, R_HIP),
            angle(shos_mid, hips_mid),
            body_h,
            aspect,
            shos_mid[0], shos_mid[1],
            hips_mid[0], hips_mid[1],
            NOSE[0], NOSE[1],
            np.linalg.norm(shos_mid - hips_mid),
            rel_h
        ], dtype=np.float32)

    except:
        return np.zeros(13, dtype=np.float32)


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

    for y, x, score in kp:

        if score < 0.1:
            continue

        px, py = map_point(y, x)
        cv2.circle(frame, (px, py), 4, (0,255,0), -1)

    for a, b in CONNECTIONS:

        if kp[a][2] < 0.1 or kp[b][2] < 0.1:
            continue

        ax, ay = map_point(kp[a][0], kp[a][1])
        bx, by = map_point(kp[b][0], kp[b][1])

        cv2.line(frame, (ax, ay), (bx, by), (255,0,0), 2)


# --------------------
# ARGUMENTS
# --------------------

parser = argparse.ArgumentParser()
parser.add_argument("--email", default="")
parser.add_argument("--location", default="Camera 1")
parser.add_argument("--display", action="store_true")

args = parser.parse_args()

EMAIL = args.email
LOCATION = args.location
DISPLAY = args.display


# --------------------
# CAMERA
# --------------------

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

API_URL = "http://127.0.0.1:5000/api/report_fall"
COOLDOWN_SEC = 10
last_sent = 0

print("Camera started")

# --------------------
# REALTIME LOOP
# --------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera read failed")
        break

    h, w, _ = frame.shape

    scale = 192 / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    img = cv2.resize(frame, (nw, nh))

    padded = np.zeros((192,192,3), dtype=np.float32)
    padded[(192-nh)//2:(192-nh)//2+nh, (192-nw)//2:(192-nw)//2+nw] = img

    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)[None,...]

    movenet.set_tensor(mn_in[0]["index"], padded_rgb)
    movenet.invoke()

    kp = movenet.get_tensor(mn_out[0]["index"])[0,0]

    feat = geom_from_kp(kp)

    if np.any(feat != 0):
        buf.append(feat)

    label = "STABLE"
    color = (0,255,0)

    if len(buf) == SEQ_LEN:

        input_data = np.array(buf)[None,...].astype(np.float32)

        fall_model.set_tensor(fall_in[0]["index"], input_data)
        fall_model.invoke()

        p = float(fall_model.get_tensor(fall_out[0]["index"])[0][0])

        if p > THRESH:
            fall_counter += 1
        else:
            fall_counter = max(0, fall_counter - 1)

        if fall_counter >= CONFIRMATION_NEEDED:

            label = "FALL CONFIRMED"
            color = (0,0,255)

            if (time.time() - last_sent) > COOLDOWN_SEC:

                last_sent = time.time()

                ok, jpg = cv2.imencode(".jpg", frame)

                if ok:

                    files = {"image": ("fall.jpg", jpg.tobytes(), "image/jpeg")}
                    data = {"email": EMAIL, "location": LOCATION}

                    try:
                        r = requests.post(API_URL, data=data, files=files, timeout=5)
                        print("Alert sent:", r.status_code)

                    except Exception as e:
                        print("API error:", e)

        elif fall_counter > 0:

            label = f"ANALYZING ({fall_counter})"
            color = (0,165,255)

    draw_skeleton(frame, kp, (192-nh)//2, (192-nw)//2, scale)

    if DISPLAY:

        cv2.putText(frame,label,(20,40),
        cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)

        cv2.imshow("MoveNet Fall Detection", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break


cap.release()
cv2.destroyAllWindows()