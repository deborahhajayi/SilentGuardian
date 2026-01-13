#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque
from tflite_runtime.interpreter import Interpreter

SEQ_LEN = 8
THRESH = 0.5
IN_SIZE = 192

# --------------------
# LOAD TFLITE MODELS
# --------------------
movenet = Interpreter(
    model_path="models/movenet_lightning.tflite",
    num_threads=2
)
movenet.allocate_tensors()
mn_in = movenet.get_input_details()
mn_out = movenet.get_output_details()

fall_lstm = Interpreter(
    model_path="models/fall_geom_lstm.tflite",
    num_threads=2
)
fall_lstm.allocate_tensors()
fl_in = fall_lstm.get_input_details()
fl_out = fall_lstm.get_output_details()

buf = deque(maxlen=SEQ_LEN)
height_buffer = deque(maxlen=300)

# --------------------
# GEOMETRY FEATURES
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

    xs, ys = kp[:, 1], kp[:, 0]
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
        px = int((x * IN_SIZE - pad_left) / scale)
        py = int((y * IN_SIZE - pad_top) / scale)
        return np.clip(px, 0, w-1), np.clip(py, 0, h-1)

    for y, x, s in kp:
        if s < 0.2:
            continue
        px, py = map_point(y, x)
        cv2.circle(frame, (px, py), 4, (0,255,0), -1)

    for a, b in CONNECTIONS:
        if kp[a][2] < 0.2 or kp[b][2] < 0.2:
            continue
        ax, ay = map_point(kp[a][0], kp[a][1])
        bx, by = map_point(kp[b][0], kp[b][1])
        cv2.line(frame, (ax, ay), (bx, by), (255,0,0), 2)

# --------------------
# REALTIME LOOP
# --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    scale = IN_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))

    pad_top = (IN_SIZE - nh) // 2
    pad_left = (IN_SIZE - nw) // 2

    img = np.zeros((IN_SIZE, IN_SIZE, 3), dtype=np.uint8)
    img[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)[None, ...]

    movenet.set_tensor(mn_in[0]["index"], img)
    movenet.invoke()
    kp = movenet.get_tensor(mn_out[0]["index"])[0,0]

    buf.append(geom_from_kp(kp))
    draw_skeleton(frame, kp, pad_top, pad_left, scale)

    label = "WAIT"
    color = (0,255,255)

    if len(buf) == SEQ_LEN:
        X = np.array(buf, dtype=np.float32)[None, ...]
        fall_lstm.set_tensor(fl_in[0]["index"], X)
        fall_lstm.invoke()
        p = float(fall_lstm.get_tensor(fl_out[0]["index"])[0][0])

        label = "FALL" if p > THRESH else "NO_FALL"
        color = (0,0,255) if label == "FALL" else (0,255,0)

    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Pi MoveNet Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
