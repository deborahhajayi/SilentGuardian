#!/usr/bin/env python3
import time
import os
import argparse
from collections import deque
from threading import Thread

import cv2
import numpy as np
import requests
from flask import Flask, Response
from picamera2 import Picamera2

# --- UNIVERSAL DEPENDENCY BLOCK ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        print("❌ Error: Dependencies missing. Install 'tflite-runtime' (Pi) or 'tensorflow' (Windows)")
        raise SystemExit(1)

# --------------------
# SETTINGS
# --------------------
SEQ_LEN = 8
THRESH = 0.5
CONFIRMATION_NEEDED = 3
COOLDOWN_SEC = 10

# --------------------
# MODEL INITIALIZATION
# --------------------
def make_interp(path: str):
    interp = tflite.Interpreter(path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()

# --------------------
# GEOMETRY FUNCTIONS
# --------------------
height_buffer = deque(maxlen=300)

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

    except Exception:
        return np.zeros(13, dtype=np.float32)

# --------------------
# SKELETON DRAWING
# --------------------
CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),(11,12),(5,11),(6,12),(11,13),
    (13,15),(12,14),(14,16)
]

def draw_skeleton(frame, kp, pad_top, pad_left, scale):
    h, w, _ = frame.shape

    def map_point(y, x):
        px = int(np.clip((x * 192 - pad_left) / scale, 0, w - 1))
        py = int(np.clip((y * 192 - pad_top) / scale, 0, h - 1))
        return px, py

    for y, x, score in kp:
        if score > 0.1:
            px, py = map_point(y, x)
            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    for a, b in CONNECTIONS:
        if kp[a][2] > 0.1 and kp[b][2] > 0.1:
            ax, ay = map_point(kp[a][0], kp[a][1])
            bx, by = map_point(kp[b][0], kp[b][1])
            cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

# --------------------
# BACKEND COMMUNICATION
# --------------------
def send_fall_to_backend(api_base, email, location, image_path):
    if not api_base:
        return

    url = api_base.rstrip("/") + "/api/report_fall"
    data = {"email": email}
    if location:
        data["location"] = location

    files = {}
    try:
        if image_path and os.path.exists(image_path):
            files["image"] = open(image_path, "rb")

        r = requests.post(url, data=data, files=files, timeout=5)
        print(f"POST -> {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"POST failed: {e}")
    finally:
        if "image" in files:
            files["image"].close()

# --------------------
# OPTIONAL LOCAL STREAM
# --------------------
app = Flask(__name__)
frame_to_stream = None

def gen_frames():
    global frame_to_stream
    while True:
        if frame_to_stream is None:
            time.sleep(0.02)
            continue

        ok, buffer = cv2.imencode(".jpg", frame_to_stream)
        if not ok:
            time.sleep(0.02)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --------------------
# MAIN
# --------------------
def main():
    global frame_to_stream

    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="User email for backend")
    parser.add_argument("--location", default="Living Room", help="Location label")
    parser.add_argument("--api_base", default="", help="Example: http://192.168.137.1:5000")

    parser.add_argument("--save_dir", default="captures")
    parser.add_argument("--movenet", default="models/movenet_lightning.tflite")
    parser.add_argument("--lstm", default="models/fall_geom_lstm_builtin.tflite")

    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stream_host", default="0.0.0.0")
    parser.add_argument("--stream_port", type=int, default=8000)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load models
    mn_interp, mn_in, mn_out = make_interp(args.movenet)
    lstm_interp, lstm_in, lstm_out = make_interp(args.lstm)

    buf = deque(maxlen=SEQ_LEN)
    fall_counter = 0
    last_sent = 0.0
    last_status_print = 0.0
    last_label = None

    # Pi camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # optional browser stream
    Thread(
        target=lambda: app.run(
            host=args.stream_host,
            port=args.stream_port,
            debug=False,
            use_reloader=False
        ),
        daemon=True
    ).start()

    print("✅ Running. Press Ctrl+C to stop.")
    print(f"📷 Stream: http://{args.stream_host}:{args.stream_port}/video_feed")
    print(f"💾 Saving snapshots to: {args.save_dir}/")
    if args.api_base:
        print(f"📡 Backend: {args.api_base}/api/report_fall")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # MoveNet pre-processing
            h, w, _ = frame.shape
            scale = 192 / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            pt, pl = (192 - nh) // 2, (192 - nw) // 2

            img = cv2.resize(frame, (nw, nh))
            padded = np.zeros((192, 192, 3), dtype=np.float32)
            padded[pt:pt + nh, pl:pl + nw] = img
            padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)[None, ...]

            # MoveNet inference
            mn_interp.set_tensor(mn_in[0]["index"], padded_rgb)
            mn_interp.invoke()
            kp = mn_interp.get_tensor(mn_out[0]["index"])[0, 0]

            # Feature engineering
            feat = geom_from_kp(kp)
            label = "STABLE"
            color = (0, 255, 0)
            p = 0.0

            if np.any(feat != 0):
                buf.append(feat)
            else:
                label = "STABLE (OUT OF RANGE)"
                color = (0, 165, 255)
                fall_counter = 0
                buf.clear()

            # LSTM inference
            if len(buf) == SEQ_LEN:
                input_data = np.array(buf, dtype=np.float32)[None, ...]
                lstm_interp.set_tensor(lstm_in[0]["index"], input_data)
                lstm_interp.invoke()
                p = float(lstm_interp.get_tensor(lstm_out[0]["index"])[0][0])

                if p > THRESH:
                    fall_counter += 1
                else:
                    if fall_counter > 0:
                        fall_counter = 0
                        buf.clear()

                # Confirmation & alerting
                if fall_counter >= CONFIRMATION_NEEDED:
                    label = "FALL CONFIRMED"
                    color = (0, 0, 255)

                    if (time.time() - last_sent) > COOLDOWN_SEC:
                        last_sent = time.time()

                        ts = time.strftime("%Y%m%d_%H%M%S")
                        out_path = os.path.join(args.save_dir, f"fall_{ts}.jpg")
                        cv2.imwrite(out_path, frame)

                        print(f"FALL DETECTED (p={p:.3f}) saved: {out_path}")

                        send_fall_to_backend(
                            args.api_base,
                            args.email,
                            args.location,
                            out_path
                        )

                elif fall_counter > 0:
                    label = f"ANALYZING ({fall_counter})"
                    color = (0, 165, 255)

            # Visuals
            draw_skeleton(frame, kp, pt, pl, scale)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            frame_to_stream = frame.copy()

            now = time.time()
            if (label != last_label) or (now - last_status_print >= 1.0):
                print(f"Status: {label} (p={p:.3f})")
                last_status_print = now
                last_label = label

            time.sleep(max(0.0, 1.0 / args.fps))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()