#!/usr/bin/env python3

import os
import cv2
import time
import math
import joblib
import signal
import threading
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from picamera2 import Picamera2

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# =============================
# CONFIG
# =============================
MODEL_PATH = "models/hand_gesture_mlp.tflite"
SCALER_PATH = "models/hand_scaler.pkl"
YOLO_MODEL_PATH = "models/yolov8_hand_pose.onnx"

THRESH = 0.6
CLASS_NAMES = ["like", "palm", "neutral"]

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_WARMUP_SECONDS = 2
YOLO_IMGSZ = 224
YOLO_CONF = 0.05

CAPTURE_DIR = "gesture_captures"
SAVE_COOLDOWN_SECONDS = 2.0
SAVE_LATEST_EVERY_N_FRAMES = 10
JPEG_QUALITY = 80

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

PRINT_SUMMARY_EVERY_N_FRAMES = 60

# =============================
# GLOBALS
# =============================
app = Flask(__name__)
stream_lock = threading.Lock()
latest_jpeg = None
latest_status = "WAIT - starting"
stop_event = threading.Event()

picam2 = None
scaler = None
yolo_model = None
interpreter = None
input_details = None
output_details = None

stats = {
    "frames": 0,
    "capture_fail": 0,
    "no_hand": 0,
    "hands_found": 0,
    "feature_ok": 0,
    "inference_ok": 0,
    "gesture_detected": 0,
    "stream_frames_encoded": 0,
    "errors": 0,
}

# =============================
# LOGGING HELPERS
# =============================
def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


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
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]

    for f in fingers:
        features.append(angle(kp[f[0]], kp[f[1]], kp[f[2]]))
        features.append(angle(kp[f[1]], kp[f[2]], kp[f[3]]))

    tips = [4, 8, 12, 16, 20]

    for t in tips:
        features.append(np.linalg.norm(kp[t]))

    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            features.append(np.linalg.norm(kp[tips[i]] - kp[tips[j]]))

    features.append(math.atan2(kp[9][1], kp[9][0]))

    return np.array(features, dtype=np.float32)


# =============================
# HELPERS
# =============================
def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def save_detected_frame(frame_bgr, gesture_label, confidence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{gesture_label}_{confidence:.2f}_{timestamp}.jpg"
    filepath = os.path.join(CAPTURE_DIR, filename)
    cv2.imwrite(filepath, frame_bgr)


def update_stream_frame(frame_bgr):
    global latest_jpeg

    ok, buffer = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        return

    with stream_lock:
        latest_jpeg = buffer.tobytes()

    stats["stream_frames_encoded"] += 1


def extract_keypoints(frame_bgr, yolo_model):
    results = yolo_model(
        frame_bgr,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        verbose=False,
        device="cpu",
    )

    if len(results) == 0:
        return None, "yolo_results=0"

    r = results[0]

    box_count = 0
    if r.boxes is not None:
        try:
            box_count = len(r.boxes)
        except Exception:
            box_count = 0

    if r.keypoints is None:
        return None, f"boxes={box_count}, keypoints=None"

    kp = r.keypoints.xy
    if kp is None or len(kp) == 0:
        return None, f"boxes={box_count}, keypoints=empty"

    kp = kp[0].cpu().numpy()
    kp_flat = kp[:, :2].flatten()
    return kp_flat, f"boxes={box_count}, keypoints={kp.shape[0]}"


def annotate_frame(frame_bgr, text, good=True):
    annotated = frame_bgr.copy()
    color = (0, 255, 0) if good else (0, 255, 255)
    cv2.putText(
        annotated,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


# =============================
# MODEL / CAMERA SETUP
# =============================
def setup():
    global scaler, yolo_model, interpreter, input_details, output_details, picam2

    log("===== HAND GESTURE MJPEG SCRIPT STARTING =====")
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    require_file(MODEL_PATH)
    require_file(SCALER_PATH)
    require_file(YOLO_MODEL_PATH)

    log("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)

    log("Loading YOLO hand-pose model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    log(f"YOLO task: {getattr(yolo_model, 'task', 'unknown')} | imgsz={YOLO_IMGSZ} | conf={YOLO_CONF}")

    log("Loading TFLite gesture classifier...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    log(f"TFLite input shape: {input_details[0]['shape']} | output shape: {output_details[0]['shape']}")

    log("Starting Picamera2...")
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(CAMERA_WARMUP_SECONDS)
    log(f"Camera ready at {FRAME_WIDTH}x{FRAME_HEIGHT}")


# =============================
# DETECTION LOOP
# =============================
def detection_loop():
    global latest_status

    frame_id = 0
    last_saved_time = 0.0
    last_printed_status = None
    last_detail_message = None

    log("Background detection loop started")

    while not stop_event.is_set():
        frame_id += 1
        stats["frames"] += 1

        try:
            frame_rgb = picam2.capture_array()
        except Exception as e:
            stats["capture_fail"] += 1
            stats["errors"] += 1
            latest_status = f"CAPTURE ERROR: {e}"
            if latest_status != last_printed_status:
                log(latest_status)
                last_printed_status = latest_status
            time.sleep(0.1)
            continue

        if frame_rgb is None:
            stats["capture_fail"] += 1
            latest_status = "WAIT - capture returned None"
            if latest_status != last_printed_status:
                log(latest_status)
                last_printed_status = latest_status
            time.sleep(0.05)
            continue

        try:
            if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            elif len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2BGR)
            else:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            stats["errors"] += 1
            latest_status = f"COLOR ERROR: {e}"
            if latest_status != last_printed_status:
                log(latest_status)
                last_printed_status = latest_status
            time.sleep(0.1)
            continue

        try:
            kp_flat, detect_info = extract_keypoints(frame_bgr, yolo_model)
        except Exception as e:
            stats["errors"] += 1
            latest_status = f"YOLO ERROR: {e}"
            if latest_status != last_printed_status:
                log(latest_status)
                last_printed_status = latest_status
            time.sleep(0.1)
            continue

        if kp_flat is None:
            stats["no_hand"] += 1
            latest_status = "WAIT - no hand"
            annotated = annotate_frame(frame_bgr, latest_status, good=False)
            update_stream_frame(annotated)

            if frame_id % SAVE_LATEST_EVERY_N_FRAMES == 0:
                cv2.imwrite("output_frame.jpg", annotated)

            if latest_status != last_printed_status:
                log(latest_status)
                last_printed_status = latest_status

            if detect_info != last_detail_message:
                log(f"Detection check: {detect_info}")
                last_detail_message = detect_info

            if frame_id % PRINT_SUMMARY_EVERY_N_FRAMES == 0:
                log(f"Summary: {stats}")

            continue

        stats["hands_found"] += 1
        if detect_info != last_detail_message:
            log(f"Detection check: {detect_info}")
            last_detail_message = detect_info

        try:
            features = extract_features_from_flat(kp_flat)
            stats["feature_ok"] += 1
        except Exception as e:
            stats["errors"] += 1
            latest_status = f"FEATURE ERROR: {e}"
            if latest_status != last_printed_status:
                log(latest_status)
                last_printed_status = latest_status
            time.sleep(0.1)
            continue

        try:
            X = scaler.transform([features]).astype(np.float32)
        except Exception as e:
            stats["errors"] += 1
            latest_status = f"SCALER ERROR: {e}"
            if latest_status != last_printed_status:
                log(latest_status)
                last_printed_status = latest_status
            time.sleep(0.1)
            continue

        try:
            interpreter.set_tensor(input_details[0]["index"], X)
            interpreter.invoke()
            prob = interpreter.get_tensor(output_details[0]["index"])[0]
            stats["inference_ok"] += 1
        except Exception as e:
            stats["errors"] += 1
            latest_status = f"TFLITE ERROR: {e}"
            if latest_status != last_printed_status:
                log(latest_status)
                last_printed_status = latest_status
            time.sleep(0.1)
            continue

        pred_idx = int(np.argmax(prob))
        confidence = float(np.max(prob))
        pred_name = CLASS_NAMES[pred_idx]

        if confidence > THRESH and pred_name != "neutral":
            latest_status = f"{pred_name} ({confidence:.2f})"
            stats["gesture_detected"] += 1
            good = True
        else:
            latest_status = f"WAIT / neutral ({pred_name}, {confidence:.2f})"
            good = False

        annotated = annotate_frame(frame_bgr, latest_status, good=good)
        update_stream_frame(annotated)

        if frame_id % SAVE_LATEST_EVERY_N_FRAMES == 0:
            cv2.imwrite("output_frame.jpg", annotated)

        now = time.time()
        if good and (now - last_saved_time) >= SAVE_COOLDOWN_SECONDS:
            save_detected_frame(annotated, pred_name, confidence)
            last_saved_time = now

        if latest_status != last_printed_status:
            log(latest_status)
            last_printed_status = latest_status

        if frame_id % PRINT_SUMMARY_EVERY_N_FRAMES == 0:
            log(f"Summary: {stats}")

    log("Background detection loop stopped")


# =============================
# FLASK ROUTES
# =============================
INDEX_HTML = '''
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pi Hand Gesture Stream</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; }
        .card { background: white; padding: 20px; border-radius: 12px; max-width: 900px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        img { width: 100%; max-width: 800px; border-radius: 8px; border: 1px solid #ddd; }
        code { background: #eee; padding: 2px 6px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Pi Hand Gesture MJPEG Stream</h1>
        <p>Status: <strong>{{ status }}</strong></p>
        <p>Video route: <code>/video_feed</code></p>
        <img src="/video_feed" alt="Live MJPEG stream">
    </div>
</body>
</html>
'''


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, status=latest_status)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    return {"status": latest_status, "stats": stats}


# =============================
# MJPEG GENERATOR
# =============================
def generate_mjpeg():
    while not stop_event.is_set():
        with stream_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.03)


# =============================
# SHUTDOWN
# =============================
def shutdown_handler(signum, frame):
    log(f"Received signal {signum}, shutting down...")
    stop_event.set()
    try:
        if picam2 is not None:
            picam2.stop()
    except Exception:
        pass


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


if __name__ == "__main__":
    setup()

    worker = threading.Thread(target=detection_loop, daemon=True)
    worker.start()

    log(f"Open browser at: http://<PI_IP>:{FLASK_PORT}/")
    log(f"Direct MJPEG stream: http://<PI_IP>:{FLASK_PORT}/video_feed")

    try:
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            threaded=True,
            debug=False,
            use_reloader=False,
        )
    finally:
        stop_event.set()
        try:
            if picam2 is not None:
                picam2.stop()
        except Exception:
            pass
        log(f"Final stats: {stats}")
        log("Camera released. Script ended.")