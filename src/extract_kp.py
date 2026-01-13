#!/usr/bin/env python3
import os
import cv2
import csv
import numpy as np
import tensorflow as tf
import re

# =============================
# CONFIG
# =============================
RAW_DIR = "data/raw"
OUT_DIR = "data/keypoints"
MODEL_PATH = "models/movenet_lightning.tflite"
FALL_FRACTION = 0.50

os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# Load MoveNet
# =============================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()

def movenet(frame):
    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)[None, ...]
    interpreter.set_tensor(inp[0]["index"], img)
    interpreter.invoke()
    kp = interpreter.get_tensor(out[0]["index"])[0, 0]  # (17,3)
    return kp.astype(np.float32)

# =============================
# Annotation helpers
# =============================
def parse_annotation(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        ints = [int(l.strip()) for l in f if l.strip().isdigit()]
    if len(ints) < 2:
        return None, None
    return ints[0], ints[1]

def find_annotation(ann_dir, vid):
    base = os.path.splitext(vid)[0].lower()
    for f in os.listdir(ann_dir):
        if f.lower().startswith(base):
            return os.path.join(ann_dir, f)
    digits = re.findall(r"\d+", base)
    for d in digits:
        for f in os.listdir(ann_dir):
            if d in f:
                return os.path.join(ann_dir, f)
    return None

# =============================
# Extraction
# =============================
def extract(video_path, video_id, fs, fe, writer):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠ Cannot open", video_path)
        return 0

    visible_start = fs + int((fe - fs) * FALL_FRACTION)
    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if fs == 0 and fe == 0:
            label = 0
        elif frame_idx < fs:
            label = 0
        elif fs <= frame_idx < visible_start:
            continue
        elif visible_start <= frame_idx <= fe:
            label = 1
        else:
            continue

        kp = movenet(frame)
        if kp.shape != (17, 3):
            continue  # hard safety

        writer.writerow(
            [video_id, frame_idx, label] +
            kp.reshape(-1).tolist()   # EXACTLY 51 VALUES
        )
        written += 1

    cap.release()
    return written

# =============================
# Main
# =============================
def main():
    out_csv = os.path.join(OUT_DIR, "all_keypoints.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["video_id", "frame_idx", "label"] +
            [f"kp_{i}" for i in range(51)]
        )

        total_rows = 0
        total_videos = 0

        for scene in os.listdir(RAW_DIR):
            scene_path = os.path.join(RAW_DIR, scene)
            if not os.path.isdir(scene_path):
                continue

            # handle nested dataset folder
            inside = os.listdir(scene_path)
            if len(inside) == 1 and os.path.isdir(os.path.join(scene_path, inside[0])):
                scene_path = os.path.join(scene_path, inside[0])

            video_dir = os.path.join(scene_path, "Videos")
            ann_dir = os.path.join(scene_path, "Annotation_files")

            if not os.path.isdir(video_dir) or not os.path.isdir(ann_dir):
                continue

            videos = [v for v in os.listdir(video_dir)
                      if v.lower().endswith((".mp4", ".avi"))]

            for vid in videos:
                ann = find_annotation(ann_dir, vid)
                if not ann:
                    continue

                fs, fe = parse_annotation(ann)
                if fs is None:
                    continue

                rows = extract(
                    os.path.join(video_dir, vid),
                    os.path.splitext(vid)[0],
                    fs, fe,
                    writer
                )

                if rows > 0:
                    total_videos += 1
                total_rows += rows

        print("\n✅ MoveNet extraction finished")
        print("   Videos used:", total_videos)
        print("   Total rows written:", total_rows)
        print("   Saved to:", out_csv)

if __name__ == "__main__":
    main()
