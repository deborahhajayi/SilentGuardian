#!/usr/bin/env python3
import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO

# =============================
# CONFIG
# =============================
RAW_DIR = "data/hand_raw"
OUT_DIR = "data/keypoints"

MODEL_PATH = "models/yolov8_hand_pose.pt"

CLASSES = {
    "train_val_like": 0,
    "train_val_palm": 1,
    "train_val_neutral": 2
}

os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# LOAD YOLOv8 POSE
# =============================
print("Loading YOLOv8 pose model...")
model = YOLO(MODEL_PATH)

# =============================
# KEYPOINT EXTRACTION
# =============================
def extract_keypoints(image):
    """
    Returns:
        (N,2) array of keypoints
        or None if detection fails
    """

    results = model(image, verbose=False)

    if len(results) == 0:
        return None

    r = results[0]

    if r.keypoints is None:
        return None

    # take first detected hand
    kp = r.keypoints.xy

    if kp is None or len(kp) == 0:
        return None

    kp = kp[0].cpu().numpy().astype(np.float32)  # (N,2)

    return kp


# =============================
# PROCESS ONE CLASS FOLDER
# =============================
def process_folder(folder_name, label):

    folder_path = os.path.join(RAW_DIR, folder_name)
    out_csv = os.path.join(OUT_DIR, f"{folder_name}.csv")

    if not os.path.isdir(folder_path):
        print("⚠ Missing folder:", folder_path)
        return

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n📁 Processing {folder_name} ({len(images)} images)")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # dynamic header after first successful detection
        header_written = False
        written = 0

        for img_name in images:

            path = os.path.join(folder_path, img_name)
            img = cv2.imread(path)

            if img is None:
                continue

            kp = extract_keypoints(img)

            if kp is None:
                continue

            flat = kp.reshape(-1).tolist()

            if not header_written:
                writer.writerow(
                    ["image_id", "label"] +
                    [f"kp_{i}" for i in range(len(flat))]
                )
                header_written = True

            writer.writerow(
                [os.path.splitext(img_name)[0], label] + flat
            )

            written += 1

    print("   ✅ Rows written:", written)
    print("   Saved:", out_csv)


# =============================
# MAIN
# =============================
def main():
    for cls, label in CLASSES.items():
        process_folder(cls, label)

    print("\n🎉 Keypoint extraction complete.")


if __name__ == "__main__":
    main()
