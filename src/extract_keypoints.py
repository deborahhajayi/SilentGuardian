# src/extract_keypoints.py
#!/usr/bin/env python3
import os
import cv2
import mediapipe as mp
import csv
from tqdm import tqdm
import re

RAW_DIR = "data/raw"
OUT_DIR = "data/keypoints"
FALL_FRACTION = 0.60  # visible fall starts at fall_start + 0.40*(fall_end - fall_start)
os.makedirs(OUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose


def parse_annotation(path):
    """Read fall start/end from annotation file."""
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    ints = [int(s) for s in lines if s.isdigit()]
    if len(ints) < 2:
        return None, None

    fall_start, fall_end = ints[:2]
    if fall_start > fall_end:
        return None, None

    return fall_start, fall_end


def find_annotation(ann_dir, video_filename):
    """Match annotation file to video by prefix or numeric ID."""
    base = os.path.splitext(video_filename)[0].lower()

    for f in os.listdir(ann_dir):
        if f.lower().endswith(".txt") and f.lower().startswith(base):
            return os.path.join(ann_dir, f)

    digits = re.findall(r"\d+", base)
    if digits:
        for f in os.listdir(ann_dir):
            if digits[0] in f:
                return os.path.join(ann_dir, f)

    return None


def extract_keypoints_from_video(video_path, video_id, fall_start, fall_end,
                                 writer_all, writer_fall, writer_no):
    """Extract keypoints frame-by-frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    visible_start = fall_start + int((fall_end - fall_start) * FALL_FRACTION)

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pbar.update(1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if not results.pose_landmarks:
                continue

            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            # Labeling rules
            if fall_start == 0 and fall_end == 0:
                label = 0
            elif frame_idx < fall_start:
                label = 0
            elif fall_start <= frame_idx < visible_start:
                label = None
            elif visible_start <= frame_idx <= fall_end:
                label = 1
            else:
                label = None

            if label is None:
                continue

            row = [video_id, frame_idx, label] + keypoints
            writer_all.writerow(row)
            if label == 1:
                writer_fall.writerow(row[3:])
            else:
                writer_no.writerow(row[3:])

        pbar.close()

    cap.release()


def main():
    print("\n🔍 Scanning raw dataset structure...")
    print("Base raw directory:", RAW_DIR)

    # Output CSVs
    all_csv_path = os.path.join(OUT_DIR, "all_keypoints.csv")
    fall_csv_path = os.path.join(OUT_DIR, "fall_keypoints.csv")
    no_csv_path = os.path.join(OUT_DIR, "no_fall_keypoints.csv")

    all_csv = open(all_csv_path, "w", newline="")
    fall_csv = open(fall_csv_path, "w", newline="")
    no_csv = open(no_csv_path, "w", newline="")
    w_all = csv.writer(all_csv)
    w_fall = csv.writer(fall_csv)
    w_no = csv.writer(no_csv)

    # Header for main CSV
    header = ["video_id", "frame_idx", "label"] + [f"kp_{i}" for i in range(33 * 4)]
    w_all.writerow(header)

    # -----------------------------
    # NEW: Traverse all subfolders
    # -----------------------------
    subfolders = [os.path.join(RAW_DIR, d) for d in os.listdir(RAW_DIR)
                  if os.path.isdir(os.path.join(RAW_DIR, d))]

    total_videos_found = 0

    for scene_folder in subfolders:

        # Look for nested folder (Coffee_room_01/Coffee_room_01/...)
        inside = os.listdir(scene_folder)
        if len(inside) == 1 and os.path.isdir(os.path.join(scene_folder, inside[0])):
            scene_folder = os.path.join(scene_folder, inside[0])

        video_dir = os.path.join(scene_folder, "Videos")
        ann_dir = os.path.join(scene_folder, "Annotation_files")

        if not os.path.isdir(video_dir) or not os.path.isdir(ann_dir):
            print("Skipping invalid structure:", scene_folder)
            continue

        print("\n📂 Using:")
        print(" Videos:", video_dir)
        print(" Annotations:", ann_dir)

        videos = [v for v in os.listdir(video_dir)
                  if v.lower().endswith((".mp4", ".avi"))]

        total_videos_found += len(videos)

        for vid in videos:
            vpath = os.path.join(video_dir, vid)
            ann_path = find_annotation(ann_dir, vid)

            if ann_path is None:
                print("⚠ Skipping (no annotation):", vid)
                continue

            fall_start, fall_end = parse_annotation(ann_path)
            if fall_start is None:
                print("⚠ Skipping invalid annotation:", ann_path)
                continue

            video_id = os.path.splitext(vid)[0]
            extract_keypoints_from_video(
                vpath, video_id, fall_start, fall_end, w_all, w_fall, w_no
            )

    all_csv.close()
    fall_csv.close()
    no_csv.close()

    print("\n✅ DONE — Saved keypoints to:", all_csv_path)
    print("Also saved fall/no_fall legacy CSVs.")


if __name__ == "__main__":
    main()
