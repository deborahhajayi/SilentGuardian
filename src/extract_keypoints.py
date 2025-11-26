import os
import cv2
import mediapipe as mp
import csv
from tqdm import tqdm
import re

# ---------------------------------------------------
# CORRECTED PATHS (your confirmed structure)
# ---------------------------------------------------
RAW_DIR = "data/raw"
DATASET_DIR = os.path.join(RAW_DIR, "dataset")
VIDEO_DIR = os.path.join(DATASET_DIR, "Videos")
ANN_DIR = os.path.join(DATASET_DIR, "Annotation_files")

OUT_DIR = "data/keypoints"
FALL_FRACTION = 0.66  # fall becomes *visibly obvious* only in last 33%

os.makedirs(OUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose


# ------------------------------
# Parse annotation file
# ------------------------------
def parse_annotation(path):
    """Reads fall start/end frame numbers from annotation file."""
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


# ------------------------------
# Find matching annotation file
# ------------------------------
def find_annotation(video_filename):
    base = os.path.splitext(video_filename)[0].lower()

    # 1) Try exact prefix match
    for f in os.listdir(ANN_DIR):
        if f.lower().endswith(".txt") and f.lower().startswith(base):
            return os.path.join(ANN_DIR, f)

    # 2) Fallback: match numeric ID
    digits = re.findall(r"\d+", base)
    if digits:
        for f in os.listdir(ANN_DIR):
            if digits[0] in f:
                return os.path.join(ANN_DIR, f)

    return None


# ------------------------------
# Extract keypoints from video
# ------------------------------
def extract_keypoints_from_video(video_path, fall_start, fall_end,
                                 writer_fall, writer_no_fall):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    visible_fall_start = fall_start + int((fall_end - fall_start) * FALL_FRACTION)

    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.5,
                      model_complexity=1) as pose:

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pbar.update(1)

            # Mediapipe Pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            # Extract 33×(x,y,z,visibility)
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            # --------------------------
            # Labeling
            # --------------------------
            if fall_start == 0 and fall_end == 0:
                label = "no_fall"
            elif frame_idx < fall_start:
                label = "no_fall"
            elif fall_start <= frame_idx < visible_fall_start:
                label = None
            elif visible_fall_start <= frame_idx <= fall_end:
                label = "fall"
            else:
                label = None

            if label == "fall":
                writer_fall.writerow(keypoints)
            elif label == "no_fall":
                writer_no_fall.writerow(keypoints)

        pbar.close()
    cap.release()


# ------------------------------
# Main
# ------------------------------
def main():
    print("Videos directory:", VIDEO_DIR)
    print("Annotation directory:", ANN_DIR)

    # Open output CSVs
    fall_csv = open(os.path.join(OUT_DIR, "fall_keypoints.csv"), "w", newline="")
    no_csv = open(os.path.join(OUT_DIR, "no_fall_keypoints.csv"), "w", newline="")
    writer_fall = csv.writer(fall_csv)
    writer_no = csv.writer(no_csv)

    # List all videos
    if not os.path.isdir(VIDEO_DIR):
        print("ERROR: Folder not found:", VIDEO_DIR)
        return

    videos = [v for v in os.listdir(VIDEO_DIR)
              if v.lower().endswith((".mp4", ".avi"))]

    print("Found", len(videos), "videos.")

    if len(videos) == 0:
        print("No videos found inside:", VIDEO_DIR)
        return

    for vid in videos:
        vpath = os.path.join(VIDEO_DIR, vid)
        ann_path = find_annotation(vid)

        if ann_path is None:
            print("Skipping (NO annotation found):", vid)
            continue

        fall_start, fall_end = parse_annotation(ann_path)
        if fall_start is None:
            print("Skipping invalid annotation:", ann_path)
            continue

        extract_keypoints_from_video(vpath, fall_start, fall_end,
                                     writer_fall, writer_no)

    fall_csv.close()
    no_csv.close()
    print("\nDONE — Keypoints saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
