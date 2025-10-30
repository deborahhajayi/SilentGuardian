# src/preprocess_data.py
import os
import cv2
import shutil
from tqdm import tqdm

# CONFIG
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
TARGET_SIZE = (128, 128)     # resize images for training
MAX_FRAMES_PER_VIDEO = 300   # set to int for quick tests, or None for all frames
FALL_MARGIN = 0              # follow README exactly: use provided window
SAVE_CROPPED = False         # per your request: do NOT crop, save full frame

# Heuristic params for impact detection (Option C)
VEL_PEAK_WINDOW = 3         # smoothing window when finding vertical velocity peak
VEL_STABLE_THRESH = 2.0     # threshold (pixels/frame) for "stabilized" vertical movement
HEIGHT_DROP_RATIO = 0.85    # heuristic: collapsed height <= this * pre-fall height

# -------------------- helpers --------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clear_dir(path):
    """Remove everything inside a directory (not the directory itself)."""
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        full = os.path.join(path, name)
        try:
            if os.path.isfile(full) or os.path.islink(full):
                os.unlink(full)
            elif os.path.isdir(full):
                shutil.rmtree(full)
        except Exception as e:
            print(f"[warn] Could not remove {full}: {e}")

def parse_annotation_file(txt_path):
    """
    Parse annotation file following README:
      - first numeric line = fall_start (frame index)
      - second numeric line = fall_end (frame index)
      - subsequent per-frame lines: frame_idx, flag?, height, width, center_x, center_y
    Returns:
      fall_start (int or None), fall_end (int or None), bbox_map: {frame_idx: {'bbox':(...),'h':..., 'w':..., 'cx':..., 'cy':...}}
    """
    if not txt_path:
        return None, None, {}

    if not os.path.exists(txt_path):
        print(f"[warn] Annotation file not found: {txt_path}")
        return None, None, {}

    with open(txt_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        return None, None, {}

    # parse numeric tokens line-by-line
    parsed_rows = []
    for ln in lines:
        parts = ln.replace(',', ' ').split()
        nums = []
        for p in parts:
            try:
                if '.' in p:
                    nums.append(float(p))
                else:
                    nums.append(int(p))
            except:
                pass
        if nums:
            parsed_rows.append(nums)

    # first two numeric lines are fall_start and fall_end per README
    fall_start = None
    fall_end = None
    if len(parsed_rows) >= 2 and len(parsed_rows[0]) >= 1 and len(parsed_rows[1]) >= 1:
        try:
            fall_start = int(parsed_rows[0][0])
            fall_end = int(parsed_rows[1][0])
        except:
            fall_start = None
            fall_end = None

    # build bbox_map from remaining rows (if available)
    bbox_map = {}
    for row in parsed_rows[2:]:
        if len(row) >= 6:
            try:
                frame_idx = int(row[0])
                h = float(row[2])
                w = float(row[3])
                cx = float(row[4])
                cy = float(row[5])
            except:
                continue
            if h > 0 and w > 0:
                x1 = int(round(cx - w / 2.0))
                y1 = int(round(cy - h / 2.0))
                x2 = int(round(cx + w / 2.0))
                y2 = int(round(cy + h / 2.0))
                bbox_map[frame_idx] = {"bbox": (x1, y1, x2, y2), "h": h, "w": w, "cx": cx, "cy": cy}
    return fall_start, fall_end, bbox_map

def smooth(arr, k=3):
    if k <= 1 or not arr:
        return arr[:]
    half = k // 2
    n = len(arr)
    out = []
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        out.append(sum(arr[l:r]) / (r - l))
    return out

def detect_impact_frame_option_c(fall_start, fall_end, bbox_map):
    """
    Use bbox center vertical velocity + height heuristics to estimate impact frame inside annotated window.
    Returns an integer frame index or None if not enough data (caller will fallback).
    """
    if fall_start is None or fall_end is None:
        return None

    seq = []
    for f in range(fall_start, fall_end + 1):
        if f in bbox_map:
            e = bbox_map[f]
            seq.append((f, e["cy"], e["h"]))
    if len(seq) < 3:
        return None

    frames = [s[0] for s in seq]
    cys = [s[1] for s in seq]
    hs = [s[2] for s in seq]

    vy = [cys[i] - cys[i - 1] for i in range(1, len(cys))]
    if not vy:
        return None
    vy_s = smooth(vy, k=VEL_PEAK_WINDOW)

    # peak downward speed (largest positive vy; y increases downward)
    peak_idx = max(range(len(vy_s)), key=lambda i: vy_s[i])
    # pre-peak height average
    pre_idx = max(0, peak_idx - 1)
    pre_heights = hs[:pre_idx + 1] if pre_idx >= 0 else hs
    avg_pre_h = sum(pre_heights) / len(pre_heights) if pre_heights else hs[0]

    stable_idx = None
    for j in range(peak_idx + 1, len(vy_s)):
        if abs(vy_s[j]) < VEL_STABLE_THRESH:
            h_after = hs[j + 1] if (j + 1) < len(hs) else hs[-1]
            if h_after <= avg_pre_h * HEIGHT_DROP_RATIO or abs(vy_s[j]) < (VEL_STABLE_THRESH / 2):
                stable_idx = j + 1  # vy index offset -> frame index in seq
                break

    if stable_idx is not None:
        return int(frames[stable_idx])
    # fallback: choose the frame at peak (peak_idx+1 in frames because vy is offset)
    fallback_idx = min(len(frames) - 1, peak_idx + 1)
    return int(frames[fallback_idx])

# -------------------- main processing --------------------
def process_video(video_path, annot_path, out_dirs):
    """
    Process a single video and save frames to out_dirs: dict with keys 'fall','falling','no_fall'.
    Follows README: frames inside [fall_start..fall_end] are considered fall-window; those are split
    into 'falling' and 'fall' using impact detection; outside window -> no_fall.
    Returns (n_fall, n_falling, n_no)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fall_start, fall_end, bbox_map = parse_annotation_file(annot_path)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    saved = {"fall": 0, "falling": 0, "no_fall": 0}

    impact_frame = None
    if fall_start is not None and fall_end is not None:
        impact_frame = detect_impact_frame_option_c(fall_start, fall_end, bbox_map)
        if impact_frame is None:
            # fallback to midpoint of annotated window
            impact_frame = (fall_start + fall_end) // 2

    frame_idx = 0
    pbar = tqdm(total=total_frames if total_frames > 0 else None,
                desc=f"Processing {basename}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1  # annotation files appear 1-indexed

        # Decide label
        if fall_start is not None and fall_end is not None and (fall_start - FALL_MARGIN) <= frame_idx <= (fall_end + FALL_MARGIN):
            # inside annotated fall window -> split based on impact_frame
            if impact_frame is None:
                # if no impact info, treat frames <= midpoint as falling, > midpoint as fall
                mid = (fall_start + fall_end) // 2
                label = "falling" if frame_idx < mid else "fall"
            else:
                if frame_idx < impact_frame:
                    label = "falling"
                else:
                    label = "fall"
        else:
            label = "no_fall"

        # resize full-frame and save (no cropping)
        try:
            resized = cv2.resize(frame, TARGET_SIZE)
        except Exception as e:
            print(f"Skipping frame {frame_idx} in {basename} due to resize error:", e)
            pbar.update(1)
            if MAX_FRAMES_PER_VIDEO and frame_idx >= MAX_FRAMES_PER_VIDEO:
                break
            continue

        out_dir = out_dirs[label]
        fname = f"{basename}_frame{frame_idx:05d}.jpg"
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, resized)
        saved[label] += 1

        pbar.update(1)

        if MAX_FRAMES_PER_VIDEO and frame_idx >= MAX_FRAMES_PER_VIDEO:
            break

    pbar.close()
    cap.release()
    return saved["fall"], saved["falling"], saved["no_fall"]

def main():
    # prepare output dirs
    ensure_dir(OUT_DIR)
    out_fall = os.path.join(OUT_DIR, "fall")
    out_falling = os.path.join(OUT_DIR, "falling")
    out_no = os.path.join(OUT_DIR, "no_fall")
    ensure_dir(out_fall); ensure_dir(out_falling); ensure_dir(out_no)

    # Clear old processed images BEFORE writing new ones (per your request)
    print("[info] Clearing old processed images...")
    clear_dir(out_fall)
    clear_dir(out_falling)
    clear_dir(out_no)

    # iterate through scene folders under RAW_DIR
    scenes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    scenes = [s for s in scenes if any(x in s.lower() for x in ["coffee", "home", "lecture"])]

    totals = {"fall": 0, "falling": 0, "no_fall": 0}
    for scene in scenes:
        scene_path = os.path.join(RAW_DIR, scene)
        inner = os.listdir(scene_path)
        if scene in inner:
            dataset_root = os.path.join(scene_path, scene)
        else:
            dataset_root = scene_path

        videos_dir = os.path.join(dataset_root, "Videos")
        ann_dir = os.path.join(dataset_root, "Annotation_files")
        if not os.path.isdir(videos_dir):
            videos_dir = dataset_root

        video_files = sorted([f for f in os.listdir(videos_dir) if f.lower().endswith(('.avi', '.mp4'))])
        print(f"\nScene {scene}: found {len(video_files)} videos (videos dir: {videos_dir})")

        for vf in video_files:
            video_path = os.path.join(videos_dir, vf)
            base = os.path.splitext(vf)[0]
            annot_candidates = []
            if os.path.isdir(ann_dir):
                for a in os.listdir(ann_dir):
                    if a.lower().startswith(base.lower()) and a.lower().endswith('.txt'):
                        annot_candidates.append(os.path.join(ann_dir, a))
                if not annot_candidates:
                    import re
                    m = re.search(r'\d+', base)
                    if m:
                        num = m.group(0)
                        for a in os.listdir(ann_dir):
                            if num in a and a.lower().endswith('.txt'):
                                annot_candidates.append(os.path.join(ann_dir, a))
            annot_path = annot_candidates[0] if annot_candidates else None

            out_dirs = {"fall": out_fall, "falling": out_falling, "no_fall": out_no}
            saved = process_video(video_path, annot_path, out_dirs)
            if saved is None:
                sf = sfal = sn = 0
            else:
                sf, sfal, sn = saved

            totals["fall"] += sf
            totals["falling"] += sfal
            totals["no_fall"] += sn
            print(f" -> Saved {sf} fall, {sfal} falling, {sn} no_fall from {vf}")

    print("\nDone. Totals:", totals)
    print("Processed images are in:", OUT_DIR)
    print("IMPORTANT: This script clears data/processed/* at start. If you want to add curated external fall images, put them in data/external/fall_images and copy them into data/processed/fall AFTER running this script.")

if __name__ == "__main__":
    main()
