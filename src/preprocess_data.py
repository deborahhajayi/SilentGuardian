# src/preprocess_data.py
import os
import cv2
from tqdm import tqdm
import shutil
import re

# ---------- CONFIG ----------
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
TARGET_SIZE = (128, 128)     # resize images for training
MAX_FRAMES_PER_VIDEO = 300   # set to an int for quick tests, or None for all frames
FALL_MARGIN = 0              # margin around annotated window (set 0 since we follow README exactly)
SAVE_CROPPED = False         # YOU REQUESTED: no cropping, save full frames
# ----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clear_dir_of_images(path):
    """Remove files (images) inside a directory. Keep subfolders if present."""
    if not os.path.isdir(path):
        return
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
        elif os.path.isdir(fpath):
            # remove files inside subdir, but keep folder
            for subf in os.listdir(fpath):
                subfp = os.path.join(fpath, subf)
                if os.path.isfile(subfp):
                    try:
                        os.remove(subfp)
                    except Exception:
                        pass

def parse_annotation_file(txt_path):
    """
    STRICT parser following the README:
      First non-empty line -> frame number of the beginning of the fall (fall_start)
      Second non-empty line -> frame number of the end of the fall (fall_end)
      Remaining lines -> per-frame info (frame_idx, label_flag, h, w, cx, cy)
    Returns: (fall_start:int or None, fall_end:int or None, bbox_map:dict)
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

    # First two lines: fall_start, fall_end (some files may provide only start or only end — handle gracefully)
    fall_start = None
    fall_end = None
    try:
        if len(lines) >= 1:
            fall_start = int(lines[0])
        if len(lines) >= 2:
            fall_end = int(lines[1])
    except Exception:
        # if parsing fails, fallback to None
        fall_start = None
        fall_end = None

    # parse per-frame bbox entries, but we won't crop if SAVE_CROPPED=False.
    bbox_map = {}
    for ln in lines[2:]:
        parts = re.split(r'[, \t]+', ln)
        nums = []
        for p in parts:
            try:
                if '.' in p:
                    nums.append(float(p))
                else:
                    nums.append(int(p))
            except:
                pass
        if len(nums) >= 6:
            frame_idx = int(nums[0])
            # order: frame_idx, ???, h, w, cx, cy  (we saw files in this format). We'll use indices 2..5.
            try:
                h = float(nums[2]); w = float(nums[3]); cx = float(nums[4]); cy = float(nums[5])
            except:
                continue
            if h > 0 and w > 0:
                x1 = int(round(cx - w/2)); y1 = int(round(cy - h/2))
                x2 = int(round(cx + w/2)); y2 = int(round(cy + h/2))
                bbox_map[frame_idx] = (x1, y1, x2, y2)
    return fall_start, fall_end, bbox_map

def process_video(video_path, annot_path, out_fall_dir, out_no_dir):
    """
    Process a single video:
      - reads fall_start, fall_end from annotation file (README spec)
      - reads frames with OpenCV
      - labels frames as:
          * "no_fall" if frame_idx < fall_start
          * "fall" if fall_start <= frame_idx <= fall_end
          * skip frames > fall_end
      - crops disabled (full frames saved) because SAVE_CROPPED=False
      - resizes to TARGET_SIZE and writes to out_fall_dir or out_no_dir
    Returns: (saved_count_fall, saved_count_no)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Failed to open video: {video_path}")
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fall_start, fall_end, bbox_map = parse_annotation_file(annot_path)

    # Apply margin if configured (we set FALL_MARGIN=0 by default to follow README)
    eff_start = None
    eff_end = None
    if fall_start is not None:
        eff_start = max(1, int(fall_start - FALL_MARGIN))
    if fall_end is not None:
        eff_end = int(fall_end + FALL_MARGIN)

    basename = os.path.splitext(os.path.basename(video_path))[0]
    saved_count_fall = 0
    saved_count_no = 0
    frame_idx = 0

    pbar = tqdm(total=total_frames if total_frames > 0 else None,
                desc=f"Processing {basename}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1  # annotation frames are 1-indexed in these files

        # Decide label using strict README logic:
        # - before fall_start => no_fall
        # - in [fall_start..fall_end] => fall
        # - after fall_end => SKIP
        label = None
        if eff_start is not None and eff_end is not None:
            if frame_idx < eff_start:
                label = "no_fall"
            elif eff_start <= frame_idx <= eff_end:
                label = "fall"
            else:
                label = None  # skip frames after fall_end
        elif fall_start is not None and fall_end is not None:
            # fallback if margin vars not set
            if frame_idx < fall_start:
                label = "no_fall"
            elif fall_start <= frame_idx <= fall_end:
                label = "fall"
            else:
                label = None
        else:
            # No annotation available: treat everything as no_fall (you can change this behavior later)
            label = "no_fall"

        # Skip frames where label is None (after fall_end)
        if label is None:
            pbar.update(1)
            if MAX_FRAMES_PER_VIDEO and frame_idx >= MAX_FRAMES_PER_VIDEO:
                break
            continue

        # NO CROPPING: we save full frame (user requested full images)
        saved_frame = frame

        # Resize (safe)
        try:
            resized = cv2.resize(saved_frame, TARGET_SIZE)
        except Exception as e:
            print(f"[warn] Skipping frame {frame_idx} in {basename} due to resize error: {e}")
            pbar.update(1)
            if MAX_FRAMES_PER_VIDEO and frame_idx >= MAX_FRAMES_PER_VIDEO:
                break
            continue

        out_dir = out_fall_dir if label == "fall" else out_no_dir
        filename = f"{basename}_frame{frame_idx:05d}.jpg"
        out_path = os.path.join(out_dir, filename)
        # write
        try:
            cv2.imwrite(out_path, resized)
        except Exception as e:
            print(f"[warn] Failed to write {out_path}: {e}")
            pbar.update(1)
            continue

        if label == "fall":
            saved_count_fall += 1
        else:
            saved_count_no += 1

        pbar.update(1)

        if MAX_FRAMES_PER_VIDEO and frame_idx >= MAX_FRAMES_PER_VIDEO:
            break

    pbar.close()
    cap.release()
    return saved_count_fall, saved_count_no


def find_all_scenes(root_raw):
    """
    Discover scene directories in data/raw that contain nested duplicated folders,
    and return a list of dataset roots to scan for Videos/Annotation_files.
    This tries to follow the structure you described:
      data/raw/<scene>/<scene>/Videos  (or sometimes just data/raw/<scene>/Videos)
    """
    scenes = []
    if not os.path.isdir(root_raw):
        return scenes
    for d in sorted(os.listdir(root_raw)):
        dpath = os.path.join(root_raw, d)
        if not os.path.isdir(dpath):
            continue
        inner_list = os.listdir(dpath)
        # If there's a subfolder with same name, use that as dataset_root
        if d in inner_list and os.path.isdir(os.path.join(dpath, d)):
            scenes.append(os.path.join(dpath, d))
        else:
            scenes.append(dpath)
    return scenes


def main():
    ensure_dir(OUT_DIR)
    fall_out = os.path.join(OUT_DIR, "fall")
    no_out = os.path.join(OUT_DIR, "no_fall")
    falling_out = os.path.join(OUT_DIR, "falling")  # optional, kept for compatibility
    ensure_dir(fall_out); ensure_dir(no_out); ensure_dir(falling_out)

    # CLEAR old processed images (user requested deletion before writing new ones)
    clear_dir_of_images(fall_out)
    clear_dir_of_images(no_out)
    clear_dir_of_images(falling_out)

    dataset_roots = find_all_scenes(RAW_DIR)
    if not dataset_roots:
        print(f"[error] No scene folders found under {RAW_DIR}. Check your dataset path.")
        return

    total_fall = 0
    total_no = 0

    for dataset_root in dataset_roots:
        # expect Videos and Annotation_files inside dataset_root
        videos_dir = os.path.join(dataset_root, "Videos")
        ann_dir = os.path.join(dataset_root, "Annotation_files")

        # If Videos folder doesn't exist, fallback to dataset_root itself
        if not os.path.isdir(videos_dir):
            videos_dir = dataset_root

        # gather video files
        try:
            video_files = sorted([f for f in os.listdir(videos_dir) if f.lower().endswith(('.avi', '.mp4'))])
        except Exception as e:
            print(f"[warn] Could not list videos in {videos_dir}: {e}")
            video_files = []

        print(f"\nScene {os.path.basename(dataset_root)}: found {len(video_files)} videos (videos dir: {videos_dir})")

        for vf in video_files:
            video_path = os.path.join(videos_dir, vf)

            # find matching annotation file by same base or numeric match
            base = os.path.splitext(vf)[0]  # e.g. "video (1)"
            annot_candidates = []
            if os.path.isdir(ann_dir):
                for a in os.listdir(ann_dir):
                    if a.lower().endswith('.txt'):
                        # direct start match
                        if a.lower().startswith(base.lower()):
                            annot_candidates.append(os.path.join(ann_dir, a))
                # fallback: find by number anywhere in filename
                if not annot_candidates:
                    m = re.search(r'\d+', base)
                    if m:
                        num = m.group(0)
                        for a in os.listdir(ann_dir):
                            if num in a and a.lower().endswith('.txt'):
                                annot_candidates.append(os.path.join(ann_dir, a))
            annot_path = annot_candidates[0] if annot_candidates else None

            saved_f, saved_n = process_video(video_path, annot_path, fall_out, no_out)
            total_fall += saved_f
            total_no += saved_n
            print(f" -> Saved {saved_f} fall frames and {saved_n} no_fall frames from {vf}")

    print("\nDone. Totals:", "fall:", total_fall, "no_fall:", total_no)
    print("Processed images are in:", OUT_DIR)


if __name__ == "__main__":
    main()
