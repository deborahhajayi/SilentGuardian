# src/preprocess_data.py  (UPDATED to skip ambiguous frames between fall_start and visible start)
import os
import cv2
import re
import shutil
from tqdm import tqdm
from datetime import datetime

# ---------- CONFIG ----------
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
TARGET_SIZE = (128, 128)     # resize images for training
MAX_FRAMES_PER_VIDEO = None  # set to an int for quick tests, or None for all frames
FALL_MARGIN = 0              # margin around annotated window (0 to follow README exactly)
SAVE_CROPPED = False         # save full frames
SKIPPED_LOG = os.path.join(OUT_DIR, "skipped_videos.txt")

# NEW CONFIGS:
# Where inside the annotated window we start labeling as 'fall' (0.66 = later in window)
FALL_VISIBLE_FRACTION = 0.66

# Number-of-frame subsampling (reduce near-duplicate frames). 1 = keep every frame.
KEEP_EVERY_N_FRAMES = 1

# External images you may want merged into processed set (optional)
EXTERNAL_FALL_DIR = "data/external/fall_images"
EXTERNAL_NO_DIR = "data/external/no_fall_images"
# ----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clear_dir_of_images(path):
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
            for subf in os.listdir(fpath):
                subfp = os.path.join(fpath, subf)
                if os.path.isfile(subfp):
                    try:
                        os.remove(subfp)
                    except Exception:
                        pass

def log_skipped(video_relpath, reason):
    try:
        ensure_dir(os.path.dirname(SKIPPED_LOG))
        with open(SKIPPED_LOG, "a", encoding="utf-8") as fh:
            ts = datetime.utcnow().isoformat()
            fh.write(f"{ts}\t{video_relpath}\t{reason}\n")
    except Exception as e:
        print(f"[warn] Failed to write skipped log: {e}")

def parse_annotation_file(txt_path):
    if not txt_path:
        return None, None, {}
    if not os.path.exists(txt_path):
        return None, None, {}

    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_lines = [ln.rstrip("\n") for ln in f]
    except Exception as e:
        print(f"[warn] Could not read annotation {txt_path}: {e}")
        return None, None, {}

    non_empty = []
    for i, ln in enumerate(raw_lines):
        if ln.strip():
            non_empty.append((i, ln.strip()))

    if not non_empty:
        return None, None, {}

    standalone_ints = []
    for idx_in_list, (orig_index, ln) in enumerate(non_empty):
        if re.fullmatch(r'[+-]?\d+', ln):
            try:
                val = int(ln)
                standalone_ints.append((idx_in_list, orig_index, val))
            except:
                pass

    fall_start = None
    fall_end = None
    if len(standalone_ints) >= 2:
        for k in range(len(standalone_ints)-1):
            i1, orig1, v1 = standalone_ints[k]
            i2, orig2, v2 = standalone_ints[k+1]
            if i2 == i1 + 1 and v1 >= 0 and v2 >= v1:
                fall_start, fall_end = int(v1), int(v2)
                break

    if fall_start is None:
        try:
            first_ln = non_empty[0][1]
            second_ln = non_empty[1][1] if len(non_empty) >= 2 else None
            if re.fullmatch(r'[+-]?\d+', first_ln) and second_ln is not None and re.fullmatch(r'[+-]?\d+', second_ln):
                a = int(first_ln); b = int(second_ln)
                if a >= 0 and b >= a:
                    fall_start, fall_end = a, b
        except Exception:
            fall_start = None
            fall_end = None

    # Build bbox_map if present (not required)
    bbox_map = {}
    for _, ln in non_empty:
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Failed to open video: {video_path}")
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fall_start, fall_end, bbox_map = parse_annotation_file(annot_path)
    rel_video = os.path.relpath(video_path)

    if annot_path is None:
        reason = "no annotation file"
        print(f"[skip] {rel_video}: {reason}")
        log_skipped(rel_video, reason)
        cap.release()
        return 0, 0

    if fall_start is None or fall_end is None:
        reason = "annotation missing valid fall start/end"
        print(f"[skip] {rel_video}: {reason}")
        log_skipped(rel_video, reason)
        cap.release()
        return 0, 0

    explicit_no_fall = (fall_start == 0 and fall_end == 0)

    # compute visible-fall start inside annotated window (only used to label 'fall')
    if not explicit_no_fall:
        window_len = max(1, fall_end - fall_start + 1)
        visible_offset = int(round(window_len * FALL_VISIBLE_FRACTION))
        fall_visible_start = fall_start + visible_offset
        eff_fall_start = max(1, int(fall_visible_start - FALL_MARGIN))
        eff_fall_end = int(fall_end + FALL_MARGIN)
    else:
        eff_fall_start = None
        eff_fall_end = None

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
        frame_idx += 1

        # subsample frames if requested
        if KEEP_EVERY_N_FRAMES > 1:
            if (frame_idx % KEEP_EVERY_N_FRAMES) != 1:
                pbar.update(1)
                if MAX_FRAMES_PER_VIDEO and frame_idx >= MAX_FRAMES_PER_VIDEO:
                    break
                continue

        # Labeling logic
        if explicit_no_fall:
            label = "no_fall"
        else:
            # IMPORTANT: user request -> NO_FALL = everything before fall_start (strict)
            if frame_idx < fall_start:
                label = "no_fall"
            # skip frames inside annotated window but before visible-fall start (ambiguous bending)
            elif fall_start <= frame_idx < eff_fall_start:
                label = None   # skip ambiguous part
            # frames from visible fall start -> fall
            elif eff_fall_start <= frame_idx <= eff_fall_end:
                label = "fall"
            else:
                label = None

        if label is None:
            pbar.update(1)
            if MAX_FRAMES_PER_VIDEO and frame_idx >= MAX_FRAMES_PER_VIDEO:
                break
            continue

        # No cropping; save full frame
        saved_frame = frame
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

# (rest of file: scene discovery and main - unchanged)
def find_all_scenes(root_raw):
    scenes = []
    if not os.path.isdir(root_raw):
        return scenes
    for d in sorted(os.listdir(root_raw)):
        dpath = os.path.join(root_raw, d)
        if not os.path.isdir(dpath):
            continue
        inner_list = os.listdir(dpath)
        if d in inner_list and os.path.isdir(os.path.join(dpath, d)):
            scenes.append(os.path.join(dpath, d))
        else:
            scenes.append(dpath)
    return scenes

def find_annotation_dir(dataset_root):
    for cand in ["Annotation_files", "Annotations_files"]:
        p = os.path.join(dataset_root, cand)
        if os.path.isdir(p):
            return p
    return None

def copy_external_images():
    fall_out = os.path.join(OUT_DIR, "fall")
    no_out = os.path.join(OUT_DIR, "no_fall")
    ensure_dir(fall_out); ensure_dir(no_out)
    copied = 0
    if os.path.isdir(EXTERNAL_FALL_DIR):
        for f in os.listdir(EXTERNAL_FALL_DIR):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                src = os.path.join(EXTERNAL_FALL_DIR, f)
                dst = os.path.join(fall_out, f)
                try:
                    shutil.copy2(src, dst); copied += 1
                except:
                    pass
    if os.path.isdir(EXTERNAL_NO_DIR):
        for f in os.listdir(EXTERNAL_NO_DIR):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                src = os.path.join(EXTERNAL_NO_DIR, f)
                dst = os.path.join(no_out, f)
                try:
                    shutil.copy2(src, dst); copied += 1
                except:
                    pass
    return copied

def main():
    ensure_dir(OUT_DIR)
    fall_out = os.path.join(OUT_DIR, "fall")
    no_out = os.path.join(OUT_DIR, "no_fall")
    ensure_dir(fall_out); ensure_dir(no_out)

    # CLEAR old processed images if you want to re-run fresh
    clear_dir_of_images(fall_out)
    clear_dir_of_images(no_out)

    try:
        if os.path.exists(SKIPPED_LOG):
            os.remove(SKIPPED_LOG)
    except:
        pass

    dataset_roots = find_all_scenes(RAW_DIR)
    if not dataset_roots:
        print(f"[error] No scene folders found under {RAW_DIR}. Check your dataset path.")
        return

    total_fall = 0
    total_no = 0
    total_videos = 0
    total_skipped = 0

    for dataset_root in dataset_roots:
        videos_dir = os.path.join(dataset_root, "Videos")
        ann_dir = find_annotation_dir(dataset_root)
        if not os.path.isdir(videos_dir):
            videos_dir = dataset_root

        try:
            video_files = sorted([f for f in os.listdir(videos_dir) if f.lower().endswith(('.avi', '.mp4'))])
        except Exception as e:
            print(f"[warn] Could not list videos in {videos_dir}: {e}")
            video_files = []

        print(f"\nScene {os.path.basename(dataset_root)}: found {len(video_files)} videos (videos dir: {videos_dir})")

        for vf in video_files:
            total_videos += 1
            video_path = os.path.join(videos_dir, vf)
            base = os.path.splitext(vf)[0]
            annot_candidates = []
            if ann_dir and os.path.isdir(ann_dir):
                for a in os.listdir(ann_dir):
                    if a.lower().endswith('.txt'):
                        if a.lower().startswith(base.lower()):
                            annot_candidates.append(os.path.join(ann_dir, a))
                if not annot_candidates:
                    m = re.search(r'\d+', base)
                    if m:
                        num = m.group(0)
                        for a in os.listdir(ann_dir):
                            if num in a and a.lower().endswith('.txt'):
                                annot_candidates.append(os.path.join(ann_dir, a))
            annot_path = annot_candidates[0] if annot_candidates else None

            saved_f, saved_n = process_video(video_path, annot_path, fall_out, no_out)
            if saved_f == 0 and saved_n == 0 and annot_path is None:
                total_skipped += 1

            total_fall += saved_f
            total_no += saved_n
            print(f" -> Saved {saved_f} fall frames and {saved_n} no_fall frames from {vf}")

    copied_external = copy_external_images()
    if copied_external:
        print(f"Copied {copied_external} external images into processed folders.")

    print("\nDone.")
    print(f"Videos scanned: {total_videos}, skipped (logged): {total_skipped}")
    print("Totals:", "fall:", total_fall, "no_fall:", total_no)
    print("Processed images are in:", OUT_DIR)
    if os.path.exists(SKIPPED_LOG):
        print("See skipped list:", SKIPPED_LOG)

if __name__ == "__main__":
    main()
