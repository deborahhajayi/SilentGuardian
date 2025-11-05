#!/usr/bin/env python3
# src/split_dataset.py
"""
Video-level stratified split script (robust on Windows).

Usage:
    python src/split_dataset.py
    python src/split_dataset.py --source data/processed --target data --seed 42 --move
"""

import os
import shutil
import random
import argparse
import re
from collections import defaultdict
from math import floor

DEFAULT_SOURCE = "data/processed"
DEFAULT_TARGET = "data"
CLASSES = ["fall", "no_fall"]
DEFAULT_SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}


def parse_args():
    p = argparse.ArgumentParser(description="Split processed images into train/val/test (video-level stratified)")
    p.add_argument("--source", default=DEFAULT_SOURCE, help="Source processed directory (contains 'fall' and 'no_fall').")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Target base dir (will create target/train, target/val, target/test).")
    p.add_argument("--train", type=float, default=DEFAULT_SPLITS["train"], help="Train split fraction.")
    p.add_argument("--val", type=float, default=DEFAULT_SPLITS["val"], help="Validation split fraction.")
    p.add_argument("--test", type=float, default=DEFAULT_SPLITS["test"], help="Test split fraction.")
    p.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    p.add_argument("--move", action="store_true", help="Move files instead of copying (saves disk space).")
    return p.parse_args()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def clear_dir_files(path):
    """
    Remove files inside a directory (non-recursive) and files inside its immediate subdirectories.
    This avoids using shutil.rmtree which can fail on Windows when a file/dir is open.
    """
    if not os.path.isdir(path):
        return
    for root, dirs, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                # Try to make writable first
                try:
                    os.chmod(fpath, 0o666)
                except Exception:
                    pass
                os.remove(fpath)
            except PermissionError:
                print(f"[warn] Permission denied removing {fpath}. Close any program using it and retry.")
            except Exception as e:
                print(f"[warn] Failed to remove {fpath}: {e}")
        # do not remove directories here; leave folder structure intact
        break  # only top-level; we call this on each class dir individually


def clear_target_dirs(target_base):
    """
    Safely clear files under target/train, target/val, target/test for each class.
    Leaves directory structure in place to avoid permission problems.
    """
    for split in ("train", "val", "test"):
        split_dir = os.path.join(target_base, split)
        if not os.path.isdir(split_dir):
            continue
        # remove files under split/class/*
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                print(f"Clearing files in {cls_dir} ...")
                clear_dir_files(cls_dir)


def extract_video_id(filename):
    """
    Given filename like "<videoBase>_frame00012.jpg", return "<videoBase>".
    """
    name = os.path.splitext(filename)[0]
    if "_frame" in name:
        return name.split("_frame")[0]
    for token in ["_frame", "-frame", ".frame"]:
        if token in name:
            return name.split(token)[0]
    return name


def group_images_by_video(src_cls_dir):
    images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    groups = defaultdict(list)
    for f in images:
        vid = extract_video_id(f)
        groups[vid].append(f)
    return groups


def choose_splits_for_videos(video_ids, target_fractions, rng):
    n = len(video_ids)
    if n == 0:
        return {}
    n_train = max(1, int(floor(n * target_fractions["train"]))) if n >= 1 else 0
    n_val = max(1, int(floor(n * target_fractions["val"]))) if n - n_train >= 1 else 0
    n_assigned = n_train + n_val
    n_test = max(0, n - n_assigned)
    if n_train == 0 and n >= 1:
        n_train = 1
        if n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
    if n_val == 0 and n - n_train >= 1:
        n_val = 1
        if n_test > 0:
            n_test -= 1
    vids = list(video_ids)
    rng.shuffle(vids)
    assignment = {}
    for v in vids[:n_train]:
        assignment[v] = "train"
    for v in vids[n_train:n_train + n_val]:
        assignment[v] = "val"
    for v in vids[n_train + n_val:]:
        assignment[v] = "test"
    return assignment


def main():
    args = parse_args()
    source = args.source
    target = args.target
    seed = args.seed
    move_files = args.move

    if not os.path.isdir(source):
        print(f"[error] Source directory not found: {source}")
        return

    sum_f = args.train + args.val + args.test
    if abs(sum_f - 1.0) > 1e-6:
        print("[error] train+val+test fractions must sum to 1.0")
        return

    fractions = {"train": args.train, "val": args.val, "test": args.test}
    rng = random.Random(seed)

    # Ensure target split dirs exist and clear old files safely
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            ensure_dir(os.path.join(target, split, cls))

    print("Clearing old files in target splits (if any)...")
    clear_target_dirs(target)

    total_counts = {"train": 0, "val": 0, "test": 0}
    per_class_counts = {cls: {"train": 0, "val": 0, "test": 0} for cls in CLASSES}
    per_class_videos = {}

    # Group images per class by video id
    for cls in CLASSES:
        src_cls_dir = os.path.join(source, cls)
        if not os.path.isdir(src_cls_dir):
            print(f"[warn] Class directory does not exist, skipping: {src_cls_dir}")
            per_class_videos[cls] = {}
            continue
        groups = group_images_by_video(src_cls_dir)
        per_class_videos[cls] = groups

    # Assign videos to splits
    video_to_split = {}
    for cls, groups in per_class_videos.items():
        vids = list(groups.keys())
        if not vids:
            continue
        assignment = choose_splits_for_videos(vids, fractions, rng)
        for v, split in assignment.items():
            video_to_split[(cls, v)] = split

    # Copy or move files into target
    for cls, groups in per_class_videos.items():
        src_cls_dir = os.path.join(source, cls)
        for vid, files in groups.items():
            split = video_to_split.get((cls, vid), "train")
            out_dir = os.path.join(target, split, cls)
            ensure_dir(out_dir)
            for fname in files:
                src_path = os.path.join(src_cls_dir, fname)
                dst_path = os.path.join(out_dir, fname)
                try:
                    if move_files:
                        shutil.move(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                    total_counts[split] += 1
                    per_class_counts[cls][split] += 1
                except PermissionError:
                    print(f"[warn] Permission denied copying/moving {src_path} -> {dst_path}")
                except Exception as e:
                    print(f"[warn] Failed to copy/move {src_path} -> {dst_path}: {e}")

    # Summary
    print("\nSplit complete. Summary:")
    print(f"  seed: {seed}   move_files: {move_files}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {total_counts[split]} images")
    print("Per-class breakdown:")
    for cls in CLASSES:
        print(f"  {cls}: train={per_class_counts[cls]['train']}, val={per_class_counts[cls]['val']}, test={per_class_counts[cls]['test']}")
    print(f"\nCreated folders under '{target}/train', '{target}/val', '{target}/test'.")


if __name__ == "__main__":
    main()
