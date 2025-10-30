"""
import os
import shutil
import random

SOURCE_DIR = "data/processed"
TARGET_DIR = "data"
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

for cls in ["fall", "no_fall"]:
    src_cls_dir = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg','.png'))]
    random.shuffle(images)
    
    n = len(images)
    train_end = int(n * SPLITS["train"])
    val_end = train_end + int(n * SPLITS["val"])
    
    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }
    
    for split_name, split_files in splits.items():
        out_dir = os.path.join(TARGET_DIR, split_name, cls)
        os.makedirs(out_dir, exist_ok=True)
        for f in split_files:
            shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(out_dir, f))

print("Dataset split complete!")

"""