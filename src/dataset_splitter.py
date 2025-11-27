# src/dataset_splitter.py
#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split

KP_CSV = "data/keypoints/all_keypoints.csv"
OUT_DIR = "data/splits"
TRAIN_DIR = os.path.join(OUT_DIR, "train")
VAL_DIR = os.path.join(OUT_DIR, "val")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

def main(test_size=0.2, random_state=42):
    if not os.path.exists(KP_CSV):
        print("Keypoints CSV not found:", KP_CSV)
        return

    df = pd.read_csv(KP_CSV)
    # determine majority label per video (used for stratification)
    video_labels = df.groupby("video_id")["label"].mean().reset_index()
    # majority: if mean label >= 0.5 -> majority fall else no_fall
    video_labels["majority"] = (video_labels["label"] >= 0.5).astype(int)

    vids = video_labels["video_id"].tolist()
    maj = video_labels["majority"].tolist()

    train_vids, val_vids = train_test_split(vids, test_size=test_size, random_state=random_state, stratify=maj)

    # write per-video csvs
    for v in train_vids:
        sub = df[df["video_id"] == v]
        sub.to_csv(os.path.join(TRAIN_DIR, f"{v}.csv"), index=False)
    for v in val_vids:
        sub = df[df["video_id"] == v]
        sub.to_csv(os.path.join(VAL_DIR, f"{v}.csv"), index=False)

    with open(os.path.join(OUT_DIR, "train_ids.txt"), "w") as f:
        f.write("\n".join(train_vids))
    with open(os.path.join(OUT_DIR, "val_ids.txt"), "w") as f:
        f.write("\n".join(val_vids))

    print("Train videos:", len(train_vids), "Val videos:", len(val_vids))
    print("Saved per-video CSVs to:", TRAIN_DIR, VAL_DIR)
    print("train_ids.txt / val_ids.txt written in", OUT_DIR)

if __name__ == "__main__":
    main()
