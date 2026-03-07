#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Input directory (where keypoint CSVs are)
KP_DIR = "data/keypoints"

# Output directory (NEW — separate from fall splits)
OUT_DIR = "data/splits_hand"

# Create output folder if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILES = [
    "train_val_palm.csv",
    "train_val_like.csv",
    "train_val_neutral.csv"
]

TRAIN_FILE = os.path.join(OUT_DIR, "train.csv")
VAL_FILE = os.path.join(OUT_DIR, "val.csv")


def main(test_size=0.2, random_state=42):

    dfs = []

    # Load all gesture CSVs
    for file in CSV_FILES:
        path = os.path.join(KP_DIR, file)

        if not os.path.exists(path):
            print("⚠ Missing:", path)
            continue

        print("Loading:", path)
        dfs.append(pd.read_csv(path))

    if len(dfs) == 0:
        print("❌ No keypoint CSVs found.")
        return

    # Combine all gesture data
    df = pd.concat(dfs, ignore_index=True)

    print("\nTotal samples:", len(df))
    print("Class distribution:")
    print(df["label"].value_counts())

    # Stratified split (keeps gesture balance)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    # Save split files
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)

    print("\n✅ Hand dataset split complete")
    print("Train samples:", len(train_df))
    print("Val samples:", len(val_df))
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()