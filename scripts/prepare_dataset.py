"""
Step 2 — Explore dataset structure, map to canonical classes,
create 70/15/15 train/val/test splits, save CSV manifests.

Handles both layouts produced by the Kaggle dataset:
  Layout A: data/raw/images/<class>/*.jpg          (flat)
  Layout B: data/raw/images/<split>/<class>/*.jpg  (pre-split)
"""

import os, sys, random
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR    = Path("data/raw/images")
SPLIT_DIR  = Path("data/processed/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Map dataset folder names → canonical 5-class names
LABEL_MAP = {
    # Positive / Benign / negative labels (ALL dataset)
    "all":      "ALL", "positive": "ALL", "pos": "ALL",
    # AML
    "aml":      "AML",
    # CLL
    "cll":      "CLL",
    # CML
    "cml":      "CML",
    # Normal / Negative
    "normal":   "Normal", "negative": "Normal", "neg": "Normal",
    "hem":      "Normal", "healthy":  "Normal",
    # Alternate names
    "benign":   "Normal", "malignant": "ALL",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
RANDOM_SEED = 42


def collect_images() -> dict[str, list[Path]]:
    """Walk RAW_DIR and return {canonical_class: [paths]} dict."""
    class_images: dict[str, list[Path]] = defaultdict(list)

    for path in RAW_DIR.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        # Try parent and grandparent for class name
        for part in [path.parent.name, path.parent.parent.name]:
            canonical = LABEL_MAP.get(part.lower().strip())
            if canonical:
                class_images[canonical].append(path)
                break

    return dict(class_images)


def print_structure():
    print("\n=== DATASET DIRECTORY STRUCTURE ===")
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_dir():
            imgs = [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS]
            if imgs:
                print(f"  {p.relative_to(RAW_DIR)}: {len(imgs)} images")


def make_splits(class_images: dict) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    rows = []
    for cls, paths in sorted(class_images.items()):
        shuffled = paths.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        for i, p in enumerate(shuffled):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rows.append({"filepath": str(p), "label": cls, "split": split})
    return pd.DataFrame(rows)


def main():
    print_structure()

    print("\n=== COLLECTING IMAGES ===")
    class_images = collect_images()

    if not class_images:
        print("ERROR: No images found with recognised class labels.")
        print("Folder names must contain one of:", list(LABEL_MAP.keys()))
        sys.exit(1)

    total = sum(len(v) for v in class_images.values())
    print(f"\nFound {total} images across {len(class_images)} classes:")
    for cls, paths in sorted(class_images.items()):
        print(f"  {cls:8s}: {len(paths):6,} images")

    print("\n=== CREATING 70/15/15 SPLITS ===")
    df = make_splits(class_images)

    for split in ["train", "val", "test"]:
        sub = df[df.split == split]
        dist = sub.label.value_counts().to_dict()
        print(f"\n  {split.upper()} ({len(sub)} images): {dist}")
        out = SPLIT_DIR / f"{split}.csv"
        sub.to_csv(out, index=False)
        print(f"  Saved: {out}")

    df.to_csv(SPLIT_DIR / "all_splits.csv", index=False)
    print(f"\nFull manifest saved: {SPLIT_DIR / 'all_splits.csv'}")
    print(f"Total images: {len(df)}")
    return df


if __name__ == "__main__":
    main()
