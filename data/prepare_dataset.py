"""
Converts ISIC 2018 data into YOLO detection format.

Pipeline:
  Task3 CSV (class labels) + Task1 masks (lesion location)
  → tight bounding boxes
  → YOLO format labels
  → train/val split
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────
IMAGES_DIR = Path("data/images_raw/ISIC2018_Task3_Training_Input")
MASKS_DIR  = Path("data/masks_raw/ISIC2018_Task1_Training_GroundTruth")
CSV_PATH   = Path("data/labels_raw/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")

OUTPUT_DIR = Path("data/yolo")

# ── Class mapping ──────────────────────────────────────────────
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_MAP   = {name: i for i, name in enumerate(CLASS_NAMES)}

# ── Class weights for imbalanced training ─────────────────────
# Inverse frequency — rare classes get higher weight
CLASS_COUNTS = {
    'MEL': 1113, 'NV': 6705, 'BCC': 514,
    'AKIEC': 327, 'BKL': 1099, 'DF': 115, 'VASC': 142
}
total = sum(CLASS_COUNTS.values())
CLASS_WEIGHTS = {
    CLASS_MAP[k]: round(total / (len(CLASS_COUNTS) * v), 3)
    for k, v in CLASS_COUNTS.items()
}


def mask_to_bbox(mask_path, img_w, img_h):
    """
    Convert binary segmentation mask → tight YOLO bounding box.

    Returns: (cx, cy, w, h) normalized 0-1
    Returns None if mask is empty or missing.
    """
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Resize mask to match image dimensions
    mask = cv2.resize(mask, (img_w, img_h))

    # Find lesion pixels
    coords = cv2.findNonZero((mask > 127).astype(np.uint8))
    if coords is None or len(coords) == 0:
        return None

    # Tight bounding box
    x, y, w, h = cv2.boundingRect(coords)

    # Add 5% padding around lesion
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img_w - x, w + 2 * pad_x)
    h = min(img_h - y, h + 2 * pad_y)

    # Convert to YOLO format (normalized cx, cy, w, h)
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h

    return cx, cy, nw, nh


def prepare():
    print("Reading CSV labels...")
    df = pd.read_csv(CSV_PATH)

    # Get class for each image (one-hot → class index)
    df['class_id'] = df[CLASS_NAMES].values.argmax(axis=1)
    df['class_name'] = df['class_id'].apply(lambda x: CLASS_NAMES[x])

    print(f"Total images: {len(df)}")
    print("\nClass distribution:")
    for name in CLASS_NAMES:
        count = (df['class_name'] == name).sum()
        weight = CLASS_WEIGHTS[CLASS_MAP[name]]
        print(f"  {name:<8} {count:>5} images  weight={weight:.3f}")

    # Train/val split — stratified to keep class ratios
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df['class_id']
    )
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

    # Create output directories
    for split in ['train', 'val']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Stats tracking
    stats = {
        'processed': 0,
        'mask_found': 0,
        'mask_missing': 0,
        'skipped': 0
    }

    for split, split_df in [('train', train_df), ('val', val_df)]:
        print(f"\nProcessing {split} split...")

        for _, row in split_df.iterrows():
            img_id    = row['image']
            class_id  = row['class_id']

            img_src = IMAGES_DIR / f"{img_id}.jpg"
            if not img_src.exists():
                stats['skipped'] += 1
                continue

            # Read image to get dimensions
            img = cv2.imread(str(img_src))
            if img is None:
                stats['skipped'] += 1
                continue
            img_h, img_w = img.shape[:2]

            # Try to get tight bbox from segmentation mask
            mask_path = MASKS_DIR / f"{img_id}_segmentation.png"
            bbox = mask_to_bbox(mask_path, img_w, img_h)

            if bbox is not None:
                stats['mask_found'] += 1
                cx, cy, w, h = bbox
            else:
                # Fallback: center crop (80% of image)
                stats['mask_missing'] += 1
                cx, cy, w, h = 0.5, 0.5, 0.8, 0.8

            # Copy image
            img_dst = OUTPUT_DIR / split / 'images' / f"{img_id}.jpg"
            cv2.imwrite(str(img_dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Write YOLO label
            label_dst = OUTPUT_DIR / split / 'labels' / f"{img_id}.txt"
            with open(label_dst, 'w') as f:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            stats['processed'] += 1

        print(f"  Done {split}!")

    # Save dataset YAML
    yaml_content = f"""path: {OUTPUT_DIR.absolute()}
train: train/images
val: val/images

nc: 7
names:
  0: MEL
  1: NV
  2: BCC
  3: AKIEC
  4: BKL
  5: DF
  6: VASC

# Class weights (inverse frequency for imbalanced training)
# MEL:{CLASS_WEIGHTS[0]}  NV:{CLASS_WEIGHTS[1]}  BCC:{CLASS_WEIGHTS[2]}
# AKIEC:{CLASS_WEIGHTS[3]}  BKL:{CLASS_WEIGHTS[4]}  DF:{CLASS_WEIGHTS[5]}  VASC:{CLASS_WEIGHTS[6]}
"""
    with open("data/skin.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"\n{'='*45}")
    print(f"  Processed   : {stats['processed']}")
    print(f"  Real masks  : {stats['mask_found']}")
    print(f"  Fallback    : {stats['mask_missing']}")
    print(f"  Skipped     : {stats['skipped']}")
    print(f"  YAML saved  : data/skin.yaml")
    print(f"{'='*45}")
    print("\n✅ Dataset ready for YOLO training!")


if __name__ == '__main__':
    prepare()