"""
Converts ISIC 2018 Task 3 data into YOLO detection format.
Uses saliency-based lesion localization since Task 1 masks
cover a different image set.

Saliency approach:
  Dermoscopy images are always centered on the lesion.
  Lesions have higher color saturation than surrounding skin.
  → Convert to HSV → threshold saturation channel
  → Find largest contiguous region → tight bounding box
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────
IMAGES_DIR = Path("data/images_raw/ISIC2018_Task3_Training_Input")
CSV_PATH   = Path("data/labels_raw/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
OUTPUT_DIR = Path("data/yolo")

# ── Class mapping ──────────────────────────────────────────────
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_MAP   = {name: i for i, name in enumerate(CLASS_NAMES)}

CLASS_COUNTS = {
    'MEL': 1113, 'NV': 6705, 'BCC': 514,
    'AKIEC': 327, 'BKL': 1099, 'DF': 115, 'VASC': 142
}
total = sum(CLASS_COUNTS.values())
CLASS_WEIGHTS = {
    CLASS_MAP[k]: round(total / (len(CLASS_COUNTS) * v), 3)
    for k, v in CLASS_COUNTS.items()
}


def saliency_bbox(img):
    """
    Automatically detect lesion region using color saliency.

    Dermoscopy images have a key property:
    - Skin background = low saturation (pinkish/brownish, desaturated)
    - Lesion = high saturation (dark brown, black, red, blue-gray)

    Steps:
    1. Convert BGR → HSV
    2. Extract saturation channel (S)
    3. Also extract value channel inverted (dark regions)
    4. Combine: lesion = high saturation OR very dark
    5. Morphological cleanup → find largest contour → bbox

    Returns: (cx, cy, w, h) normalized, or fallback center crop
    """
    img_h, img_w = img.shape[:2]

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]    # saturation channel
    val = hsv[:, :, 2]    # value channel

    # Lesion mask: high saturation OR very dark pixels
    _, sat_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask   = (val < 80).astype(np.uint8) * 255

    combined = cv2.bitwise_or(sat_mask, dark_mask)

    # Morphological cleanup — remove noise, fill holes
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel)

    # Find contours — take the largest one (the lesion)
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # Fallback: center 70% crop
        return 0.5, 0.5, 0.7, 0.7

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    # Reject if too small (< 1% of image) or too large (> 95%)
    img_area = img_w * img_h
    if area < 0.01 * img_area or area > 0.95 * img_area:
        return 0.5, 0.5, 0.7, 0.7

    x, y, w, h = cv2.boundingRect(largest)

    # Add 8% padding
    pad_x = int(w * 0.08)
    pad_y = int(h * 0.08)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img_w - x, w + 2 * pad_x)
    h = min(img_h - y, h + 2 * pad_y)

    # Normalize to 0-1
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h

    return cx, cy, nw, nh


def prepare():
    print("Reading CSV labels...")
    df = pd.read_csv(CSV_PATH)
    df['class_id']   = df[CLASS_NAMES].values.argmax(axis=1)
    df['class_name'] = df['class_id'].apply(lambda x: CLASS_NAMES[x])

    print(f"Total images: {len(df)}")
    print("\nClass distribution:")
    for name in CLASS_NAMES:
        count  = (df['class_name'] == name).sum()
        weight = CLASS_WEIGHTS[CLASS_MAP[name]]
        print(f"  {name:<8} {count:>5} images  weight={weight:.3f}")

    # Stratified train/val split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['class_id']
    )
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

    # Create output dirs
    for split in ['train', 'val']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    stats = {
        'processed': 0, 'saliency_good': 0,
        'fallback': 0,   'skipped': 0
    }

    for split, split_df in [('train', train_df), ('val', val_df)]:
        print(f"\nProcessing {split} ({len(split_df)} images)...")
        count = 0

        for _, row in split_df.iterrows():
            img_id   = row['image']
            class_id = row['class_id']

            img_src = IMAGES_DIR / f"{img_id}.jpg"
            if not img_src.exists():
                stats['skipped'] += 1
                continue

            img = cv2.imread(str(img_src))
            if img is None:
                stats['skipped'] += 1
                continue

            # Get bbox via saliency
            cx, cy, w, h = saliency_bbox(img)

            # Track quality
            if cx == 0.5 and cy == 0.5 and w == 0.7:
                stats['fallback'] += 1
            else:
                stats['saliency_good'] += 1

            # Save image resized to 640x640
            img_resized = cv2.resize(img, (640, 640))
            img_dst     = OUTPUT_DIR / split / 'images' / f"{img_id}.jpg"
            cv2.imwrite(str(img_dst), img_resized,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save YOLO label
            label_dst = OUTPUT_DIR / split / 'labels' / f"{img_id}.txt"
            with open(label_dst, 'w') as f:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            stats['processed'] += 1
            count += 1

            if count % 500 == 0:
                print(f"  {count}/{len(split_df)} done...")

        print(f"  ✅ {split} complete!")

    # Save YAML
    weights_str = ' '.join([str(CLASS_WEIGHTS[i]) for i in range(7)])
    yaml_content = f"""path: {OUTPUT_DIR.absolute().as_posix()}
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
"""
    with open("data/skin.yaml", 'w') as f:
        f.write(yaml_content)

    # Quick visual check — save 5 sample images with bbox drawn
    print("\nGenerating bbox preview images...")
    os.makedirs("outputs/bbox_preview", exist_ok=True)
    sample_df = val_df.head(5)

    for _, row in sample_df.iterrows():
        img_id   = row['image']
        class_id = row['class_id']

        img_path   = OUTPUT_DIR / 'val' / 'images' / f"{img_id}.jpg"
        label_path = OUTPUT_DIR / 'val' / 'labels' / f"{img_id}.txt"

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        with open(label_path) as f:
            parts = f.read().strip().split()
            cx_n, cy_n, w_n, h_n = map(float, parts[1:])

        # Convert back to pixel coords
        cx_px = int(cx_n * w)
        cy_px = int(cy_n * h)
        bw    = int(w_n * w)
        bh    = int(h_n * h)
        x1    = cx_px - bw // 2
        y1    = cy_px - bh // 2
        x2    = cx_px + bw // 2
        y2    = cy_px + bh // 2

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, CLASS_NAMES[class_id], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        out_path = f"outputs/bbox_preview/{img_id}_preview.jpg"
        cv2.imwrite(out_path, img)

    print(f"\n{'='*48}")
    print(f"  Processed        : {stats['processed']}")
    print(f"  Saliency bbox    : {stats['saliency_good']}")
    print(f"  Fallback bbox    : {stats['fallback']}")
    print(f"  Skipped          : {stats['skipped']}")
    print(f"  YAML saved       : data/skin.yaml")
    print(f"  Preview images   : outputs/bbox_preview/")
    print(f"{'='*48}")
    print("\n✅ Dataset ready!")


if __name__ == '__main__':
    prepare()