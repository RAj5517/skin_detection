# YOLOv11s Skin Lesion Detection — ISIC 2018

Fine-tuning YOLOv11s for dermoscopic skin lesion detection and classification on the ISIC 2018 dataset. Includes saliency-based lesion localization, class imbalance handling, and Grad-CAM++ explainability.

[![HuggingFace](https://img.shields.io/badge/HuggingFace-raj5517%2Fyolov11s--skin--lesion--isic2018-yellow)](https://huggingface.co/raj5517/yolov11s-skin-lesion-isic2018)

---

## Results

| Metric | v1 (50 epochs) | v2 (80 epochs + cos_lr) |
|--------|---------------|--------------------------|
| mAP@0.5 | 0.551 | **0.603** |
| mAP@0.5:0.95 | 0.473 | **0.526** |
| Precision | 0.486 | 0.541 |
| Recall | 0.585 | 0.595 |
| Inference speed | — | 5.2ms/image (~190 FPS) |

### Per-class AP@0.5 (v2)

| Class | Full Name | Samples | AP@0.5 |
|-------|-----------|---------|--------|
| MEL | Melanoma | 1113 | 0.546 |
| NV | Melanocytic Nevus | 6705 | 0.956 |
| BCC | Basal Cell Carcinoma | 514 | 0.556 |
| AKIEC | Actinic Keratosis | 327 | 0.441 |
| BKL | Benign Keratosis | 1099 | 0.569 |
| DF | Dermatofibroma | 23 | 0.200 |
| VASC | Vascular Lesion | 142 | 0.850 |

---

## Architecture

```
Input (640x640)
      |
YOLOv11s Backbone
  Conv x4 + C3k2 x4          <- feature extraction
  SPPF + C2PSA               <- spatial pyramid pooling + attention
      |
FPN Neck
  Upsample + Concat x2       <- multi-scale feature fusion
      |
Detection Head
  Detect [7 classes]         <- classification + bbox regression
      |
Output: bbox + class + confidence
```

- **Parameters**: 9.4M
- **GFLOPs**: 21.6
- **Pretrained on**: COCO (80 classes) → fine-tuned on ISIC 2018 (7 classes)

---

## Dataset

[ISIC 2018 Task 3](https://challenge.isic-archive.com/data/#2018) — 10,015 dermoscopy images, 7 classes.

**Severe class imbalance:**
```
NV    : 6705  (67%)   ← dominant
MEL   : 1113  (11%)
BKL   : 1099  (11%)
BCC   :  514  ( 5%)
AKIEC :  327  ( 3%)
VASC  :  142  ( 1%)
DF    :   23  ( 0.2%) ← rarest
```

Split: 80% train (8,012) / 20% val (2,003), stratified by class.

---

## Methodology

### Saliency-Based Bounding Box Generation

ISIC 2018 Task 3 provides no bounding box annotations — only image-level labels. Ground truth segmentation masks from Task 1 use a different image ID series and are incompatible.

Solution: HSV-based saliency detection to generate approximate lesion bboxes:

```python
# 1. Convert to HSV
# 2. Extract saturation channel (lesions are more saturated than skin)
# 3. Detect dark pixels (melanotic lesions)
# 4. Combine with Otsu threshold
# 5. Morphological cleanup (close + open)
# 6. Find largest contour -> bounding box + 8% padding
# Fallback: center 70% crop if no contour found
```

- Success rate: 9,962/10,015 (99.5%)
- Fallback (center crop): 53 images

### Class Imbalance Handling

Inverse-frequency weights computed per class:

```
DF:    12.441x  |  VASC: 10.075x  |  AKIEC: 4.375x
BCC:    2.783x  |  BKL:   1.302x  |  MEL:   1.285x
NV:     0.213x  (down-weighted, dominant class)
```

### Training Configuration

```python
model     = YOLO('yolo11s.pt')       # pretrained on COCO
epochs    = 80
optimizer = 'AdamW'
lr0       = 0.001
cos_lr    = True                     # cosine LR annealing
batch     = 32
imgsz     = 640
patience  = 15                       # early stopping
```

**Augmentation:**
- HSV jitter (hue 0.015, saturation 0.7, value 0.4)
- Rotation ±15°, horizontal + vertical flip
- Mosaic (p=0.5), MixUp (p=0.05)
- Albumentations: Blur, MedianBlur, CLAHE, ToGray

---

## Explainability — Grad-CAM++

Manual Grad-CAM++ implementation using PyTorch hooks on backbone layer 8 (C3k2), bypassing the detection head to avoid inference tensor conflicts.

```
Backbone layer 8 (C3k2)
    |
Forward hook  -> capture activations
Backward hook -> capture gradients
    |
alpha = mean(gradients²)     <- Grad-CAM++ weighting
CAM   = ReLU(sum(alpha * activations))
```

### Discovered Attention Patterns

**Border Ring Detection** — on well-defined lesions, the model focuses on the lesion perimeter, aligning with clinical dermoscopy criteria where border irregularity is a primary diagnostic indicator. Learned without explicit border supervision.

**Multi-focal Pigment Tracking** — on irregular lesions, attention distributes across multiple pigment-dense sub-regions simultaneously, mirroring dermatologist assessment of pigment distribution patterns.

These clinically meaningful features emerged from detection training alone — no segmentation masks, no border annotations.

---

## Project Structure

```
skin_detection/
├── data/
│   ├── prepare_dataset.py     # saliency bbox generation + YOLO format
│   └── skin.yaml              # dataset config
├── weights/
│   └── best_v2.pt             # best model (download from HuggingFace)
├── outputs/
│   └── gradcam/               # Grad-CAM++ visualizations
├── runs/                      # training logs + plots
├── train.py                   # YOLOv11s fine-tuning
├── gradcam.py                 # Grad-CAM++ explainability
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/RAj5517/skin_detection
cd skin_detection
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Download weights from HuggingFace:

```bash
huggingface-cli download raj5517/yolov11s-skin-lesion-isic2018 best_v2.pt --local-dir weights/
```

---

## Inference

```python
from ultralytics import YOLO

model   = YOLO('weights/best_v2.pt')
results = model('your_image.jpg')
results[0].show()
```

---

## Run Grad-CAM

```bash
python gradcam.py
# outputs saved to outputs/gradcam/
```

---

## Training From Scratch

1. Download ISIC 2018 Task 3 from https://challenge.isic-archive.com/data/#2018
2. Place images in `data/images_raw/` and ground truth CSV in `data/labels_raw/`
3. Run dataset preparation:
```bash
python data/prepare_dataset.py
```
4. Train:
```bash
python train.py
```

---

## Limitations

- **DF** (23 training samples) and **AKIEC** (65 samples) are data-starved — performance bounded by dataset size, not model capacity
- Saliency-based bboxes are approximate — not ground-truth segmentation
- Validated on dermoscopy images only, not clinical photography
- NV dominance (67%) inflates overall mAP metric

---

## References

- [ISIC 2018 Challenge](https://challenge.isic-archive.com/data/#2018)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Grad-CAM++: Improved Visual Explanations](https://arxiv.org/abs/1710.11063)
- [SimCLR Contrastive Learning](https://arxiv.org/abs/2002.05709)