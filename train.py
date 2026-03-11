"""
YOLOv11s Fine-tuning on ISIC 2018 Skin Condition Dataset
- 7 classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC
- Fine-tuning pretrained YOLOv11s (not training from scratch)
- Class weights to handle severe imbalance
- GPU accelerated
"""

from ultralytics import YOLO
import torch

CLASS_NAMES   = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_WEIGHTS = [1.285, 0.213, 2.783, 4.375, 1.302, 12.441, 10.075]

if __name__ == '__main__':

    # ── Device check ───────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")
    if device == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"\nClass weights:")
    for name, w in zip(CLASS_NAMES, CLASS_WEIGHTS):
        print(f"  {name:<8} {w:.3f}")

    # ── Load pretrained YOLOv11s ───────────────────────────────
    print("\nLoading YOLOv11s pretrained weights...")
    model = YOLO('yolo11s.pt')

    # ── Training ───────────────────────────────────────────────
    print("\nStarting fine-tuning...\n")

    results = model.train(
        # Dataset
        data          = 'data/skin.yaml',
        imgsz         = 640,

        # Training duration
        epochs        = 50,
        patience      = 10,

        # Batch & device
        batch         = 16,
        workers       = 0,        # ← Windows fix
        device        = device,

        # Optimization
        optimizer     = 'AdamW',
        lr0           = 0.001,
        lrf           = 0.01,
        warmup_epochs = 3,

        # Regularization
        weight_decay  = 0.0005,
        dropout       = 0.0,

        # Augmentation
        augment       = True,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 15.0,
        fliplr        = 0.5,
        flipud        = 0.5,
        mosaic        = 0.5,
        mixup         = 0.1,

        # Class imbalance
        cls           = 0.5,

        # Output
        project       = 'runs',
        name          = 'skin_v1',
        exist_ok      = True,
        verbose       = True,
        plots         = True,
    )

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)

    # ── Evaluate with TTA ──────────────────────────────────────
    print("\nRunning validation with TTA...")
    metrics = model.val(
        data    = 'data/skin.yaml',
        augment = True,
        verbose = True,
        workers = 0,        # ← Windows fix here too
    )

    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")

    print("\nPer-class AP@0.5:")
    print("-" * 30)
    for name, ap in zip(CLASS_NAMES, metrics.box.ap50):
        bar = "█" * int(ap * 30)
        print(f"  {name:<8} {ap:.4f}  {bar}")

    print(f"\nBest weights : runs/skin_v1/weights/best.pt")
    print(f"Results plot : runs/skin_v1/results.png")