"""
Grad-CAM Explainability for YOLOv11s Skin Lesion Detection
Manual implementation using forward/backward hooks — no external CAM library
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import os, random

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
MODEL_PATH  = 'weights/best_v2.pt'
VAL_DIR     = 'data/yolo/val/images'
SAVE_DIR    = 'outputs/gradcam'
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────
print("Loading model...")
model       = YOLO(MODEL_PATH)
torch_model = model.model
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_model = torch_model.to(device)
print(f"Device: {device}")

# ── Manual Grad-CAM via hooks ──────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        # Forward hook — captures feature maps
        target_layer.register_forward_hook(self._save_activation)
        # Backward hook — captures gradients
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor):
        self.model.eval()

        # Need gradients on input
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass — run only backbone (layers 0-10)
        x       = input_tensor
        saved   = {}
        for i, layer in enumerate(self.model.model):
            if i > 10:
                break
            if hasattr(layer, 'f'):
                f = layer.f
                if isinstance(f, list):
                    x = [saved[j] if j != -1 else x for j in f]
                elif f != -1:
                    x = saved[f]
            x = layer(x)
            saved[i] = x if not isinstance(x, list) else x

        # Target: mean of all activations (maximize feature response)
        target = x.flatten(2).max(dim=-1)[0].mean()
        self.model.zero_grad()
        target.backward()

        # Grad-CAM formula
        # alpha = global average of gradients per channel
        alpha   = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam     = (alpha * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


# ── Initialize Grad-CAM on layer 10 (C2PSA — deepest semantic) ─
gradcam = GradCAM(torch_model, torch_model.model[8])


# ── Single image processing ────────────────────────────────────
def run_gradcam(img_path, save_path=None):
    # Load + resize
    img_bgr  = cv2.imread(str(img_path))
    img_bgr  = cv2.resize(img_bgr, (640, 640))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0

    # Tensor
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    # Compute CAM
    cam = gradcam.compute(tensor)

    # Resize CAM to 640x640
    cam_resized = cv2.resize(cam, (640, 640))

    # Colormap overlay
    heatmap_raw  = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_raw, cv2.COLOR_BGR2RGB)
    overlay      = (0.5 * img_norm + 0.5 * heatmap_rgb.astype(np.float32)/255.0)
    overlay      = np.clip(overlay, 0, 1)

    # YOLO inference for bounding boxes
    results    = model(str(img_path), verbose=False)[0]
    boxes      = results.boxes
    pred_label = "No detection"

    overlay_draw = (overlay * 255).astype(np.uint8)
    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            cv2.rectangle(overlay_draw, (x1,y1), (x2,y2), (255,255,0), 2)
            cv2.putText(overlay_draw, label, (x1, max(y1-8,15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cls_id     = int(boxes[0].cls[0])
        conf       = float(boxes[0].conf[0])
        pred_label = f"{CLASS_NAMES[cls_id]}  conf={conf:.2f}"

    # Plot 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_rgb);          axes[0].set_title('Original',           fontsize=13); axes[0].axis('off')
    axes[1].imshow(cam_resized, cmap='jet'); axes[1].set_title('Grad-CAM Heatmap',   fontsize=13); axes[1].axis('off')
    axes[2].imshow(overlay_draw);     axes[2].set_title('Overlay + Detection', fontsize=13); axes[2].axis('off')
    fig.suptitle(f'Prediction: {pred_label}', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close()


# ── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    val_images = list(Path(VAL_DIR).glob('*.jpg'))
    samples    = random.sample(val_images, min(8, len(val_images)))

    print(f"\nRunning Grad-CAM on {len(samples)} images...\n")
    for i, img_path in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {img_path.name}")
        run_gradcam(img_path, save_path=f'{SAVE_DIR}/gradcam_{i+1:02d}.png')

    print(f"\n✅ Done! {len(samples)} images saved to {SAVE_DIR}/")