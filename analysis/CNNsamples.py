#!/usr/bin/env python3
"""
CNNsamples.py

Visualize N sample colorization results from a trained CNN.
Outputs: Grayscale | Model Prediction | Ground Truth

Saves:
  - colorization_samples_chromaloss/grid_gray_pred_gt.png
  - colorization_samples_chromaloss/sample_XX_grayscale.png
  - colorization_samples_chromaloss/sample_XX_prediction.png
  - colorization_samples_chromaloss/sample_XX_ground_truth.png

Usage examples:
  python CNNsamples.py
  python CNNsamples.py --num-samples 10 --micro-batch 1
  python CNNsamples.py --device cpu
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from torch.utils.data import Subset

# Import your project components
from unet_train import (
    CocoMMapCropDataset,
    make_split_indices,
    SHARD_DIR,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    DEVICE as DEFAULT_DEVICE,
)
from CNNtest import ColorizationCNN


def lab_to_rgb(L, ab):
    """
    L:  (N,1,H,W) in [0,1]
    ab: (N,2,H,W) in [-1,1]

    Converts to RGB using OpenCV instead of skimage.
    Returns RGB float images in [0,1].
    """
    # Convert normalized → real Lab units
    L = L * 100.0
    ab = ab * 128.0

    lab = torch.cat([L, ab], dim=1)  # (N,3,H,W)
    lab_np = lab.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)  # (N,H,W,3)

    rgb_images = []

    for img in lab_np:
        # OpenCV expects:
        # L in [0,100]
        # a,b in [-128,127]
        # float32 input
        rgb = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
        rgb = np.clip(rgb, 0, 1)
        rgb_images.append(rgb)

    return np.stack(rgb_images, axis=0)



@torch.inference_mode()
def visualize_samples(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_samples: int = 10,
    save_dir: str = "colorization_samples_chromaloss",
    infer_device: torch.device | None = None,
    micro_batch: int = 2,
    use_amp: bool = True,
):
    """
    Generate and save visualization of colorization results.
    Uses micro-batching to reduce VRAM usage.
    """
    model.eval()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Collect exactly num_samples from loader
    L_list, ab_list = [], []
    got = 0
    for L, ab in test_loader:
        take = min(L.size(0), num_samples - got)
        L_list.append(L[:take])
        ab_list.append(ab[:take])
        got += take
        if got >= num_samples:
            break

    if got < num_samples:
        raise RuntimeError(f"Could only collect {got} samples, requested {num_samples}.")

    L_all = torch.cat(L_list, dim=0)    # (N,1,H,W)
    ab_all = torch.cat(ab_list, dim=0)  # (N,2,H,W)

    if infer_device is None:
        infer_device = next(model.parameters()).device

    # Predict ab in micro-batches to avoid OOM
    pred_chunks = []
    for i in range(0, num_samples, micro_batch):
        L_mb = L_all[i : i + micro_batch].to(infer_device, non_blocking=True)

        if infer_device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_ab_mb = model(L_mb)
        else:
            pred_ab_mb = model(L_mb)

        pred_chunks.append(pred_ab_mb.detach().cpu())

    pred_ab = torch.cat(pred_chunks, dim=0)  # on CPU, (N,2,H,W)

    # Convert to RGB
    L_cpu = L_all.cpu()
    grayscale_rgb = lab_to_rgb(L_all, torch.zeros_like(pred_ab))
    pred_rgb = lab_to_rgb(L_cpu, pred_ab)
    gt_rgb = lab_to_rgb(L_cpu, ab_all)

    # Save one combined image per sample:
    # [ grayscale | prediction | ground truth ]
    for i in range(num_samples):

        gray = (grayscale_rgb[i] * 255).clip(0, 255).astype(np.uint8)
        pred = (pred_rgb[i] * 255).clip(0, 255).astype(np.uint8)
        gt   = (gt_rgb[i] * 255).clip(0, 255).astype(np.uint8)

        # Concatenate horizontally
        combined = np.concatenate([gray, pred, gt], axis=1)

        # Convert RGB → BGR for OpenCV saving
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(save_path / f"sample_{i+1:02d}.png"), combined_bgr)

    print(f"Saved {num_samples} combined samples to: {save_path}")
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(_PROJECT_ROOT / "data" / "models" / "cnn_colorizer_best.pt"))
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--save-dir", type=str, default="colorization_samples_chromaloss")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--device", type=str, default=None,
                   help="cuda, cuda:0, cpu, etc.")
    return p.parse_args()


def main():
    args = parse_args()

    # Device selection (GPU!)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    # Load dataset
    ds = CocoMMapCropDataset(SHARD_DIR, max_shards=None)
    n = len(ds)

    train_idx, val_idx, test_idx = make_split_indices(
        n, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    test_ds = Subset(ds, test_idx)

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test set size: {len(test_idx)}")

    # Load model
    model = ColorizationCNN().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model" not in checkpoint:
        raise KeyError("Checkpoint missing 'model' key")

    model.load_state_dict(checkpoint["model"], strict=True)

    print("Model loaded successfully.")
    print("Generating visualizations...")

    # Run visualization
    visualize_samples(
        model=model,
        test_loader=test_loader,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        infer_device=device,
        micro_batch=max(1, args.micro_batch),
        use_amp=(device.type == "cuda"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
