#!/usr/bin/env python3
"""
visualize_colorization_samples.py

Visualize N sample colorization results from a trained CNN.
Outputs: Grayscale | Model Prediction | Ground Truth

Saves:
  - colorization_samples/grid_gray_pred_gt.png
  - colorization_samples/sample_XX_grayscale.png
  - colorization_samples/sample_XX_prediction.png
  - colorization_samples/sample_XX_ground_truth.png

Usage examples:
  python visualize_colorization_samples.py
  python visualize_colorization_samples.py --num-samples 10 --micro-batch 1
  python visualize_colorization_samples.py --device cpu
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.color import lab2rgb
from torch.utils.data import Subset

# Import your project components
from regression_train import (
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
    Converts to real Lab units for skimage and returns RGB in [0,1].
    """
    # Convert normalized -> real Lab units for skimage
    L = L * 100.0
    ab = ab * 128.0

    lab = torch.cat([L, ab], dim=1)  # (N,3,H,W)
    lab_np = lab.permute(0, 2, 3, 1).detach().cpu().numpy()  # (N,H,W,3)

    from skimage.color import lab2rgb
    rgb_images = []
    for img in lab_np:
        rgb = lab2rgb(img)  # float in [0,1]
        rgb_images.append(rgb.astype(np.float32))
    return np.stack(rgb_images, axis=0)


@torch.inference_mode()
def visualize_samples(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_samples: int = 10,
    save_dir: str = "colorization_samples",
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

    # Save individual sample triplets
    for i in range(num_samples):
        gray_bgr = cv2.cvtColor((grayscale_rgb[i] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor((pred_rgb[i] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        gt_bgr = cv2.cvtColor((gt_rgb[i] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(save_path / f"sample_{i+1:02d}_grayscale.png"), gray_bgr)
        cv2.imwrite(str(save_path / f"sample_{i+1:02d}_prediction.png"), pred_bgr)
        cv2.imwrite(str(save_path / f"sample_{i+1:02d}_ground_truth.png"), gt_bgr)

    # Save a single grid image: N rows × 3 cols
    H, W = grayscale_rgb.shape[1], grayscale_rgb.shape[2]
    grid = np.zeros((num_samples * H, 3 * W, 3), dtype=np.uint8)

    for i in range(num_samples):
        r0, r1 = i * H, (i + 1) * H
        grid[r0:r1, 0:W] = (grayscale_rgb[i] * 255).clip(0, 255).astype(np.uint8)
        grid[r0:r1, W:2 * W] = (pred_rgb[i] * 255).clip(0, 255).astype(np.uint8)
        grid[r0:r1, 2 * W:3 * W] = (gt_rgb[i] * 255).clip(0, 255).astype(np.uint8)

    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path / "grid_gray_pred_gt.png"), grid_bgr)

    print(f"Saved {num_samples} samples to: {save_path}")
    print(f"Saved grid to: {save_path / 'grid_gray_pred_gt.png'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="cnn_colorizer_best.pt",
                   help="Path to checkpoint (expects checkpoint['model']).")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--save-dir", type=str, default="colorization_samples")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=2,
                   help="Inference micro-batch size to reduce VRAM.")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision on CUDA.")
    p.add_argument("--device", type=str, default=None,
                   help="Override device: 'cpu', 'cuda', 'cuda:0', etc. Default uses regression_train.DEVICE")
    return p.parse_args()


def main():
    args = parse_args()

    # Choose device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        # DEFAULT_DEVICE comes from regression_train.py
        device = DEFAULT_DEVICE if isinstance(DEFAULT_DEVICE, torch.device) else torch.device(DEFAULT_DEVICE)

    print(f"Using device: {device}")

    # Load test data
    ds = CocoMMapCropDataset(SHARD_DIR, max_shards=None)
    n = len(ds)
    train_idx, val_idx, test_idx = make_split_indices(n, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    test_ds = Subset(ds, test_idx)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test set: {len(test_idx)} samples")

    # Load model + checkpoint
    model = ColorizationCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint {args.checkpoint} missing key 'model'. Found keys: {list(checkpoint.keys())}")
    model.load_state_dict(checkpoint["model"], strict=True)

    # If CUDA is crowded, this helps reduce fragmentation (optional)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Generating visualizations...")
    visualize_samples(
        model=model,
        test_loader=test_loader,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        infer_device=device,
        micro_batch=max(1, args.micro_batch),
        use_amp=(device.type == "cuda" and (not args.no_amp)),
    )
    print("Done.")


if __name__ == "__main__":
    main()