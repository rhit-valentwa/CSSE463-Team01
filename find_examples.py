import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from regression_train import CocoMMapCropDataset, UNetColorizer

import numpy as np
from skimage.color import lab2rgb

# --------------------
# Config (MUST MATCH TRAINING)
# --------------------
SHARD_DIR = Path("data/coco/train2017_cache_256_mmap")
CHECKPOINT_PATH = "unet_colorizer_best.pt"

BASE_CH = 32
BATCH_SIZE = 1
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_GOOD = 5
NUM_BAD = 5

OUT_DIR = Path("colorization_examples")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------
# Load model
# --------------------
def load_model():
    model = UNetColorizer(base=BASE_CH).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model


# --------------------
# Dataset
# --------------------
def get_loader():
    ds = CocoMMapCropDataset(SHARD_DIR)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

def to_rgb(L01, ab11):
    # L01: [1,H,W] float in [0,1]
    # ab11: [2,H,W] float in [-1,1] (gt) or model pred (tanh)

    L = (L01.clamp(0, 1) * 100.0)          # -> [0,100]
    ab = (ab11.clamp(-1, 1) * 128.0)       # -> roughly [-128,128]

    Lab = torch.cat([L, ab], dim=0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    rgb = lab2rgb(Lab)                     # [H,W,3] in [0,1]
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


# --------------------
# Find top-K best and worst examples (STREAMING)
# --------------------
@torch.no_grad()
def find_examples(model, loader):
    best = []   # list of (mae, L, ab, pred, idx)
    worst = []

    for idx, (L, ab) in enumerate(loader):
        L = L.to(DEVICE, non_blocking=True)
        ab = ab.to(DEVICE, non_blocking=True)

        pred = model(L)
        mae = (pred - ab).abs().mean().item()

        entry = (
            mae,
            L.squeeze(0).cpu(),
            ab.squeeze(0).cpu(),
            pred.squeeze(0).cpu(),
            idx,
        )

        # ---- best K (lowest MAE)
        best.append(entry)
        best.sort(key=lambda x: x[0])
        if len(best) > NUM_GOOD:
            best.pop()

        # ---- worst K (highest MAE)
        worst.append(entry)
        worst.sort(key=lambda x: -x[0])
        if len(worst) > NUM_BAD:
            worst.pop()

        if idx % 1000 == 0 and idx > 0:
            print(f"Processed {idx} images")

    return best, worst


# --------------------
# Save images
# --------------------
def save_examples(examples, prefix):
    for i, (mae, L, ab, pred, idx) in enumerate(examples):
        save_image(L, OUT_DIR / f"{prefix}_{i}_L.png")
        save_image(to_rgb(L, ab),   OUT_DIR / f"{prefix}_{i}_gt_rgb.png")
        save_image(to_rgb(L, pred), OUT_DIR / f"{prefix}_{i}_pred_rgb.png")

        print(f"{prefix.upper()} {i}: idx={idx}, MAE={mae:.5f}")

# --------------------
# Optional plotting
# --------------------
def plot_examples(examples, title):
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    for i, (mae, L, ab, pred, _) in enumerate(examples):
        axes[i, 0].imshow(L[0], cmap="gray")
        axes[i, 0].set_title("L")

        axes[i, 1].imshow(to_rgb(L, ab).permute(1, 2, 0))
        axes[i, 1].set_title("GT RGB")

        axes[i, 2].imshow(to_rgb(L, pred).permute(1, 2, 0))
        axes[i, 2].set_title(f"Pred RGB\nMAE={mae:.4f}")


        for j in range(3):
            axes[i, j].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# --------------------
# Main
# --------------------
def main():
    model = load_model()
    loader = get_loader()

    good, bad = find_examples(model, loader)

    save_examples(good, "good")
    save_examples(bad, "bad")

    # Comment out if running headless
    plot_examples(good, "Good Colorizations")
    plot_examples(bad, "Bad Colorizations")

    print(f"\nSaved results to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
