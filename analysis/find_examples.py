import sys
import torch
#import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
#from torchvision.utils import save_image, make_grid

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image
from unet_train import CocoMMapCropDataset, UNetColorizer

import numpy as np
from skimage.color import lab2rgb
import random
import csv

# Config (MUST MATCH TRAINING)

# Prefer val/test shards for evaluation (change if needed)
SHARD_DIR = _PROJECT_ROOT / "data" / "coco" / "train2017_cache_256_mmap"
CHECKPOINT_PATH = str(_PROJECT_ROOT / "data" / "models" / "unet_colorizer_best.pt")

BASE_CH = 32
BATCH_SIZE = 1
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# How many examples to save
NUM_GOOD = 5
NUM_BAD = 6

# How much of the dataset to scan (keeps this script fast/repeatable)
SCAN_LIMIT = 2000

# Frozen evaluation set support (optional, but great for "stability")
EVAL_COUNT = 20
EVAL_INDICES_PATH = Path("qualitative_outputs/eval_indices.txt")

# Output folders/files
OUT_DIR = Path("colorization_examples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUAL_DIR = Path("qualitative_outputs")
QUAL_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = QUAL_DIR / "summary.csv"

# Make runs repeatable
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Model
def load_model():
    model = UNetColorizer(base=BASE_CH).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model

# Dataset
def get_loader():
    ds = CocoMMapCropDataset(SHARD_DIR)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

# Utilities
def to_rgb(L01, ab11):
    """
    L01:  [1,H,W] float in [0,1]
    ab11: [2,H,W] float in [-1,1] (gt) or model pred (tanh)
    """
    L = (L01.clamp(0, 1) * 100.0)      # -> [0,100]
    ab = (ab11.clamp(-1, 1) * 128.0)   # -> roughly [-128,128]
    Lab = torch.cat([L, ab], dim=0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    rgb = lab2rgb(Lab)                # [H,W,3] in [0,1]
    return torch.from_numpy(rgb).permute(2, 0, 1).float()

def baseline_pred(ab_gt):
    """Baseline: predict zero chroma (gray) for all pixels."""
    return torch.zeros_like(ab_gt)

def load_eval_indices():
    if EVAL_INDICES_PATH.exists():
        idxs = [
            int(x.strip())
            for x in EVAL_INDICES_PATH.read_text().splitlines()
            if x.strip()
        ]
        return idxs
    return None

def save_eval_indices(idxs):
    EVAL_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_INDICES_PATH.write_text("\n".join(map(str, idxs)) + "\n")

def append_summary_row(row):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["group", "rank", "dataset_idx", "mae_baseline", "mae_cnn", "delta"])
        w.writerow(row)

def save_panel(L, ab_gt, ab_pred, ab_base, out_path):
    """
    Save a 1x4 panel:
      grayscale | baseline | cnn | ground truth
    """
    img_L = L.repeat(3, 1, 1)  # 3-channel grayscale for consistent grid
    img_base = to_rgb(L, ab_base)
    img_pred = to_rgb(L, ab_pred)
    img_gt = to_rgb(L, ab_gt)

    panel = torch.cat([img_L, img_base, img_pred, img_gt], dim=2)
    save_image(panel, out_path)

def save_image(img, path):
    """
    Save a tensor image to disk without torchvision.
    img: [C,H,W] in [0,1] float tensor
    """
    img = img.detach().cpu().clamp(0, 1)

    # Ensure 3 channels for saving
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    arr = (img * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)

# Core: Find examples (streaming)
@torch.no_grad()
def find_examples(model, loader):
    """
    Returns:
      best:  list of (mae_cnn, mae_base, L, ab_gt, ab_pred, idx) with lowest mae_cnn
      worst: list of (mae_cnn, mae_base, L, ab_gt, ab_pred, idx) with highest mae_cnn
    """
    best = []
    worst = []

    for idx, (L, ab) in enumerate(loader):
        if idx >= SCAN_LIMIT:
            break

        L = L.to(DEVICE, non_blocking=True)
        ab = ab.to(DEVICE, non_blocking=True)

        pred = model(L)
        mae_cnn = (pred - ab).abs().mean().item()

        pred_base = torch.zeros_like(ab)  # baseline in normalized ab-space
        mae_base = (pred_base - ab).abs().mean().item()

        entry = (
            mae_cnn,
            mae_base,
            L.squeeze(0).cpu(),
            ab.squeeze(0).cpu(),
            pred.squeeze(0).cpu(),
            idx,
        )

        # Keep top-K best (lowest CNN MAE)
        best.append(entry)
        if len(best) > NUM_GOOD:
            best.sort(key=lambda x: x[0])
            best = best[:NUM_GOOD]

        # Keep top-K worst (highest CNN MAE)
        worst.append(entry)
        if len(worst) > NUM_BAD:
            worst.sort(key=lambda x: -x[0])
            worst = worst[:NUM_BAD]

        if idx % 250 == 0 and idx > 0:
            print(f"Processed {idx} images...")

    best.sort(key=lambda x: x[0])
    worst.sort(key=lambda x: -x[0])
    return best, worst

# Frozen evaluation set (optional but powerful)
@torch.no_grad()
def build_or_load_eval_set(model, loader):
    """
    Creates a fixed list of dataset indices the first time it runs,
    then reuses it on future runs for stable comparisons.
    """
    idxs = load_eval_indices()
    if idxs is not None and len(idxs) > 0:
        print(f"Loaded frozen eval set with {len(idxs)} indices from {EVAL_INDICES_PATH}")
        return idxs

    # Otherwise, generate one from the first SCAN_LIMIT samples
    candidates = list(range(min(SCAN_LIMIT, EVAL_COUNT * 200)))  # simple pool
    random.shuffle(candidates)
    idxs = sorted(candidates[:EVAL_COUNT])
    save_eval_indices(idxs)
    print(f"Created frozen eval set with {len(idxs)} indices at {EVAL_INDICES_PATH}")
    return idxs

@torch.no_grad()
def eval_on_indices(model, loader, idxs):
    """
    Evaluate only on specific dataset indices (idxs).
    Returns list of entries:
      (mae_cnn, mae_base, L, ab_gt, ab_pred, idx)
    """
    idx_set = set(idxs)
    results = []

    for idx, (L, ab) in enumerate(loader):
        if idx > max(idxs):
            break
        if idx not in idx_set:
            continue

        L = L.to(DEVICE, non_blocking=True)
        ab = ab.to(DEVICE, non_blocking=True)

        pred = model(L)
        mae_cnn = (pred - ab).abs().mean().item()

        pred_base = torch.zeros_like(ab)
        mae_base = (pred_base - ab).abs().mean().item()

        results.append((
            mae_cnn,
            mae_base,
            L.squeeze(0).cpu(),
            ab.squeeze(0).cpu(),
            pred.squeeze(0).cpu(),
            idx,
        ))

    # Sort by CNN MAE
    results.sort(key=lambda x: x[0])
    return results

# Save examples
def save_examples(examples, prefix):
    for i, (mae_cnn, mae_base, L, ab, pred, idx) in enumerate(examples):
        base = baseline_pred(ab)

        # Save individual components (optional)
        save_image(L, OUT_DIR / f"{prefix}_{i}_idx{idx}_L.png")
        save_image(to_rgb(L, ab), OUT_DIR / f"{prefix}_{i}_idx{idx}_gt_rgb.png")
        save_image(to_rgb(L, pred), OUT_DIR / f"{prefix}_{i}_idx{idx}_pred_rgb.png")
        save_image(to_rgb(L, base), OUT_DIR / f"{prefix}_{i}_idx{idx}_base_rgb.png")

        # Save panel (BEST for report)
        panel_path = OUT_DIR / f"{prefix}_{i}_idx{idx}_panel.png"
        save_panel(L, ab, pred, base, panel_path)

        append_summary_row([
            prefix,
            i,
            idx,
            f"{mae_base:.6f}",
            f"{mae_cnn:.6f}",
            f"{(mae_base - mae_cnn):.6f}",
        ])

        print(
            f"{prefix.upper()} {i}: idx={idx}, "
            f"baseline={mae_base:.5f}, cnn={mae_cnn:.5f}, "
            f"gain={mae_base - mae_cnn:.5f}"
        )

# Optional plotting (saves to file; headless-safe)
#def plot_examples(examples, title, out_path):
#    n = len(examples)
#    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
#    if n == 1:
#        axes = np.expand_dims(axes, 0)

#    for i, (mae_cnn, mae_base, L, ab, pred, idx) in enumerate(examples):
#        base = baseline_pred(ab)

#        axes[i, 0].imshow(L[0], cmap="gray")
#        axes[i, 0].set_title(f"L (idx={idx})")

#       axes[i, 1].imshow(to_rgb(L, base).permute(1, 2, 0))
#       axes[i, 1].set_title(f"Baseline\nMAE={mae_base:.4f}")

#       axes[i, 2].imshow(to_rgb(L, pred).permute(1, 2, 0))
#       axes[i, 2].set_title(f"CNN\nMAE={mae_cnn:.4f}")

#        axes[i, 3].imshow(to_rgb(L, ab).permute(1, 2, 0))
#        axes[i, 3].set_title("GT")
#
#        for j in range(4):
#            axes[i, j].axis("off")

#    fig.suptitle(title)
#    plt.tight_layout()
#    fig.savefig(out_path, dpi=200)
#    plt.close(fig)

# Main
def main():
    model = load_model()
    loader = get_loader()

    # 1) Find best/worst within a limited scan (quick)
    good, bad = find_examples(model, loader)
    save_examples(good, "good")
    save_examples(bad, "bad")
    #plot_examples(good, "Good Colorizations (Gray | Baseline | CNN | GT)", OUT_DIR / "good_grid.png")
    #plot_examples(bad, "Bad Colorizations (Gray | Baseline | CNN | GT)", OUT_DIR / "bad_grid.png")

    # 2) Frozen eval set (repeatable stability evidence)
    loader2 = get_loader()  # fresh loader
    eval_idxs = build_or_load_eval_set(model, loader2)

    loader3 = get_loader()
    eval_results = eval_on_indices(model, loader3, eval_idxs)

    # Save top/bottom within the frozen set too
    save_examples(eval_results[:min(10, len(eval_results))], "eval_best")
    save_examples(list(reversed(eval_results[-min(10, len(eval_results)):])), "eval_worst")
    #plot_examples(eval_results[:min(10, len(eval_results))],
                  #"Frozen Eval Set - Best (Gray | Baseline | CNN | GT)",
                  #OUT_DIR / "eval_best_grid.png")

    print(f"\nSaved results to:\n- {OUT_DIR.resolve()}\n- {SUMMARY_CSV.resolve()}")
    print(f"Frozen eval indices at: {EVAL_INDICES_PATH.resolve()}")

if __name__ == "__main__":
    main()

