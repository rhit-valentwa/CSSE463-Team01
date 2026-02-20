#!/usr/bin/env python3
# evaluate.py
#
# Evaluates both trained colorization models (UNet and VGG16-CNN) plus a zero
# baseline (predict a*=b*=0) on:
#   1. ALL images in the COCO training cache
#   2. 5 000 randomly-sampled images from the ImageNet test cache
#
# Metrics reported per model/baseline per dataset:
#   - Chroma-Weighted L1  (custom loss from regression_train.py, alpha=6.0)
#   - MAE  (mean absolute error, both channels combined)
#   - MAE_a / MAE_b  (per channel)
#   - MSE  (mean squared error)
#   - RMSE
#   - Normalised Accuracy  = 1 - MAE/2  (ab ∈ [-1,1] → max error per element = 2)
#   - Saturated-pixel MAE  (pixels where gt chroma > SAT_TAU = 0.15)
#   - Mean predicted chroma  vs  Mean GT chroma
#   - Pixel-accuracy @ 0.05, 0.10, 0.20
#
# Usage (run from the demo/ directory):
#   python evaluate.py
#
# All paths are relative to this script's location so the script can be run
# from any directory as long as the repo layout is unchanged.

from __future__ import annotations

import bisect
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models

# CONFIG

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

UNET_CHECKPOINT  = PROJECT_DIR / "data" / "models" / "unet_colorizer_best.pt"
CNN_CHECKPOINT   = PROJECT_DIR / "data" / "models" / "cnn_colorizer_best.pt"

COCO_SHARD_DIR     = PROJECT_DIR / "data" / "coco"    / "train2017_cache_256_mmap"
IMAGENET_SHARD_DIR = PROJECT_DIR / "data" / "ImageNet" / "test_cache_256_mmap"

UNET_BASE_CH = 32

BATCH_SIZE   = 64
NUM_WORKERS  = 8
PIN_MEMORY   = True
USE_AMP      = True
CHANNELS_LAST = True

IMAGENET_SAMPLE_N = 5_000
IMAGENET_SEED     = 42

# Mirrors the training configuration from regression_train.py
CHROMA_LOSS_ALPHA = 6.0
SAT_TAU           = 0.15        # "saturated" threshold in normalised ab ∈ [-1,1]
ACCURACY_THRESHOLDS = (0.05, 0.10, 0.20)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset (memory-mapped LAB shards, compatible with both COCO and ImageNet)

class MMapLabDataset(Dataset):
    """
    Loads paired (L, ab) samples from memory-mapped numpy shards:
      shard_XXXXX_L.npy   (N, 256, 256) uint8
      shard_XXXXX_ab.npy  (N,   2, 256, 256) int8

    Returns:
      L_t  : (1, H, W)  float32 in [0, 1]
      ab_t : (2, H, W)  float32 in [-1, 1]
    """

    def __init__(self, shard_dir: Path):
        shard_dir = Path(shard_dir)
        if not shard_dir.is_dir():
            raise RuntimeError(f"Shard directory not found: {shard_dir.resolve()}")

        L_paths  = sorted(shard_dir.glob("shard_*_L.npy"))
        ab_paths = sorted(shard_dir.glob("shard_*_ab.npy"))

        if not L_paths or not ab_paths:
            raise RuntimeError(f"No shard files found in {shard_dir.resolve()}")
        if len(L_paths) != len(ab_paths):
            raise RuntimeError("Mismatched number of L and ab shard files.")

        self.L_paths  = L_paths
        self.ab_paths = ab_paths

        self._sizes: List[int] = []
        for Lp, abp in zip(L_paths, ab_paths):
            Lm  = np.load(Lp,  mmap_mode="r")
            abm = np.load(abp, mmap_mode="r")
            if Lm.shape[0] != abm.shape[0]:
                raise RuntimeError(f"N mismatch: {Lp.name}")
            self._sizes.append(int(Lm.shape[0]))

        self._offsets: List[int] = [0]
        for n in self._sizes:
            self._offsets.append(self._offsets[-1] + n)
        self._len = self._offsets[-1]

        self._cached_si = None
        self._L_mmap    = None
        self._ab_mmap   = None

    def __len__(self) -> int:
        return self._len

    def _load_shard(self, si: int):
        if self._cached_si != si:
            self._L_mmap    = np.load(self.L_paths[si],  mmap_mode="r")
            self._ab_mmap   = np.load(self.ab_paths[si], mmap_mode="r")
            self._cached_si = si
        return self._L_mmap, self._ab_mmap

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        si = bisect.bisect_right(self._offsets, idx) - 1
        li = idx - self._offsets[si]
        Lm, abm = self._load_shard(si)

        L_t  = torch.from_numpy(Lm[li].astype(np.float32)  / 255.0).unsqueeze(0)  # (1,H,W)
        ab_t = torch.from_numpy(abm[li].astype(np.float32) / 128.0)               # (2,H,W)
        return L_t, ab_t


# Model architectures (must match training code exactly)

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetColorizer(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base,     base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.pool      = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.dec4 = ConvBlock(base * 16 + base * 8, base * 8)
        self.dec3 = ConvBlock(base * 8  + base * 4, base * 4)
        self.dec2 = ConvBlock(base * 4  + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2  + base,     base)

        self.out = nn.Conv2d(base, 2, 1)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._up_to(b,  e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up_to(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up_to(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up_to(d2, e1), e1], dim=1))
        return torch.tanh(self.out(d1))


class ColorizationCNN(nn.Module):
    """VGG16-BN-based colorizer (matches CNNtest.py / cnn_train.py)."""

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        vgg      = models.vgg16_bn(pretrained=pretrained_backbone)
        features = list(vgg.features.children())

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(*features[6:13])
        self.enc3 = nn.Sequential(*features[13:23])
        self.enc4 = nn.Sequential(*features[23:33])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dec4 = nn.Conv2d(512 + 512, 512, 3, padding=1)
        self.dec3 = nn.Conv2d(512 + 256, 256, 3, padding=1)
        self.dec2 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.dec1 = nn.Conv2d(128 + 64,   64, 3, padding=1)
        self.out  = nn.Conv2d(64,           2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b  = self.bottleneck(F.max_pool2d(e4, 2))
        up = lambda t, s: F.interpolate(t, scale_factor=s, mode="bilinear", align_corners=False)

        d4 = F.relu(self.dec4(torch.cat([up(b,  2), e4], dim=1)))
        d3 = F.relu(self.dec3(torch.cat([up(d4, 2), e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([up(d3, 2), e2], dim=1)))
        d1 = F.relu(self.dec1(torch.cat([up(d2, 2), e1], dim=1)))
        return torch.tanh(self.out(d1))


# Loss / metric helpers

def chroma_weighted_l1(
    pred_ab: torch.Tensor,
    gt_ab:   torch.Tensor,
    alpha:   float = 6.0,
    eps:     float = 1e-6,
) -> float:
    """
    Custom chroma-weighted L1 loss (from regression_train.py).
    Weights each pixel by  1 + alpha * sqrt(a^2 + b^2 + eps).
    Emphasises saturated pixels so the model does not collapse to gray.
    """
    with torch.no_grad():
        c = torch.sqrt(gt_ab[:, 0] ** 2 + gt_ab[:, 1] ** 2 + eps)  # (N, H, W)
        w = (1.0 + alpha * c).unsqueeze(1)                          # (N, 1, H, W)
    return float((w * (pred_ab - gt_ab).abs()).mean().item())


def _autocast():
    device_type = "cuda" if str(DEVICE).startswith("cuda") else "cpu"
    amp_enabled  = USE_AMP and str(DEVICE).startswith("cuda")
    return torch.amp.autocast(device_type=device_type, enabled=amp_enabled)


# Evaluation loop

@torch.inference_mode()
def evaluate_model(
    model:   nn.Module,
    loader:  DataLoader,
    name:    str,
) -> Dict[str, float]:
    """Run inference over *loader* and return all metrics."""
    model.eval()

    total_abs  = 0.0
    total_sq   = 0.0
    total_elems = 0
    total_abs_a = 0.0
    total_abs_b = 0.0

    chroma_loss_sum = 0.0
    chroma_loss_batches = 0

    thr_hits = {float(t): 0 for t in ACCURACY_THRESHOLDS}

    sat_abs_sum = 0.0
    sat_count   = 0.0
    pred_chroma_sum = 0.0
    gt_chroma_sum   = 0.0
    chroma_batches  = 0

    t0 = time.time()

    for batch_idx, (L, ab) in enumerate(loader):
        L  = L .to(DEVICE, non_blocking=True)
        ab = ab.to(DEVICE, non_blocking=True)

        if CHANNELS_LAST and DEVICE.startswith("cuda"):
            L  = L .contiguous(memory_format=torch.channels_last)
            ab = ab.contiguous(memory_format=torch.channels_last)

        with _autocast():
            pred = model(L)

        pred = pred.float()
        ab   = ab.float()

        # standard regression errors
        err     = pred - ab
        abs_err = err.abs()
        sq_err  = err * err

        n_elems = abs_err.numel()
        total_elems += n_elems
        total_abs   += float(abs_err.sum().item())
        total_sq    += float(sq_err.sum().item())
        total_abs_a += float(abs_err[:, 0].sum().item())
        total_abs_b += float(abs_err[:, 1].sum().item())

        for t in thr_hits:
            thr_hits[t] += int((abs_err < t).sum().item())

        # custom chroma-weighted L1
        chroma_loss_sum += chroma_weighted_l1(pred, ab, alpha=CHROMA_LOSS_ALPHA)
        chroma_loss_batches += 1

        # saturated-only MAE
        gt_c = torch.sqrt(ab[:, 0] ** 2 + ab[:, 1] ** 2 + 1e-6)   # (N, H, W)
        mask = (gt_c > SAT_TAU).unsqueeze(1)                        # (N, 1, H, W)
        sat_abs_sum += float((abs_err * mask).sum().item())
        sat_count   += float(mask.sum().item())

        # mean chroma bookkeeping
        pred_chroma_sum += float(
            torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-6).mean().item()
        )
        gt_chroma_sum += float(gt_c.mean().item())
        chroma_batches += 1

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{name}] batch {batch_idx+1}/{len(loader)}  "
                  f"({elapsed:.0f}s elapsed)", end="\r", flush=True)

    print()  # newline after the \r progress line

    mae  = total_abs / max(total_elems, 1)
    mse  = total_sq  / max(total_elems, 1)
    rmse = math.sqrt(mse)
    norm_acc = max(0.0, min(1.0, 1.0 - (mae / 2.0)))

    out: Dict[str, float] = {
        "chroma_loss":       chroma_loss_sum / max(chroma_loss_batches, 1),
        "mae":               mae,
        "mae_a":             total_abs_a / max(total_elems / 2, 1),
        "mae_b":             total_abs_b / max(total_elems / 2, 1),
        "mse":               mse,
        "rmse":              rmse,
        "norm_acc":          norm_acc,
        "sat_mae":           sat_abs_sum / max(sat_count, 1.0),
        "mean_pred_chroma":  pred_chroma_sum / max(chroma_batches, 1),
        "mean_gt_chroma":    gt_chroma_sum   / max(chroma_batches, 1),
    }
    for t, hits in thr_hits.items():
        out[f"acc@{t:g}"] = hits / max(total_elems, 1)

    return out


@torch.inference_mode()
def evaluate_baseline(loader: DataLoader, name: str) -> Dict[str, float]:
    """Evaluate the zero-prediction baseline (predict a*=b*=0 everywhere)."""
    total_abs   = 0.0
    total_sq    = 0.0
    total_elems = 0
    total_abs_a = 0.0
    total_abs_b = 0.0

    chroma_loss_sum     = 0.0
    chroma_loss_batches = 0

    thr_hits = {float(t): 0 for t in ACCURACY_THRESHOLDS}

    sat_abs_sum = 0.0
    sat_count   = 0.0
    gt_chroma_sum  = 0.0
    chroma_batches = 0

    t0 = time.time()

    for batch_idx, (_, ab) in enumerate(loader):
        ab   = ab.to(DEVICE, non_blocking=True).float()
        pred = torch.zeros_like(ab)   # baseline: always predict gray

        err     = pred - ab
        abs_err = err.abs()
        sq_err  = err * err

        n_elems = abs_err.numel()
        total_elems += n_elems
        total_abs   += float(abs_err.sum().item())
        total_sq    += float(sq_err.sum().item())
        total_abs_a += float(abs_err[:, 0].sum().item())
        total_abs_b += float(abs_err[:, 1].sum().item())

        for t in thr_hits:
            thr_hits[t] += int((abs_err < t).sum().item())

        chroma_loss_sum += chroma_weighted_l1(pred, ab, alpha=CHROMA_LOSS_ALPHA)
        chroma_loss_batches += 1

        gt_c = torch.sqrt(ab[:, 0] ** 2 + ab[:, 1] ** 2 + 1e-6)
        mask = (gt_c > SAT_TAU).unsqueeze(1)
        sat_abs_sum += float((abs_err * mask).sum().item())
        sat_count   += float(mask.sum().item())

        gt_chroma_sum  += float(gt_c.mean().item())
        chroma_batches += 1

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{name}/baseline] batch {batch_idx+1}/{len(loader)}  "
                  f"({elapsed:.0f}s elapsed)", end="\r", flush=True)

    print()

    mae  = total_abs / max(total_elems, 1)
    mse  = total_sq  / max(total_elems, 1)
    rmse = math.sqrt(mse)
    norm_acc = max(0.0, min(1.0, 1.0 - (mae / 2.0)))

    out: Dict[str, float] = {
        "chroma_loss":       chroma_loss_sum / max(chroma_loss_batches, 1),
        "mae":               mae,
        "mae_a":             total_abs_a / max(total_elems / 2, 1),
        "mae_b":             total_abs_b / max(total_elems / 2, 1),
        "mse":               mse,
        "rmse":              rmse,
        "norm_acc":          norm_acc,
        "sat_mae":           sat_abs_sum / max(sat_count, 1.0),
        "mean_pred_chroma":  0.0,          # baseline always predicts 0 chroma
        "mean_gt_chroma":    gt_chroma_sum / max(chroma_batches, 1),
    }
    for t, hits in thr_hits.items():
        out[f"acc@{t:g}"] = hits / max(total_elems, 1)

    return out


# Checkpoint loading

def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if any(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise RuntimeError(f"Unrecognised checkpoint format: {type(obj)}")


def load_checkpoint(model: nn.Module, path: Path) -> None:
    obj   = torch.load(str(path), map_location=DEVICE)
    state = _unwrap_state_dict(obj)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"  [warn] strict load failed: {e}; retrying strict=False")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [warn] missing  : {missing[:6]}")
        if unexpected:
            print(f"  [warn] unexpected: {unexpected[:6]}")


# Pretty-print

METRIC_LABELS = {
    "chroma_loss":      "Chroma-Weighted L1 (α=6)",
    "mae":              "MAE (both channels)",
    "mae_a":            "MAE  a* channel",
    "mae_b":            "MAE  b* channel",
    "mse":              "MSE",
    "rmse":             "RMSE",
    "norm_acc":         "Normalised Accuracy",
    "sat_mae":          f"Saturated-pixel MAE (chroma>{SAT_TAU})",
    "mean_pred_chroma": "Mean Predicted Chroma",
    "mean_gt_chroma":   "Mean GT Chroma",
    "acc@0.05":         "Pixel Accuracy @ 0.05",
    "acc@0.1":          "Pixel Accuracy @ 0.10",
    "acc@0.2":          "Pixel Accuracy @ 0.20",
}

METRIC_ORDER = [
    "chroma_loss", "mae", "mae_a", "mae_b",
    "mse", "rmse", "norm_acc", "sat_mae",
    "mean_pred_chroma", "mean_gt_chroma",
    "acc@0.05", "acc@0.1", "acc@0.2",
]


def print_results_table(dataset_name: str, results: Dict[str, Dict[str, float]]) -> None:
    col_names = list(results.keys())
    col_w = max(14, max(len(c) for c in col_names) + 2)
    label_w = max(len(v) for v in METRIC_LABELS.values()) + 2

    sep = "─" * (label_w + col_w * len(col_names) + 2)
    header = f"{'Metric':<{label_w}}" + "".join(f"{c:>{col_w}}" for c in col_names)

    print()
    print(f"{'':=<{len(sep)}}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'':=<{len(sep)}}")
    print(header)
    print(sep)

    for key in METRIC_ORDER:
        if key not in next(iter(results.values())):
            continue
        label = METRIC_LABELS.get(key, key)
        row = f"{label:<{label_w}}"
        for col in col_names:
            val = results[col].get(key, float("nan"))
            row += f"{val:>{col_w}.6f}"
        print(row)

    print(sep)


# DataLoader factory

def make_loader(dataset: Dataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        drop_last=False,
    )


# Main

def main() -> None:
    print(f"Device      : {DEVICE}")
    print(f"UNet ckpt   : {UNET_CHECKPOINT}")
    print(f"CNN  ckpt   : {CNN_CHECKPOINT}")
    print(f"COCO shards : {COCO_SHARD_DIR}")
    print(f"ImageNet    : {IMAGENET_SHARD_DIR}  (sample={IMAGENET_SAMPLE_N})")
    print()

    # Load models
    print("Loading UNet …")
    unet = UNetColorizer(base=UNET_BASE_CH).to(DEVICE)
    if CHANNELS_LAST and DEVICE.startswith("cuda"):
        unet = unet.to(memory_format=torch.channels_last)
    load_checkpoint(unet, UNET_CHECKPOINT)
    unet.eval()
    print("  done.\n")

    print("Loading VGG16-CNN …")
    cnn = ColorizationCNN(pretrained_backbone=False).to(DEVICE)
    if CHANNELS_LAST and DEVICE.startswith("cuda"):
        cnn = cnn.to(memory_format=torch.channels_last)
    load_checkpoint(cnn, CNN_CHECKPOINT)
    cnn.eval()
    print("  done.\n")

    # Build datasets
    print("Building COCO training dataset …")
    coco_ds = MMapLabDataset(COCO_SHARD_DIR)
    print(f"  {len(coco_ds):,} images total\n")

    print("Building ImageNet dataset …")
    imagenet_full = MMapLabDataset(IMAGENET_SHARD_DIR)
    n_imagenet = len(imagenet_full)
    rng = np.random.default_rng(IMAGENET_SEED)
    sample_idx = rng.choice(n_imagenet, size=min(IMAGENET_SAMPLE_N, n_imagenet), replace=False).tolist()
    imagenet_ds = Subset(imagenet_full, sample_idx)
    print(f"  {n_imagenet:,} images available → using {len(imagenet_ds):,} random samples\n")

    coco_loader     = make_loader(coco_ds)
    imagenet_loader = make_loader(imagenet_ds)

    # Evaluate
    datasets = {
        "COCO (full train)": coco_loader,
        f"ImageNet ({len(imagenet_ds):,} random)": imagenet_loader,
    }

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for ds_name, loader in datasets.items():
        print(f"\n{'─'*60}")
        print(f"Evaluating on: {ds_name}")
        print(f"{'─'*60}")

        results: Dict[str, Dict[str, float]] = {}

        print(f"  Baseline (predict a*=b*=0) …")
        results["Baseline (a*=b*=0)"] = evaluate_baseline(loader, ds_name)

        print(f"  UNet …")
        results["UNet"] = evaluate_model(unet, loader, f"UNet/{ds_name}")

        print(f"  VGG16-CNN …")
        results["VGG16-CNN"] = evaluate_model(cnn, loader, f"VGG16-CNN/{ds_name}")

        all_results[ds_name] = results
        print_results_table(ds_name, results)

    # Summary: improvement over baseline
    print()
    print("=" * 60)
    print("  Summary: Model improvement over Baseline")
    print("=" * 60)

    improve_metrics = ["chroma_loss", "mae", "rmse", "sat_mae"]
    improve_labels  = {
        "chroma_loss": "Chroma-Weighted L1",
        "mae":         "MAE",
        "rmse":        "RMSE",
        "sat_mae":     "Saturated MAE",
    }

    for ds_name, results in all_results.items():
        print(f"\n  [{ds_name}]")
        base = results["Baseline (a*=b*=0)"]
        for model_name in ("UNet", "VGG16-CNN"):
            m = results[model_name]
            print(f"    {model_name}:")
            for key in improve_metrics:
                b_val = base[key]
                m_val = m[key]
                if b_val > 0:
                    pct = (b_val - m_val) / b_val * 100.0
                    direction = "↓" if pct >= 0 else "↑"
                    print(f"      {improve_labels[key]:<25} {m_val:.6f}  ({direction}{abs(pct):.1f}% vs baseline)")
                else:
                    print(f"      {improve_labels[key]:<25} {m_val:.6f}")


if __name__ == "__main__":
    main()
