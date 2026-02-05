import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import time

import os, time, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from grid_train import CocoMMapCropDataset  # replace with actual import

from CNNtest import ColorizationCNN


from grid_train import (
    SHARD_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)
# ----------------------------
# Device
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True

# ----------------------------
# Quick train/test subset settings
# ----------------------------
GRID_TRAIN_SUBSAMPLE_N = 1024
GRID_TEST_SUBSAMPLE_N = 1024
BATCH_SIZE = 16
EPOCHS_QUICK = 3  # few epochs for quick check
PRINT_EVERY = 20

ACCURACY_THRESHOLDS = [0.01, 0.05, 0.1]

# ----------------------------
# Utilities
# ----------------------------
def make_split_indices(n: int, seed: int, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+n_val]
    test_idx = perm[n_train+n_val:]
    return list(train_idx), list(val_idx), list(test_idx)

def make_subset_indices(full_indices, n_subsample, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(full_indices, size=n_subsample, replace=False).tolist()

def build_loader(ds, indices, batch_size, shuffle=True, drop_last=False):
    return DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

@torch.no_grad()
def zero_baseline_mae(loader: DataLoader, device: str):
    total_abs = 0.0
    total_elems = 0
    for _, ab in loader:
        ab = ab.to(device)
        total_abs += float(ab.abs().sum().item())
        total_elems += ab.numel()
    return total_abs / max(total_elems, 1)

@torch.no_grad()
def evaluate_metrics(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total_abs, total_sq, total_elems = 0.0, 0.0, 0
    thr_hits = {float(t): 0 for t in ACCURACY_THRESHOLDS}
    for L, ab in loader:
        L = L.to(device)
        ab = ab.to(device)
        # L_rgb = L.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]
        # pred = model(L_rgb)
        pred = model(L)
        err = pred - ab
        abs_err = err.abs()
        sq_err = err * err
        elems = abs_err.numel()
        total_elems += elems
        total_abs += float(abs_err.sum().item())
        total_sq += float(sq_err.sum().item())
        for t in thr_hits:
            thr_hits[t] += int((abs_err < t).sum().item())

    mae = total_abs / max(total_elems, 1)
    mse = total_sq / max(total_elems, 1)
    rmse = float(np.sqrt(mse))
    norm_acc = max(0.0, min(1.0, 1.0 - mae / 2.0))
    out = {"mae": mae, "mse": mse, "rmse": rmse, "norm_acc": norm_acc}
    for t, hits in thr_hits.items():
        out[f"acc@{t:g}"] = hits / max(total_elems, 1)
    return out

# ----------------------------
# Quick train function
# ----------------------------
def quick_train(model, train_loader, val_loader, epochs=EPOCHS_QUICK, lr=1e-3, grad_clip=0.0):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.startswith("cuda")))
    amp_enabled = USE_AMP and DEVICE.startswith("cuda")

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for step, (L, ab) in enumerate(train_loader, start=1):
            L = L.to(DEVICE)
            ab = ab.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # L_rgb = L.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]
                # pred = model(L_rgb)

                pred = model(L)
                loss = loss_fn(pred, ab)
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            if step % PRINT_EVERY == 0:
                print(f"[epoch {epoch}/{epochs}] step {step}/{len(train_loader)} avg_loss={running/step:.4f}")
        print(f"Epoch {epoch} finished | avg train loss: {running/len(train_loader):.4f}")

    val_metrics = evaluate_metrics(model, val_loader, DEVICE)
    print(f"\nValidation metrics after {epochs} epochs: {val_metrics}")
    return model, val_metrics

# ----------------------------
# Original U-Net we've been using
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetColorizer(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.dec4 = ConvBlock(base * 16 + base * 8, base * 8)
        self.dec3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)

        self.out = nn.Conv2d(base, 2, 1)

    @staticmethod
    def _up_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self._up_to(b, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self._up_to(d4, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self._up_to(d3, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self._up_to(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.tanh(self.out(d1))


# ----------------------------
# Main quick test
# ----------------------------
def main_quick_test(ds):
    n = len(ds)
    train_idx, val_idx, test_idx = make_split_indices(n, seed=42)
    train_idx_1k = make_subset_indices(train_idx, GRID_TRAIN_SUBSAMPLE_N, seed=100)
    test_idx_1k = make_subset_indices(test_idx, GRID_TEST_SUBSAMPLE_N, seed=200)

    train_loader_1k = build_loader(ds, train_idx_1k, batch_size=BATCH_SIZE)
    test_loader_1k = build_loader(ds, test_idx_1k, batch_size=BATCH_SIZE)

    # Baseline
    train0 = zero_baseline_mae(train_loader_1k, DEVICE)
    test0 = zero_baseline_mae(test_loader_1k, DEVICE)
    print(f"Zero-pred baseline MAE train 1K: {train0:.6f}, test 1K: {test0:.6f}")

    # Model quick training

    # model = ColorizationCNN(pretrained_backbone=True)
    model = UNetColorizer(base=32)
    model, val_metrics = quick_train(model, train_loader_1k, test_loader_1k, epochs=EPOCHS_QUICK, lr=1e-3)
    print("Quick test finished.")

# ----------------------------
# Usage
# ----------------------------
if __name__ == "__main__":
    ds = CocoMMapCropDataset(SHARD_DIR)
    main_quick_test(ds)
