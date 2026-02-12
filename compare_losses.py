"""
Train models with different loss functions and compare results.
Each loss function trains its own model from scratch, then we compare
validation metrics to see which loss produces the best colorizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import math
import time
from typing import Callable, Dict, Tuple

from regression_train import (
    CocoMMapCropDataset,
    UNetColorizer,
    make_split_indices,
    make_loader,
    SHARD_DIR,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    CHANNELS_LAST,
    autocast_ctx,
    batch_mean_chroma,
)
from CNNtest import ColorizationCNN

# =========================
# Loss Functions to Compare
# =========================

def l1_loss(pred_ab: torch.Tensor, gt_ab: torch.Tensor) -> torch.Tensor:
    """Standard L1 loss."""
    return F.l1_loss(pred_ab, gt_ab)


def l2_loss(pred_ab: torch.Tensor, gt_ab: torch.Tensor) -> torch.Tensor:
    """Standard L2/MSE loss."""
    return F.mse_loss(pred_ab, gt_ab)


def smooth_l1_loss(pred_ab: torch.Tensor, gt_ab: torch.Tensor) -> torch.Tensor:
    """Huber loss (smooth L1)."""
    return F.smooth_l1_loss(pred_ab, gt_ab)


def chroma_weighted_l1(pred_ab: torch.Tensor, gt_ab: torch.Tensor, alpha: float = 6.0, eps: float = 1e-6) -> torch.Tensor:
    """Chroma-weighted L1 (from regression_train.py)."""
    with torch.no_grad():
        c = torch.sqrt(gt_ab[:, 0] ** 2 + gt_ab[:, 1] ** 2 + eps)
        w = (1.0 + alpha * c).unsqueeze(1)
    return (w * (pred_ab - gt_ab).abs()).mean()


def perceptual_l1(pred_ab: torch.Tensor, gt_ab: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Chroma-based perceptual weighting (inverse)."""
    with torch.no_grad():
        c = torch.sqrt(gt_ab[:, 0] ** 2 + gt_ab[:, 1] ** 2 + eps)
        w = 1.0 / (1.0 + c).unsqueeze(1)  # weight grays more
    return (w * (pred_ab - gt_ab).abs()).mean()


def cosine_similarity_loss(pred_ab: torch.Tensor, gt_ab: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Direction-aware loss using cosine similarity + magnitude."""
    pred_flat = pred_ab.view(pred_ab.shape[0], 2, -1)
    gt_flat = gt_ab.view(gt_ab.shape[0], 2, -1)
    
    cos_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1, eps=eps)
    cos_loss = (1.0 - cos_sim).mean()
    
    pred_mag = torch.sqrt(pred_flat[:, 0] ** 2 + pred_flat[:, 1] ** 2 + eps)
    gt_mag = torch.sqrt(gt_flat[:, 0] ** 2 + gt_flat[:, 1] ** 2 + eps)
    mag_loss = F.l1_loss(pred_mag, gt_mag)
    
    return 0.5 * cos_loss + 0.5 * mag_loss


def channel_weighted_l1(pred_ab: torch.Tensor, gt_ab: torch.Tensor) -> torch.Tensor:
    """Separate L1 for a and b channels with different weights."""
    loss_a = F.l1_loss(pred_ab[:, 0], gt_ab[:, 0])
    loss_b = F.l1_loss(pred_ab[:, 1], gt_ab[:, 1])
    return 0.5 * loss_a + 0.5 * loss_b


# =========================
# Evaluation
# =========================

@torch.inference_mode()
def evaluate_loss_on_set(model: nn.Module, loader: DataLoader, loss_fn: Callable, device: str, use_amp: bool) -> Dict:
    """Compute metrics using a specific loss function."""
    model.eval()
    amp_enabled = use_amp and str(device).startswith("cuda")
    
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_elems = 0
    total_mae_a = 0.0
    total_mae_b = 0.0
    total_pred_chroma = 0.0
    total_gt_chroma = 0.0
    batch_count = 0
    
    for L, ab in loader:
        L = L.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)
        
        if CHANNELS_LAST and device.startswith("cuda"):
            L = L.contiguous(memory_format=torch.channels_last)
            ab = ab.contiguous(memory_format=torch.channels_last)
        
        with autocast_ctx(enabled=amp_enabled):
            pred = model(L)
            loss = loss_fn(pred, ab)
        
        with torch.no_grad():
            err = (pred - ab).float()
            abs_err = err.abs()
            sq_err = err * err
            
            elems = abs_err.numel()
            total_elems += elems
            total_loss += float(loss.item())
            total_mae += float(abs_err.sum().item())
            total_mse += float(sq_err.sum().item())
            total_mae_a += float(abs_err[:, 0].sum().item())
            total_mae_b += float(abs_err[:, 1].sum().item())
            total_pred_chroma += batch_mean_chroma(pred)
            total_gt_chroma += batch_mean_chroma(ab)
            batch_count += 1
    
    mae = total_mae / max(total_elems, 1)
    mse = total_mse / max(total_elems, 1)
    rmse = math.sqrt(mse)
    
    return {
        "loss": total_loss / max(batch_count, 1),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mae_a": total_mae_a / max(total_elems / 2, 1),
        "mae_b": total_mae_b / max(total_elems / 2, 1),
        "mean_pred_chroma": total_pred_chroma / max(batch_count, 1),
        "mean_gt_chroma": total_gt_chroma / max(batch_count, 1),
    }


# =========================
# Training
# =========================

def train_with_loss(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable,
    loss_name: str,
    epochs: int = 1,
    lr: float = 2e-4,
    device: str = "cuda",
    use_amp: bool = True,
    print_every: int = 200,
) -> Dict:
    """Train model with a specific loss function."""
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.startswith("cuda") else None
    amp_enabled = use_amp and device.startswith("cuda")
    
    best_val_mae = float('inf')
    total_batches = 0
    
    print(f"\n{'='*80}")
    print(f"Training with {loss_name} loss")
    print(f"{'='*80}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        for batch_idx, (L, ab) in enumerate(train_loader):
            L = L.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)
            
            if CHANNELS_LAST and device.startswith("cuda"):
                L = L.contiguous(memory_format=torch.channels_last)
                ab = ab.contiguous(memory_format=torch.channels_last)
            
            optimizer.zero_grad()
            
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    pred = model(L)
                    loss = loss_fn(pred, ab)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(L)
                loss = loss_fn(pred, ab)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            epoch_loss += float(loss.item())
            batch_count += 1
            total_batches += 1
            
            if (batch_idx + 1) % print_every == 0:
                avg_loss = epoch_loss / batch_count
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1} | Avg Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
        
        # Validation
        print(f"  Evaluating on validation set...")
        val_metrics = evaluate_loss_on_set(model, val_loader, loss_fn, device, use_amp)
        val_mae = val_metrics["mae"]
        
        print(f"  Val MAE: {val_mae:.6f} | Val RMSE: {val_metrics['rmse']:.6f} | Val Chroma: {val_metrics['mean_pred_chroma']:.6f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
    
    return val_metrics


# =========================
# Main
# =========================

def main():
    print(f"Device: {DEVICE}")
    print(f"Shard dir: {SHARD_DIR.resolve()}\n")
    
    # Choose model
    model_choice = input("Choose model (unet/cnn): ").strip().lower()
    if model_choice == "cnn":
        model_class = ColorizationCNN
        model_name = "ColorizationCNN"
    else:
        model_class = UNetColorizer
        model_name = "UNetColorizer"
    
    print(f"Using {model_name}\n")
    
    # Load dataset and create splits
    ds = CocoMMapCropDataset(SHARD_DIR, max_shards=1)  # Use 1 shard for testing
    n = len(ds)
    train_idx, val_idx, test_idx = make_split_indices(n, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    
    train_loader = make_loader(train_ds, shuffle=True, sampler=None, drop_last=True)
    val_loader = make_loader(val_ds, shuffle=False, sampler=None, drop_last=False)
    
    print(f"Dataset size: {n} | Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    print(f"Batch size: {BATCH_SIZE}\n")
    
    # Define loss functions to compare
    losses = {
        "L1": lambda p, g: l1_loss(p, g),
        "L2": lambda p, g: l2_loss(p, g),
        "Smooth L1": lambda p, g: smooth_l1_loss(p, g),
        "Chroma-weighted L1 (α=6)": lambda p, g: chroma_weighted_l1(p, g, alpha=6.0),
        "Chroma-weighted L1 (α=3)": lambda p, g: chroma_weighted_l1(p, g, alpha=3.0),
        "Perceptual L1": lambda p, g: perceptual_l1(p, g),
        "Cosine + Magnitude": lambda p, g: cosine_similarity_loss(p, g),
        "Channel-weighted L1": lambda p, g: channel_weighted_l1(p, g),
    }
    
    epochs = int(input("Number of epochs to train: ") or "1")
    
    print("\n" + "="*120)
    print(f"{'Loss Function':<25} {'MAE':<12} {'RMSE':<12} {'MAE_a':<12} {'MAE_b':<12} {'Pred Chr':<12} {'GT Chr':<12}")
    print("="*120)
    
    results = {}
    
    for loss_name, loss_fn in losses.items():
        # Create fresh model for each loss
        if model_choice == "cnn":
            model = ColorizationCNN().to(DEVICE)
        else:
            model = UNetColorizer(base=32).to(DEVICE)
        
        if CHANNELS_LAST and DEVICE.startswith("cuda"):
            model = model.to(memory_format=torch.channels_last)
        
        # Train with this loss
        metrics = train_with_loss(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            loss_name=loss_name,
            epochs=epochs,
            device=DEVICE,
            use_amp=True,
        )
        
        results[loss_name] = metrics
        
        print(
            f"{loss_name:<25} "
            f"{metrics['mae']:<12.6f} "
            f"{metrics['rmse']:<12.6f} "
            f"{metrics['mae_a']:<12.6f} "
            f"{metrics['mae_b']:<12.6f} "
            f"{metrics['mean_pred_chroma']:<12.6f} "
            f"{metrics['mean_gt_chroma']:<12.6f}"
        )
    
    print("="*120)
    
    # Summary
    best_mae = min(results.items(), key=lambda x: x[1]["mae"])
    print(f"\n✓ Best by MAE: {best_mae[0]} (MAE={best_mae[1]['mae']:.6f})")
    
    best_rmse = min(results.items(), key=lambda x: x[1]["rmse"])
    print(f"✓ Best by RMSE: {best_rmse[0]} (RMSE={best_rmse[1]['rmse']:.6f})")
    
    best_chroma = max(results.items(), key=lambda x: x[1]["mean_pred_chroma"])
    print(f"✓ Best chroma coverage: {best_chroma[0]} (Pred Chroma={best_chroma[1]['mean_pred_chroma']:.6f})")


if __name__ == "__main__":
    main()