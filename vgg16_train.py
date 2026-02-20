# Train ColorizationCNN with chroma-weighted L1 loss
# Based on regression_train.py but uses ColorizationCNN instead of UNetColorizer

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import math

from torchvision import models
from unet_train import (
    CocoMMapCropDataset,
    make_split_indices,
    SHARD_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    SAVE_DIR,
    PRINT_EVERY,
    SEED,
    DEVICE,
    PREFETCH_FACTOR,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    USE_AMP,
    CHANNELS_LAST,
    DETERMINISTIC,
    GRAD_CLIP_NORM,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    ACCURACY_THRESHOLDS,
    PLOT_PATH,
    CHROMA_LOSS_ALPHA,
    SAT_TAU,
    SAT_SAMPLING,
    SAT_SCORE_STRIDE,
    SAT_SAMPLING_POWER,
    SAT_SAMPLING_MIN_WEIGHT,
    SAT_SAMPLING_MAX_WEIGHT,
    SAT_SCORE_CACHE_DIR,
    SAT_SCORE_CACHE_PREFIX,
)

# MODEL

class ColorizationCNN(nn.Module):
    """
    VGG16-BN encoder with a skip-connection decoder for LAB ab prediction.
    Input:  (N, 1, H, W)  — grayscale L channel in [0, 1]
    Output: (N, 2, H, W)  — predicted ab channels in [-1, 1]
    """
    def __init__(self, pretrained_backbone: bool = False) -> None:
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

        self.dec4 = nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1)
        self.dec3 = nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(128 + 64,   64, kernel_size=3, padding=1)
        self.out  = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b  = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = F.relu(self.dec4(torch.cat([F.interpolate(b,  scale_factor=2, mode="bilinear", align_corners=False), e4], dim=1)))
        d3 = F.relu(self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False), e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], dim=1)))
        d1 = F.relu(self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], dim=1)))
        return torch.tanh(self.out(d1))


# CONFIG (overrides)
SAVE_LAST_PATH = SAVE_DIR / "cnn_colorizer_last.pt"
SAVE_BEST_PATH = SAVE_DIR / "cnn_colorizer_best.pt"

# Loss function choice
LOSS_TYPE = "chroma_weighted_l1"  # Options: "l1", "l2", "smooth_l1", "chroma_weighted_l1", "channel_weighted_l1"
CHROMA_LOSS_ALPHA = 6.0

# LOSS FUNCTIONS

def chroma_weighted_l1(pred_ab: torch.Tensor, gt_ab: torch.Tensor, alpha: float = 6.0, eps: float = 1e-6) -> torch.Tensor:
    """Chroma-weighted L1 loss - increases weight for saturated pixels."""
    with torch.no_grad():
        c = torch.sqrt(gt_ab[:, 0] ** 2 + gt_ab[:, 1] ** 2 + eps)
        w = (1.0 + alpha * c).unsqueeze(1)
    return (w * (pred_ab - gt_ab).abs()).mean()


def channel_weighted_l1(pred_ab: torch.Tensor, gt_ab: torch.Tensor) -> torch.Tensor:
    """Channel-weighted L1 - treats a and b equally."""
    loss_a = F.l1_loss(pred_ab[:, 0], gt_ab[:, 0])
    loss_b = F.l1_loss(pred_ab[:, 1], gt_ab[:, 1])
    return 0.5 * loss_a + 0.5 * loss_b


def get_loss_fn(loss_type: str, device: str) -> tuple:
    """Get loss function and its name."""
    if loss_type == "l1":
        return F.l1_loss, "L1"
    elif loss_type == "l2":
        return F.mse_loss, "L2/MSE"
    elif loss_type == "smooth_l1":
        return F.smooth_l1_loss, "Smooth L1"
    elif loss_type == "chroma_weighted_l1":
        return lambda p, g: chroma_weighted_l1(p, g, alpha=CHROMA_LOSS_ALPHA), f"Chroma-weighted L1 (α={CHROMA_LOSS_ALPHA})"
    elif loss_type == "channel_weighted_l1":
        return channel_weighted_l1, "Channel-weighted L1"
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# UTILITIES

def batch_mean_chroma(ab: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute mean chroma (saturation) for a batch."""
    c = torch.sqrt(ab[:, 0] ** 2 + ab[:, 1] ** 2 + eps)
    return float(c.mean().item())


def make_loader(
    ds: Dataset,
    shuffle: bool = True,
    sampler: Optional[DistributedSampler] = None,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader."""
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0) and PERSISTENT_WORKERS,
        drop_last=drop_last,
    )


def autocast_ctx(enabled: bool):
    """Return appropriate autocast context."""
    if enabled:
        return torch.cuda.amp.autocast()
    else:
        return torch.nullcontext()


# TRAINING

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: callable,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str,
    use_amp: bool,
    print_every: int,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch."""
    model.train()
    amp_enabled = use_amp and str(device).startswith("cuda")
    
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, (L, ab) in enumerate(loader):
        L = L.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)
        
        if CHANNELS_LAST and device.startswith("cuda"):
            L = L.contiguous(memory_format=torch.channels_last)
            ab = ab.contiguous(memory_format=torch.channels_last)
        
        optimizer.zero_grad()
        
        with autocast_ctx(enabled=amp_enabled):
            pred = model(L)
            loss = loss_fn(pred, ab)
        
        if amp_enabled:
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        
        total_loss += float(loss.item())
        batch_count += 1
        
        if (batch_idx + 1) % print_every == 0:
            avg_loss = total_loss / batch_count
            print(f"[Epoch {epoch+1}/{total_epochs}] Batch {batch_idx+1} | Avg Loss: {avg_loss:.6f}")
    
    return total_loss / max(batch_count, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: callable,
    device: str,
    use_amp: bool,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
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
        "rmse": rmse,
        "mae_a": total_mae_a / max(total_elems / 2, 1),
        "mae_b": total_mae_b / max(total_elems / 2, 1),
        "pred_chroma": total_pred_chroma / max(batch_count, 1),
        "gt_chroma": total_gt_chroma / max(batch_count, 1),
    }

def print_hyperparameters():
    print("="*80)
    print("TRAINING HYPERPARAMETERS")
    print("="*80)
    
    # General training
    print(f"Seed: {SEED}")
    print(f"Batch_Size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Epoches: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Optimizer: Adam")  # Change if you use AdamW
    print(f"LR scheduler: CosineAnnealingLR")
    
    # Model-specific (if your ColorizationCNN had multipliers, kernels, etc.)
    print(f"Model: ColorizationCNN")
    
    # Loss function
    print(f"Loss function: {LOSS_TYPE} (α={CHROMA_LOSS_ALPHA} for chroma-weighted L1)")
    
    # Saturation / chroma weighting details (from your SAT_* hyperparameters)
    print("\nSaturation weighting:")
    print(f"\tweight = min_w + K * (score ** power)")
    print(f"\tsampling K = {SAT_SAMPLING}")
    print(f"\tsampling power = {SAT_SAMPLING_POWER}")
    print(f"\tsampling min weight = {SAT_SAMPLING_MIN_WEIGHT}")
    print(f"\tsampling max weight = {SAT_SAMPLING_MAX_WEIGHT}")
    print("="*80 + "\n")


# MAIN

def main():
    print("="*80)
    print("ColorizationCNN Training with Chroma-weighted L1 Loss")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Loss function: {LOSS_TYPE} (α={CHROMA_LOSS_ALPHA} for chroma-weighted)")
    print(f"Shard dir: {SHARD_DIR.resolve()}\n")
    
    # Load dataset
    ds = CocoMMapCropDataset(SHARD_DIR, max_shards=None)
    n = len(ds)
    train_idx, val_idx, test_idx = make_split_indices(n, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)
    
    train_loader = make_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds, shuffle=False, drop_last=False)
    test_loader = make_loader(test_ds, shuffle=False, drop_last=False)
    
    print(f"Dataset split: {len(train_idx)} train | {len(val_idx)} val | {len(test_idx)} test\n")
    
    # Create model
    model = ColorizationCNN().to(DEVICE)
    if CHANNELS_LAST and DEVICE.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: ColorizationCNN")
    print(f"Parameters: {param_count:,}\n")
    
    # Get loss function
    loss_fn, loss_name = get_loss_fn(LOSS_TYPE, DEVICE)
    print(f"Loss function: {loss_name}\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and DEVICE.startswith("cuda") else None
    
    print_hyperparameters()
    
    # Training loop
    best_val_mae = float('inf')
    
    print("="*80)
    print("Training")
    print("="*80)
    
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler,
            DEVICE, USE_AMP, PRINT_EVERY, epoch, EPOCHS
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, DEVICE, USE_AMP)
        
        print(f"[Epoch {epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_metrics['loss']:.6f} | "
              f"Val MAE: {val_metrics['mae']:.6f} | "
              f"Val RMSE: {val_metrics['rmse']:.6f} | "
              f"Val Chroma: {val_metrics['pred_chroma']:.6f}")
        
        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": val_metrics,
            }
            if scaler is not None:
                ckpt["scaler"] = scaler.state_dict()
            torch.save(ckpt, SAVE_BEST_PATH)
            print(f"  → Saved best model (MAE: {best_val_mae:.6f})")
        
        # Save last model
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, SAVE_LAST_PATH)
        
        scheduler.step()
    
    # Test set evaluation
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    
    # Load best model
    best_ckpt = torch.load(SAVE_BEST_PATH, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model"])
    
    test_metrics = evaluate(model, test_loader, loss_fn, DEVICE, USE_AMP)
    
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE_a: {test_metrics['mae_a']:.6f}")
    print(f"Test MAE_b: {test_metrics['mae_b']:.6f}")
    print(f"Test Pred Chroma: {test_metrics['pred_chroma']:.6f}")
    print(f"Test GT Chroma: {test_metrics['gt_chroma']:.6f}")
    
    print("\n" + "="*80)
    print(f"Training complete! Best model saved to {SAVE_BEST_PATH}")
    print("="*80)



if __name__ == "__main__":
    if DETERMINISTIC:
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    main()