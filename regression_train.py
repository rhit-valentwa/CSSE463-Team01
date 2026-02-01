from pathlib import Path
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

SHARD_DIR = Path("data/coco/train2017_cache_256_mmap")
BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 5
LR = 2e-4
BASE_CH = 32
SAVE_MODEL_PATH = "unet_colorizer.pt"
SAVE_BEST_MODEL_PATH = "unet_colorizer_best.pt"
PRINT_EVERY = 200
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFETCH_FACTOR = 4
USE_AMP = True

# split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

ACCURACY_THRESHOLDS = (0.05, 0.10, 0.20)

PLOT_PATH = "train_val_test_accuracy.png"

torch.backends.cudnn.benchmark = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CocoMMapCropDataset(Dataset):
    """
    Loads:
      shard_XXXXX_L.npy  : (N,256,256) uint8
      shard_XXXXX_ab.npy : (N,2,256,256) int8
    via memory-mapping for fast random access.

    Returns:
      L_t  : (1,256,256) float32 in [0,1]
      ab_t : (2,256,256) float32 in [-1,1]
    """
    def __init__(self, shard_dir: Path):
        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.is_dir():
            raise RuntimeError(f"Shard directory not found: {self.shard_dir.resolve()}")

        self.L_paths = sorted(self.shard_dir.glob("shard_*_L.npy"))
        self.ab_paths = sorted(self.shard_dir.glob("shard_*_ab.npy"))
        if not self.L_paths or not self.ab_paths:
            raise RuntimeError(f"Expected shard_*_L.npy and shard_*_ab.npy in {self.shard_dir.resolve()}")

        if len(self.L_paths) != len(self.ab_paths):
            raise RuntimeError("Mismatched number of L and ab shard files.")

        # Compute global index: global_idx -> (shard_idx, local_idx)
        self._index = []
        for si, Lp in enumerate(self.L_paths):
            Lm = np.load(Lp, mmap_mode="r")
            n = int(Lm.shape[0])
            self._index.extend((si, li) for li in range(n))

        # Per-worker cache of currently opened mmaps
        self._cached_si = None
        self._L_mmap = None
        self._ab_mmap = None

    def __len__(self):
        return len(self._index)

    def _load_shard(self, si: int):
        if self._cached_si != si:
            self._L_mmap = np.load(self.L_paths[si], mmap_mode="r")
            self._ab_mmap = np.load(self.ab_paths[si], mmap_mode="r")
            self._cached_si = si
        return self._L_mmap, self._ab_mmap

    def __getitem__(self, idx):
        si, li = self._index[idx]
        Lm, abm = self._load_shard(si)

        L_u8 = Lm[li]      # (256,256) uint8
        ab_i8 = abm[li]    # (2,256,256) int8

        L = (L_u8.astype(np.float32) / 255.0)
        ab = (ab_i8.astype(np.float32) / 128.0)

        L_t = torch.from_numpy(L).unsqueeze(0)   # (1,256,256)
        ab_t = torch.from_numpy(ab)              # (2,256,256)
        return L_t, ab_t

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

def make_split_indices(n: int, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

@torch.no_grad()
def evaluate_metrics(model: nn.Module, loader: DataLoader, device: str, use_amp: bool, thresholds=()):
    model.eval()
    amp_enabled = use_amp and str(device).startswith("cuda")

    total_abs = 0.0
    total_sq = 0.0
    total_elems = 0

    thr_hits = {float(t): 0 for t in thresholds}

    for L, ab in loader:
        L = L.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred = model(L)

        err = (pred - ab).float()
        abs_err = err.abs()
        sq_err = err * err

        # count elements: N * 2 * H * W
        elems = abs_err.numel()
        total_elems += elems

        total_abs += float(abs_err.sum().item())
        total_sq += float(sq_err.sum().item())

        for t in thr_hits:
            thr_hits[t] += int((abs_err < t).sum().item())

    mae = total_abs / max(total_elems, 1)
    mse = total_sq / max(total_elems, 1)
    rmse = float(np.sqrt(mse))

    # Since ab is in [-1,1], max absolute error per element is 2.
    norm_acc = 1.0 - (mae / 2.0)
    norm_acc = max(0.0, min(1.0, norm_acc))

    out = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "norm_acc": norm_acc,
    }
    for t, hits in thr_hits.items():
        out[f"acc@{t:g}"] = hits / max(total_elems, 1)

    return out


def plot_curves(train_acc, val_acc, test_acc, out_path: str):
    # Use matplotlib only inside this function to keep dependencies optional-ish
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_acc) + 1))

    plt.figure()
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.plot(epochs, test_acc, label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (normalized)")
    plt.title("Train / Val / Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def train():
    set_seed(SEED)
    print("Device:", DEVICE)
    print("Shard dir:", SHARD_DIR.resolve())
    print(f"BATCH_SIZE={BATCH_SIZE} workers={NUM_WORKERS} AMP={USE_AMP}")
    print(f"Split: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}")

    ds = CocoMMapCropDataset(SHARD_DIR)
    n = len(ds)
    train_idx, val_idx, test_idx = make_split_indices(n, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    print(f"Samples: total={n} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=False,
    )

    model = UNetColorizer(base=BASE_CH).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.startswith("cuda")))

    best_val = float("inf")

    # track curves (we'll plot normalized accuracy)
    train_acc_curve = []
    val_acc_curve = []
    test_acc_curve = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for step, (L, ab) in enumerate(train_loader, start=1):
            L = L.to(DEVICE, non_blocking=True)
            ab = ab.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.startswith("cuda"))):
                pred = model(L)
                loss = loss_fn(pred, ab)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())

            if PRINT_EVERY and (step % PRINT_EVERY == 0):
                print(f"epoch {epoch}/{EPOCHS} step {step} train_batch_l1 {loss.item():.4f} train_avg {running/step:.4f}")

        train_epoch_l1 = running / max(len(train_loader), 1)

        # Evaluate MAE/MSE/RMSE + threshold "accuracy" on train/val/test (test is optional each epoch,
        # but you explicitly asked to plot it too, so we compute it each epoch).
        train_metrics = evaluate_metrics(model, train_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
        val_metrics = evaluate_metrics(model, val_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
        test_metrics = evaluate_metrics(model, test_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)

        # Print summary for this epoch
        print(
            f"epoch {epoch}/{EPOCHS} "
            f"train_l1(batchavg) {train_epoch_l1:.4f} | "
            f"train_mae {train_metrics['mae']:.4f} val_mae {val_metrics['mae']:.4f} test_mae {test_metrics['mae']:.4f} | "
            f"train_acc {train_metrics['norm_acc']:.4f} val_acc {val_metrics['norm_acc']:.4f} test_acc {test_metrics['norm_acc']:.4f}"
        )

        train_acc_curve.append(train_metrics["norm_acc"])
        val_acc_curve.append(val_metrics["norm_acc"])
        test_acc_curve.append(test_metrics["norm_acc"])

        # Save best by validation MAE (lower is better)
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            torch.save(model.state_dict(), SAVE_BEST_MODEL_PATH)
            print(f"  saved best -> {SAVE_BEST_MODEL_PATH} (val_mae={best_val:.4f})")

    # Save final
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Saved final model ->", SAVE_MODEL_PATH)

    # Plot curves
    plot_curves(train_acc_curve, val_acc_curve, test_acc_curve, PLOT_PATH)
    print("Saved plot ->", PLOT_PATH)

    # Final report on BEST checkpoint (more standard)
    if os.path.exists(SAVE_BEST_MODEL_PATH):
        model.load_state_dict(torch.load(SAVE_BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded best checkpoint for final metrics: {SAVE_BEST_MODEL_PATH} (best val_mae={best_val:.4f})")

    final_train = evaluate_metrics(model, train_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
    final_val = evaluate_metrics(model, val_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
    final_test = evaluate_metrics(model, test_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)

    def fmt(m):
        parts = [
            f"MAE={m['mae']:.6f}",
            f"MSE={m['mse']:.6f}",
            f"RMSE={m['rmse']:.6f}",
            f"NormAcc={m['norm_acc']:.6f}",
        ]
        for t in ACCURACY_THRESHOLDS:
            parts.append(f"Acc@{t:g}={m[f'acc@{t:g}']:.6f}")
        return "  " + "  ".join(parts)

    print("\nFinal metrics (best checkpoint):")
    print("TRAIN:", fmt(final_train))
    print("VAL  :", fmt(final_val))
    print("TEST :", fmt(final_test))

    return model


if __name__ == "__main__":
    train()