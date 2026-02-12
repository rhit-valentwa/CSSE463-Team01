# regression_train.py
#
# Train a UNet to predict Lab ab channels from L channel (normalized).
# Data: memory-mapped COCO shards:
#   shard_XXXXX_L.npy  : (N,256,256) uint8
#   shard_XXXXX_ab.npy : (N,2,256,256) int8
#
# Changes included:
# - Runs ONE epoch (EPOCHS=1)
# - Chroma-weighted L1 training loss (reduces "gray collapse")
# - Saturation-aware sampling for TRAIN (WeightedRandomSampler using per-sample chroma score)
#   + Cached saturation scores to disk so they are computed once
# - Extra eval metrics:
#     * saturated-only MAE (mask by chroma_gt > SAT_TAU)
#     * mean predicted chroma + mean GT chroma
# - Zero-ab baseline report (predict a=b=0)
# - Uses torch.amp.autocast(device_type=...) to avoid deprecation warning
# - Resume-friendly checkpoints (dict with "model"/"opt"/"scaler"/etc.)

from pathlib import Path
import os
import random
import math
import time
import bisect
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

# =========================
# CONFIG
# =========================
SHARD_DIR = Path("data/coco/train2017_cache_256_mmap")

BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 10                   # <-- run one epoch
LR = 2e-4
WEIGHT_DECAY = 1e-4

BASE_CH = 32
SAVE_DIR = Path(".")
SAVE_LAST_PATH = SAVE_DIR / "unet_colorizer_last.pt"
SAVE_BEST_PATH = SAVE_DIR / "unet_colorizer_best.pt"

PRINT_EVERY = 200
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREFETCH_FACTOR = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

USE_AMP = True
CHANNELS_LAST = True
DETERMINISTIC = False
GRAD_CLIP_NORM = 1.0  # set 0 to disable
SAT_SAMPLING_K = 4.0

# split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

ACCURACY_THRESHOLDS = (0.05, 0.10, 0.20)
PLOT_PATH = "train_val_test_accuracy.png"

# Weighted loss + saturated metrics
CHROMA_LOSS_ALPHA = 6.0   # try 4-10
SAT_TAU = 0.15            # "saturated" threshold in normalized ab units [-1,1]

# Saturation-aware sampling (TRAIN only)
SAT_SAMPLING = True
# To keep scoring fast, we estimate per-sample saturation using a pixel stride (subsample grid).
# stride=8 reads 1/64 of pixels.
SAT_SCORE_STRIDE = 8
# weights = MIN_WEIGHT + (score ** POWER)
SAT_SAMPLING_POWER = 2.0
SAT_SAMPLING_MIN_WEIGHT = 0.20
# Optional weight cap (helps prevent overfitting to super-saturated tail)
SAT_SAMPLING_MAX_WEIGHT: Optional[float] = 2.5  # e.g. 5.0 or None

# Saturation score caching (computed once, then reused)
SAT_SCORE_CACHE_DIR = SHARD_DIR
SAT_SCORE_CACHE_PREFIX = "sat_scores"

# Optional: for quick debugging
MAX_SHARDS: Optional[int] = None  # e.g. 2
# =========================


# =========================
# Repro
# =========================
def set_seed(seed: int, deterministic: bool):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}m{s:.0f}s"


# =========================
# Dataset
# =========================
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
    def __init__(self, shard_dir: Path, max_shards: Optional[int] = None):
        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.is_dir():
            raise RuntimeError(f"Shard directory not found: {self.shard_dir.resolve()}")

        L_paths = sorted(self.shard_dir.glob("shard_*_L.npy"))
        ab_paths = sorted(self.shard_dir.glob("shard_*_ab.npy"))
        if not L_paths or not ab_paths:
            raise RuntimeError(f"Expected shard_*_L.npy and shard_*_ab.npy in {self.shard_dir.resolve()}")

        if len(L_paths) != len(ab_paths):
            raise RuntimeError("Mismatched number of L and ab shard files.")

        if max_shards is not None:
            L_paths = L_paths[:max_shards]
            ab_paths = ab_paths[:max_shards]

        self.L_paths = L_paths
        self.ab_paths = ab_paths

        # Store shard sizes and cumulative offsets for O(log S) lookup
        self._sizes: List[int] = []
        for Lp, abp in zip(self.L_paths, self.ab_paths):
            Lm = np.load(Lp, mmap_mode="r")
            abm = np.load(abp, mmap_mode="r")

            if Lm.ndim != 3:
                raise RuntimeError(f"{Lp.name} expected (N,H,W), got {Lm.shape}")
            if abm.ndim != 4 or abm.shape[1] != 2:
                raise RuntimeError(f"{abp.name} expected (N,2,H,W), got {abm.shape}")
            if Lm.shape[0] != abm.shape[0]:
                raise RuntimeError(f"{Lp.name} and {abp.name} N mismatch: {Lm.shape[0]} vs {abm.shape[0]}")

            self._sizes.append(int(Lm.shape[0]))

        self._offsets: List[int] = [0]
        for n in self._sizes:
            self._offsets.append(self._offsets[-1] + n)
        self._len = self._offsets[-1]

        # Per-worker cache of currently opened mmaps
        self._cached_si = None
        self._L_mmap = None
        self._ab_mmap = None

    def __len__(self):
        return self._len

    def _global_to_local(self, idx: int) -> Tuple[int, int]:
        si = bisect.bisect_right(self._offsets, idx) - 1
        li = idx - self._offsets[si]
        return si, li

    def _load_shard(self, si: int):
        if self._cached_si != si:
            self._L_mmap = np.load(self.L_paths[si], mmap_mode="r")
            self._ab_mmap = np.load(self.ab_paths[si], mmap_mode="r")
            self._cached_si = si
        return self._L_mmap, self._ab_mmap

    def get_ab_i8(self, idx: int) -> np.ndarray:
        """Fast access to raw int8 ab for sampling/scoring."""
        si, li = self._global_to_local(idx)
        _, abm = self._load_shard(si)
        return abm[li]  # (2,H,W) int8

    def __getitem__(self, idx):
        si, li = self._global_to_local(idx)
        Lm, abm = self._load_shard(si)

        L_u8 = Lm[li]      # (256,256) uint8
        ab_i8 = abm[li]    # (2,256,256) int8

        L = (L_u8.astype(np.float32) / 255.0)
        ab = (ab_i8.astype(np.float32) / 128.0)

        L_t = torch.from_numpy(L).unsqueeze(0)   # (1,256,256)
        ab_t = torch.from_numpy(ab)              # (2,256,256)
        return L_t, ab_t


# =========================
# Model
# =========================
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

    def forward(self, x):
        return self.net(x)


class UNetColorizer(nn.Module):
    def __init__(self, base: int = 32):
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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


# =========================
# Split / Loader
# =========================
def make_split_indices(n: int, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def make_loader(ds, shuffle: bool, sampler=None, drop_last: bool = False):
    # Note: DataLoader forbids specifying shuffle=True when sampler is set.
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=drop_last,
    )


# =========================
# Loss + extra metrics helpers
# =========================
def chroma_weighted_l1(pred_ab: torch.Tensor, gt_ab: torch.Tensor, alpha: float = 6.0, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_ab, gt_ab: (N,2,H,W) in [-1,1]
    weights pixels by 1 + alpha * chroma(gt)
    """
    with torch.no_grad():
        c = torch.sqrt(gt_ab[:, 0] ** 2 + gt_ab[:, 1] ** 2 + eps)  # (N,H,W)
        w = (1.0 + alpha * c).unsqueeze(1)                          # (N,1,H,W)
    return (w * (pred_ab - gt_ab).abs()).mean()


@torch.inference_mode()
def batch_mean_chroma(ab: torch.Tensor, eps: float = 1e-6) -> float:
    c = torch.sqrt(ab[:, 0] ** 2 + ab[:, 1] ** 2 + eps)
    return float(c.mean().item())


# =========================
# AMP helpers (no deprecation warnings)
# =========================
def autocast_ctx(enabled: bool):
    device_type = "cuda" if str(DEVICE).startswith("cuda") else "cpu"
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def make_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# =========================
# Saturation-aware sampling + caching
# =========================
def estimate_sample_saturation_score(ab_i8: np.ndarray, stride: int = 8) -> float:
    """
    Fast saturation proxy for a single sample from int8 ab.
    ab_i8: (2,H,W) int8
    Returns mean chroma in normalized units (~[-1,1] space).
    Uses strided subsampling for speed.
    """
    if stride <= 1:
        a = ab_i8[0].astype(np.float32) / 128.0
        b = ab_i8[1].astype(np.float32) / 128.0
    else:
        a = ab_i8[0, ::stride, ::stride].astype(np.float32) / 128.0
        b = ab_i8[1, ::stride, ::stride].astype(np.float32) / 128.0
    chroma = np.sqrt(a * a + b * b + 1e-6)
    return float(chroma.mean())


def sat_cache_path(ds_len: int) -> Path:
    """
    Cache filename encodes major factors that change the score:
    - dataset length (different shards / MAX_SHARDS)
    - stride (subsampling)
    """
    name = f"{SAT_SCORE_CACHE_PREFIX}_len{ds_len}_stride{SAT_SCORE_STRIDE}.npy"
    return Path(SAT_SCORE_CACHE_DIR) / name


def compute_all_saturation_scores(ds: CocoMMapCropDataset) -> np.ndarray:
    """
    Compute saturation score for EVERY sample index in ds once.
    Returns float32 array of shape (len(ds),).
    """
    t0 = time.time()
    scores = np.zeros(len(ds), dtype=np.float32)
    for idx in range(len(ds)):
        ab_i8 = ds.get_ab_i8(idx)  # (2,H,W) int8
        scores[idx] = estimate_sample_saturation_score(ab_i8, stride=SAT_SCORE_STRIDE)

        if (idx + 1) % 20000 == 0:
            print(f"  sat-score computed: {idx+1}/{len(ds)}")

    print(f"Computed all saturation scores in {human_time(time.time() - t0)}")
    return scores


def load_or_compute_all_saturation_scores(ds: CocoMMapCropDataset) -> np.ndarray:
    """
    Load cached scores if present, else compute and cache.
    """
    cache_path = sat_cache_path(len(ds))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        scores = np.load(cache_path)
        if scores.shape == (len(ds),):
            print(f"Loaded saturation score cache: {cache_path}")
            return scores.astype(np.float32, copy=False)
        print(f"[warn] saturation cache shape mismatch: {scores.shape} vs {(len(ds),)}; recomputing")

    print(f"No saturation cache found. Computing and saving to: {cache_path}")
    scores = compute_all_saturation_scores(ds)

    # Save atomically: write temp then rename
    tmp = cache_path.with_suffix(".tmp.npy")
    np.save(tmp, scores)
    tmp.replace(cache_path)
    print(f"Saved saturation score cache: {cache_path}")
    return scores


def build_saturation_sampler(ds: CocoMMapCropDataset, train_idx: List[int]) -> Tuple[WeightedRandomSampler, np.ndarray]:
    """
    Uses cached per-global-index saturation scores to build weights for TRAIN subset.

    Returns:
      sampler, weights_np (len(train_idx))
    """
    all_scores = load_or_compute_all_saturation_scores(ds)  # (len(ds),)
    scores = all_scores[np.array(train_idx, dtype=np.int64)]  # (len(train_idx),)

    # Convert scores -> sampling weights
    weights = SAT_SAMPLING_MIN_WEIGHT + SAT_SAMPLING_K * np.power(scores, SAT_SAMPLING_POWER).astype(np.float32)
    if SAT_SAMPLING_MAX_WEIGHT is not None:
        weights = np.minimum(weights, float(SAT_SAMPLING_MAX_WEIGHT))

    # Print some diagnostics
    print("Saturation-aware sampling (from cached scores):")
    print(f"  score:  min={scores.min():.6f}  mean={scores.mean():.6f}  p50={np.median(scores):.6f}  p90={np.quantile(scores, 0.90):.6f}  max={scores.max():.6f}")
    print(f"  weight: min={weights.min():.6f} mean={weights.mean():.6f} p50={np.median(weights):.6f} p90={np.quantile(weights, 0.90):.6f} max={weights.max():.6f}\n")

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),   # one "epoch" worth of samples
        replacement=True,
    )
    return sampler, weights


# =========================
# Evaluation
# =========================
@torch.inference_mode()
def evaluate_metrics(model: nn.Module, loader: DataLoader, device: str, use_amp: bool, thresholds=()) -> Dict[str, float]:
    model.eval()
    amp_enabled = use_amp and str(device).startswith("cuda")

    total_abs = 0.0
    total_sq = 0.0
    total_elems = 0

    total_abs_a = 0.0
    total_abs_b = 0.0

    thr_hits = {float(t): 0 for t in thresholds}

    # new metrics
    sat_abs_sum = 0.0
    sat_count = 0.0
    pred_chroma_sum = 0.0
    gt_chroma_sum = 0.0
    chroma_batches = 0

    for L, ab in loader:
        L = L.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)

        if CHANNELS_LAST and device.startswith("cuda"):
            L = L.contiguous(memory_format=torch.channels_last)
            ab = ab.contiguous(memory_format=torch.channels_last)

        with autocast_ctx(enabled=amp_enabled):
            pred = model(L)

        err = (pred - ab).float()
        abs_err = err.abs()
        sq_err = err * err

        elems = abs_err.numel()
        total_elems += elems
        total_abs += float(abs_err.sum().item())
        total_sq += float(sq_err.sum().item())

        total_abs_a += float(abs_err[:, 0].sum().item())
        total_abs_b += float(abs_err[:, 1].sum().item())

        for t in thr_hits:
            thr_hits[t] += int((abs_err < t).sum().item())

        # saturated-only MAE accumulation
        c = torch.sqrt(ab[:, 0] ** 2 + ab[:, 1] ** 2 + 1e-6)         # (N,H,W)
        mask = (c > SAT_TAU).unsqueeze(1)                            # (N,1,H,W)
        sat_abs_sum += float((abs_err * mask).sum().item())
        sat_count += float(mask.sum().item())

        # mean chroma (pred vs gt)
        pred_chroma_sum += batch_mean_chroma(pred)
        gt_chroma_sum += batch_mean_chroma(ab)
        chroma_batches += 1

    mae = total_abs / max(total_elems, 1)
    mse = total_sq / max(total_elems, 1)
    rmse = float(math.sqrt(mse))

    # ab in [-1,1], max abs error per element is 2
    norm_acc = 1.0 - (mae / 2.0)
    norm_acc = float(max(0.0, min(1.0, norm_acc)))

    out = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mae_a": total_abs_a / max(total_elems / 2, 1),
        "mae_b": total_abs_b / max(total_elems / 2, 1),
        "norm_acc": norm_acc,
        "sat_mae": (sat_abs_sum / max(sat_count, 1.0)),
        "mean_pred_chroma": (pred_chroma_sum / max(chroma_batches, 1)),
        "mean_gt_chroma": (gt_chroma_sum / max(chroma_batches, 1)),
    }
    for t, hits in thr_hits.items():
        out[f"acc@{t:g}"] = hits / max(total_elems, 1)
    return out


@torch.inference_mode()
def evaluate_zero_baseline(loader: DataLoader, device: str, thresholds=()) -> Dict[str, float]:
    total_abs = 0.0
    total_sq = 0.0
    total_elems = 0
    thr_hits = {float(t): 0 for t in thresholds}

    total_abs_ab = 0.0
    total_chroma = 0.0
    total_pixels = 0

    # saturated-only baseline
    sat_abs_sum = 0.0
    sat_count = 0.0

    for _, ab in loader:
        ab = ab.to(device, non_blocking=True).float()
        pred = torch.zeros_like(ab)

        err = pred - ab
        abs_err = err.abs()
        sq_err = err * err

        total_abs += float(abs_err.sum().item())
        total_sq += float(sq_err.sum().item())
        total_elems += abs_err.numel()

        chroma = torch.sqrt(ab[:, 0] ** 2 + ab[:, 1] ** 2 + 1e-6)
        total_abs_ab += float(ab.abs().sum().item())
        total_chroma += float(chroma.sum().item())
        total_pixels += int(chroma.numel())

        for t in thr_hits:
            thr_hits[t] += int((abs_err < t).sum().item())

        mask = (chroma > SAT_TAU).unsqueeze(1)
        sat_abs_sum += float((abs_err * mask).sum().item())
        sat_count += float(mask.sum().item())

    mae = total_abs / max(total_elems, 1)
    mse = total_sq / max(total_elems, 1)
    rmse = float(math.sqrt(mse))

    out = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mean_abs_ab": total_abs_ab / max(total_elems, 1),
        "mean_chroma": total_chroma / max(total_pixels, 1),
        "sat_mae": (sat_abs_sum / max(sat_count, 1.0)),
    }
    for t, hits in thr_hits.items():
        out[f"acc@{t:g}"] = hits / max(total_elems, 1)
    return out


# =========================
# LR schedule (cosine + warmup)
# =========================
def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================
# Plot
# =========================
def plot_curves(train_acc, val_acc, test_acc, out_path: str):
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


# =========================
# Train
# =========================
def train():
    set_seed(SEED, DETERMINISTIC)
    print("Device:", DEVICE)
    print("Shard dir:", SHARD_DIR.resolve())
    print(f"BATCH_SIZE={BATCH_SIZE} workers={NUM_WORKERS} AMP={USE_AMP} channels_last={CHANNELS_LAST}")
    print(f"Deterministic={DETERMINISTIC}")
    print(f"Split: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}")
    print(f"Loss: chroma_weighted_l1(alpha={CHROMA_LOSS_ALPHA}) | Sat MAE tau={SAT_TAU}")
    if SAT_SAMPLING:
        print(f"Train sampling: saturation-aware (stride={SAT_SCORE_STRIDE}, power={SAT_SAMPLING_POWER}, min_w={SAT_SAMPLING_MIN_WEIGHT}, max_w={SAT_SAMPLING_MAX_WEIGHT})")
        print(f"Sat-score cache: dir={SAT_SCORE_CACHE_DIR} prefix={SAT_SCORE_CACHE_PREFIX}")
    else:
        print("Train sampling: uniform shuffle")

    ds = CocoMMapCropDataset(SHARD_DIR, max_shards=MAX_SHARDS)
    n = len(ds)
    train_idx, val_idx, test_idx = make_split_indices(n, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    print(f"Samples: total={n} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    # Build sampler for train (cached scores)
    if SAT_SAMPLING:
        train_sampler, _weights = build_saturation_sampler(ds, train_idx)
        train_loader = make_loader(train_ds, shuffle=False, sampler=train_sampler, drop_last=True)
    else:
        train_loader = make_loader(train_ds, shuffle=True, sampler=None, drop_last=True)

    val_loader = make_loader(val_ds, shuffle=False, sampler=None, drop_last=False)
    test_loader = make_loader(test_ds, shuffle=False, sampler=None, drop_last=False)

    model = UNetColorizer(base=BASE_CH).to(DEVICE)
    if CHANNELS_LAST and DEVICE.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    amp_enabled = USE_AMP and DEVICE.startswith("cuda")
    scaler = make_scaler(enabled=amp_enabled)

    total_steps = EPOCHS * max(len(train_loader), 1)
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = make_scheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps)

    # Zero baseline reports
    z_val = evaluate_zero_baseline(val_loader, DEVICE, thresholds=ACCURACY_THRESHOLDS)
    z_test = evaluate_zero_baseline(test_loader, DEVICE, thresholds=ACCURACY_THRESHOLDS)
    print("\nZero-ab baseline (predict a=b=0):")
    print(
        f"  VAL : MAE={z_val['mae']:.6f} MSE={z_val['mse']:.6f} RMSE={z_val['rmse']:.6f} "
        f"satMAE={z_val['sat_mae']:.6f} mean_abs_ab={z_val['mean_abs_ab']:.6f} mean_chroma={z_val['mean_chroma']:.6f}"
    )
    print(
        f"  TEST: MAE={z_test['mae']:.6f} MSE={z_test['mse']:.6f} RMSE={z_test['rmse']:.6f} "
        f"satMAE={z_test['sat_mae']:.6f} mean_abs_ab={z_test['mean_abs_ab']:.6f} mean_chroma={z_test['mean_chroma']:.6f}\n"
    )

    best_val_mae = float("inf")

    train_acc_curve, val_acc_curve, test_acc_curve = [], [], []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for step, (L, ab) in enumerate(train_loader, start=1):
            L = L.to(DEVICE, non_blocking=True)
            ab = ab.to(DEVICE, non_blocking=True)

            if CHANNELS_LAST and DEVICE.startswith("cuda"):
                L = L.contiguous(memory_format=torch.channels_last)
                ab = ab.contiguous(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)

            with autocast_ctx(enabled=amp_enabled):
                pred = model(L)
                loss = chroma_weighted_l1(pred, ab, alpha=CHROMA_LOSS_ALPHA)

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(opt)
            scaler.update()
            scheduler.step()

            running += float(loss.item())

            if PRINT_EVERY and (step % PRINT_EVERY == 0):
                lr_now = opt.param_groups[0]["lr"]
                elapsed = human_time(time.time() - start_time)
                print(
                    f"epoch {epoch}/{EPOCHS} step {step}/{len(train_loader)} "
                    f"loss {loss.item():.4f} avg {running/step:.4f} lr {lr_now:.2e} elapsed {elapsed}"
                )

        # Evaluate
        train_metrics = evaluate_metrics(model, train_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
        val_metrics = evaluate_metrics(model, val_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
        test_metrics = evaluate_metrics(model, test_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)

        print(
            f"\nEpoch {epoch}/{EPOCHS} summary:"
            f"\n  Train: MAE={train_metrics['mae']:.6f} RMSE={train_metrics['rmse']:.6f} "
            f"satMAE={train_metrics['sat_mae']:.6f} "
            f"(a={train_metrics['mae_a']:.6f}, b={train_metrics['mae_b']:.6f}) "
            f"NormAcc={train_metrics['norm_acc']:.4f} "
            f"predChroma={train_metrics['mean_pred_chroma']:.6f} gtChroma={train_metrics['mean_gt_chroma']:.6f}"
            f"\n  Val  : MAE={val_metrics['mae']:.6f} RMSE={val_metrics['rmse']:.6f} "
            f"satMAE={val_metrics['sat_mae']:.6f} "
            f"(a={val_metrics['mae_a']:.6f}, b={val_metrics['mae_b']:.6f}) "
            f"NormAcc={val_metrics['norm_acc']:.4f} "
            f"predChroma={val_metrics['mean_pred_chroma']:.6f} gtChroma={val_metrics['mean_gt_chroma']:.6f}"
            f"\n  Test : MAE={test_metrics['mae']:.6f} RMSE={test_metrics['rmse']:.6f} "
            f"satMAE={test_metrics['sat_mae']:.6f} "
            f"(a={test_metrics['mae_a']:.6f}, b={test_metrics['mae_b']:.6f}) "
            f"NormAcc={test_metrics['norm_acc']:.4f} "
            f"predChroma={test_metrics['mean_pred_chroma']:.6f} gtChroma={test_metrics['mean_gt_chroma']:.6f}\n"
        )

        train_acc_curve.append(train_metrics["norm_acc"])
        val_acc_curve.append(val_metrics["norm_acc"])
        test_acc_curve.append(test_metrics["norm_acc"])

        # Save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
            "best_val_mae": best_val_mae,
            "config": {
                "BASE_CH": BASE_CH,
                "LR": LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "CHROMA_LOSS_ALPHA": CHROMA_LOSS_ALPHA,
                "SAT_TAU": SAT_TAU,
                "SAT_SAMPLING": SAT_SAMPLING,
                "SAT_SCORE_STRIDE": SAT_SCORE_STRIDE,
                "SAT_SAMPLING_POWER": SAT_SAMPLING_POWER,
                "SAT_SAMPLING_MIN_WEIGHT": SAT_SAMPLING_MIN_WEIGHT,
                "SAT_SAMPLING_MAX_WEIGHT": SAT_SAMPLING_MAX_WEIGHT,
                "SAT_SCORE_CACHE_DIR": str(SAT_SCORE_CACHE_DIR),
                "SAT_SCORE_CACHE_PREFIX": SAT_SCORE_CACHE_PREFIX,
            },
        }
        torch.save(ckpt, SAVE_LAST_PATH)

        # Save best by validation MAE
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            ckpt["best_val_mae"] = best_val_mae
            torch.save(ckpt, SAVE_BEST_PATH)
            print(f"Saved best -> {SAVE_BEST_PATH} (val_mae={best_val_mae:.6f})")

    # Plot curves (still works for 1 epoch)
    try:
        plot_curves(train_acc_curve, val_acc_curve, test_acc_curve, PLOT_PATH)
        print("Saved plot ->", PLOT_PATH)
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    # Load best and report final metrics
    if SAVE_BEST_PATH.exists():
        best = torch.load(SAVE_BEST_PATH, map_location=DEVICE)
        model.load_state_dict(best["model"])
        print(f"\nLoaded best checkpoint: {SAVE_BEST_PATH} (epoch={best.get('epoch')}, best_val_mae={best.get('best_val_mae')})")

    final_train = evaluate_metrics(model, train_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
    final_val = evaluate_metrics(model, val_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
    final_test = evaluate_metrics(model, test_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)

    def fmt(m):
        parts = [
            f"MAE={m['mae']:.6f}",
            f"MSE={m['mse']:.6f}",
            f"RMSE={m['rmse']:.6f}",
            f"satMAE={m['sat_mae']:.6f}",
            f"NormAcc={m['norm_acc']:.6f}",
            f"MAE_a={m['mae_a']:.6f}",
            f"MAE_b={m['mae_b']:.6f}",
            f"predChroma={m['mean_pred_chroma']:.6f}",
            f"gtChroma={m['mean_gt_chroma']:.6f}",
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
