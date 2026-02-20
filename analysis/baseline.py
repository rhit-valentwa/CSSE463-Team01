from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SHARD_DIR = Path("data/coco/train2017_cache_256_mmap")
BATCH_SIZE = 32
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN_A = 0.020185
MEAN_B = 0.058019

# "Accuracy" for regression needs a tolerance.
# eps is in the same scale as your ab after /128.0 (roughly [-1, 1]).
ACC_EPS = 0.05  # tweak this (e.g., 0.02, 0.05, 0.1)


class CocoMMapABDataset(Dataset):
    def __init__(self, shard_dir: Path):
        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.is_dir():
            raise RuntimeError(f"Shard directory not found: {self.shard_dir.resolve()}")

        self.ab_paths = sorted(self.shard_dir.glob("shard_*_ab.npy"))
        if not self.ab_paths:
            raise RuntimeError(f"Expected shard_*_ab.npy in {self.shard_dir.resolve()}")

        # Global index: global_idx -> (shard_idx, local_idx)
        self._index = []
        for si, p in enumerate(self.ab_paths):
            m = np.load(p, mmap_mode="r")
            n = int(m.shape[0])
            self._index.extend((si, li) for li in range(n))

        # Per-worker cache of opened memmaps
        self._cached_si = None
        self._ab_mmap = None

    def __len__(self):
        return len(self._index)

    def _load_shard(self, si: int):
        if self._cached_si != si:
            self._ab_mmap = np.load(self.ab_paths[si], mmap_mode="r")
            self._cached_si = si
        return self._ab_mmap

    def __getitem__(self, idx):
        si, li = self._index[idx]
        abm = self._load_shard(si)

        ab_i8 = abm[li]  # (2,256,256) int8
        ab = (ab_i8.astype(np.float32) / 128.0)  # -> ~[-1,1]
        return torch.from_numpy(ab)  # (2,256,256)


@torch.no_grad()
def compute_mean_ab_metrics(
    loader: DataLoader,
    device: str,
    mean_a: float,
    mean_b: float,
    acc_eps: float,
):
    total_sq = 0.0
    total_abs = 0.0
    total_elems = 0

    # per-channel MAE
    total_abs_a = 0.0
    total_abs_b = 0.0
    total_elems_per_ch = 0

    # accuracy@eps: count pixel as correct if BOTH channels within eps
    correct_pixels = 0
    total_pixels = 0

    mean = torch.tensor([mean_a, mean_b], device=device, dtype=torch.float32).view(1, 2, 1, 1)

    for ab in loader:  # ab: (N,2,H,W)
        ab = ab.to(device, non_blocking=True)
        pred = mean.expand_as(ab)  # constant prediction per pixel

        err = pred - ab
        abs_err = err.abs()

        total_sq += (err * err).sum().item()
        total_abs += abs_err.sum().item()
        total_elems += err.numel()

        # per-channel MAE
        # abs_err[:, 0] is a channel, abs_err[:, 1] is b channel
        total_abs_a += abs_err[:, 0].sum().item()
        total_abs_b += abs_err[:, 1].sum().item()
        total_elems_per_ch += abs_err[:, 0].numel()  # N*H*W

        # accuracy@eps per pixel: both channels within eps
        within = (abs_err <= acc_eps)                 # (N,2,H,W) bool
        within_both = within[:, 0] & within[:, 1]     # (N,H,W) bool
        correct_pixels += within_both.sum().item()
        total_pixels += within_both.numel()

    mse = total_sq / max(total_elems, 1)
    rmse = float(np.sqrt(mse))
    mae = total_abs / max(total_elems, 1)

    mae_a = total_abs_a / max(total_elems_per_ch, 1)
    mae_b = total_abs_b / max(total_elems_per_ch, 1)

    acc = correct_pixels / max(total_pixels, 1)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mae_a": mae_a,
        "mae_b": mae_b,
        "acc_eps": acc_eps,
        "acc": acc,
    }


def main():
    print("Device:", DEVICE)
    print("Shard dir:", SHARD_DIR.resolve())
    print(f"Baseline mean_a={MEAN_A} mean_b={MEAN_B}")
    print(f"Accuracy tolerance eps={ACC_EPS} (ab scale after /128.0)")

    ds = CocoMMapABDataset(SHARD_DIR)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=False,
    )

    metrics = compute_mean_ab_metrics(loader, DEVICE, MEAN_A, MEAN_B, ACC_EPS)

    print("\nMean-ab baseline results (over all pixels, both channels):")
    print(f"MSE : {metrics['mse']:.8f}")
    print(f"RMSE: {metrics['rmse']:.8f}")
    print(f"MAE : {metrics['mae']:.8f}")
    print(f"MAE_a: {metrics['mae_a']:.8f}")
    print(f"MAE_b: {metrics['mae_b']:.8f}")
    print(f"Accuracy@eps={metrics['acc_eps']}: {metrics['acc']*100:.2f}%  (both channels within eps per pixel)")


if __name__ == "__main__":
    main()