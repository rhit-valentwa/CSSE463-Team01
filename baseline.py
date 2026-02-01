# from pathlib import Path
# import numpy as np

# SHARD_DIR = Path("data/coco/train2017_cache_256_mmap")
# SEED = 42
# TRAIN_RATIO = 0.80
# VAL_RATIO = 0.10
# TEST_RATIO = 0.10

# def make_split_indices(n: int, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
#     if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
#         raise ValueError("Train/val/test ratios must sum to 1.0")
#     rng = np.random.default_rng(seed)
#     perm = rng.permutation(n)

#     n_train = int(n * train_ratio)
#     n_val = int(n * val_ratio)

#     train_idx = perm[:n_train]
#     val_idx = perm[n_train:n_train + n_val]
#     test_idx = perm[n_train + n_val:]
#     return train_idx, val_idx, test_idx

# def compute_train_mean_ab(shard_dir: Path) -> tuple[float, float]:
#     ab_paths = sorted(Path(shard_dir).glob("shard_*_ab.npy"))
#     if not ab_paths:
#         raise RuntimeError(f"No shard_*_ab.npy found in {Path(shard_dir).resolve()}")

#     # Build a global index -> (shard_id, local_id)
#     index = []
#     counts = []
#     for si, p in enumerate(ab_paths):
#         mm = np.load(p, mmap_mode="r")  # (N,2,256,256) int8
#         n = int(mm.shape[0])
#         counts.append(n)
#         index.extend((si, li) for li in range(n))

#     n_total = len(index)
#     train_idx, _, _ = make_split_indices(n_total, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

#     # Streaming sums in float64 for numerical stability
#     sum_a = 0.0
#     sum_b = 0.0
#     num_pixels = 0

#     cached_si = None
#     ab_mmap = None

#     H = W = 256  # your cached crop size
#     pixels_per_img_per_chan = H * W

#     for gi in train_idx:
#         si, li = index[int(gi)]
#         if cached_si != si:
#             ab_mmap = np.load(ab_paths[si], mmap_mode="r")
#             cached_si = si

#         ab_i8 = ab_mmap[li]  # (2,256,256) int8

#         # Sum in int64 first (fast + safe), then scale once
#         sum_a += float(ab_i8[0].sum(dtype=np.int64))
#         sum_b += float(ab_i8[1].sum(dtype=np.int64))
#         num_pixels += pixels_per_img_per_chan

#     # Convert from int8 units to float ab units in [-1,1] via /128.0
#     mean_a = (sum_a / num_pixels) / 128.0
#     mean_b = (sum_b / num_pixels) / 128.0
#     return mean_a, mean_b

# if __name__ == "__main__":
#     mean_a, mean_b = compute_train_mean_ab(SHARD_DIR)
#     print(f"TRAIN mean a: {mean_a:.6f}")
#     print(f"TRAIN mean b: {mean_b:.6f}")

# # TRAIN mean a: 0.020185
# # TRAIN mean b: 0.058019


from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------
# CONFIG (edit here only)
# -----------------------
SHARD_DIR = Path("data/coco/train2017_cache_256_mmap")
BATCH_SIZE = 32
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Your TRAIN set mean ab (in [-1,1] space)
MEAN_A = 0.020185
MEAN_B = 0.058019
# -----------------------


class CocoMMapABDataset(Dataset):
    """
    Loads:
      shard_XXXXX_ab.npy : (N,2,256,256) int8
    via memory-mapping. Returns:
      ab_t : (2,256,256) float32 in [-1,1]
    """
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
def compute_mean_ab_mse(loader: DataLoader, device: str, mean_a: float, mean_b: float):
    total_sq = 0.0
    total_elems = 0

    mean = torch.tensor([mean_a, mean_b], device=device, dtype=torch.float32).view(1, 2, 1, 1)

    for ab in loader:  # ab: (N,2,H,W)
        ab = ab.to(device, non_blocking=True)
        pred = mean.expand_as(ab)  # constant prediction per pixel

        err = pred - ab
        total_sq += (err * err).sum().item()
        total_elems += err.numel()

    mse = total_sq / max(total_elems, 1)
    rmse = float(np.sqrt(mse))
    return mse, rmse


def main():
    print("Device:", DEVICE)
    print("Shard dir:", SHARD_DIR.resolve())
    print(f"Baseline mean_a={MEAN_A} mean_b={MEAN_B}")

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

    mse, rmse = compute_mean_ab_mse(loader, DEVICE, MEAN_A, MEAN_B)

    print("\nMean-ab baseline results (over all pixels, both channels):")
    print(f"MSE : {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")


if __name__ == "__main__":
    main()
