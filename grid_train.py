from pathlib import Path
import os
import time
import json
import csv
import random
import itertools
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
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.999)

LOSS_NAME = "l1"  # "l1" | "huber" | "charbonnier"
HUBER_BETA = 0.1
CHARBONNIER_EPS = 1e-3

SCHEDULER = "none"  # "none" | "cosine" | "plateau"
COSINE_ETA_MIN = 0.0
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 1

GRAD_CLIP_NORM = 0.0  # 0.0 = off

SAVE_MODEL_PATH = "unet_colorizer.pt"
SAVE_BEST_MODEL_PATH = "unet_colorizer_best.pt"
PLOT_PATH = "train_val_test_accuracy.png"

PRINT_EVERY = 200
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFETCH_FACTOR = 4
USE_AMP = True

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

ACCURACY_THRESHOLDS = (0.05, 0.10, 0.20)

torch.backends.cudnn.benchmark = True

GRID_SEARCH = True

GRID_OUT_DIR = Path(".")
STAGE1_CSV = GRID_OUT_DIR / "grid_results_stage1.csv"
STAGE2_CSV = GRID_OUT_DIR / "grid_results_stage2.csv"
BEST_JSON = GRID_OUT_DIR / "grid_best_config.json"

GRID_TRAIN_SUBSAMPLE_N = 1000
GRID_VAL_SUBSAMPLE_N = 1000
GRID_TEST_SUBSAMPLE_N = 1000

GRID_EPOCHS_STAGE1 = 2
GRID_EPOCHS_STAGE2 = 3

GRID_PRINT_EVERY = 0
GRID_DROP_LAST = True

BASELINE_ON_FULL_SPLITS = False  # set True if you want train/test full baseline MAE too

GRID_STAGE1 = {
    "lr": [5e-5, 1e-4, 2e-4, 5e-4],
    "adam_betas": [(0.9, 0.999), (0.5, 0.999)],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "base_ch": [24, 32, 48, 64],
    "loss_name": ["l1", "huber", "charbonnier"],
    "scheduler": ["none", "cosine"],
}

GRID_STAGE2_TEMPLATE = {
    "lr": "around_best",
    "base_ch": "around_best",
    "adam_betas": [(0.5, 0.999), (0.9, 0.999)],
    "weight_decay": [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
    "loss_name": "best_two_from_stage1",
    "huber_beta": [0.05, 0.1, 0.2],
    "charbonnier_eps": [1e-3, 3e-3, 1e-2],
    "scheduler": ["cosine", "plateau"],
    "cosine_eta_min": [0.0, 1e-6],
    "plateau_factor": [0.5, 0.2],
    "plateau_patience": [0, 1],
    "grad_clip_norm": [0.0, 1.0],
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CocoMMapCropDataset(Dataset):
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

        # global_idx -> (shard_idx, local_idx)
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

        L_t = torch.from_numpy(L).unsqueeze(0)  # (1,256,256)
        ab_t = torch.from_numpy(ab)             # (2,256,256)
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

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def make_subset_indices(full_indices, n_subsample, seed):
    if n_subsample is None or n_subsample <= 0 or n_subsample >= len(full_indices):
        return list(full_indices)
    rng = np.random.default_rng(seed)
    return rng.choice(full_indices, size=n_subsample, replace=False).tolist()


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

        elems = abs_err.numel()
        total_elems += elems
        total_abs += float(abs_err.sum().item())
        total_sq += float(sq_err.sum().item())

        for t in thr_hits:
            thr_hits[t] += int((abs_err < t).sum().item())

    mae = total_abs / max(total_elems, 1)
    mse = total_sq / max(total_elems, 1)
    rmse = float(np.sqrt(mse))

    # ab in [-1,1], max abs err per element is 2
    norm_acc = 1.0 - (mae / 2.0)
    norm_acc = max(0.0, min(1.0, norm_acc))

    out = {"mae": mae, "mse": mse, "rmse": rmse, "norm_acc": norm_acc}
    for t, hits in thr_hits.items():
        out[f"acc@{t:g}"] = hits / max(total_elems, 1)
    return out


@torch.no_grad()
def zero_baseline_mae(loader: DataLoader, device: str) -> float:
    total_abs = 0.0
    total_elems = 0
    for _, ab in loader:
        ab = ab.to(device, non_blocking=True)
        total_abs += float(ab.abs().sum().item())
        total_elems += ab.numel()
    return total_abs / max(total_elems, 1)


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


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + (self.eps * self.eps)))


def make_loss(loss_name: str, huber_beta: float, charbonnier_eps: float):
    ln = (loss_name or "l1").lower()
    if ln == "l1":
        return nn.L1Loss()
    if ln in ("huber", "smoothl1", "smooth_l1"):
        # SmoothL1Loss supports beta in modern PyTorch; use as Huber-like
        return nn.SmoothL1Loss(beta=float(huber_beta))
    if ln == "charbonnier":
        return CharbonnierLoss(eps=float(charbonnier_eps))
    raise ValueError(f"Unknown loss_name: {loss_name}")

def build_loader(ds, indices, batch_size, shuffle, drop_last):
    return DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=drop_last,
    )


def build_loaders(ds, train_idx, val_idx, test_idx, batch_size, drop_last_train: bool):
    train_loader = build_loader(ds, train_idx, batch_size, shuffle=True, drop_last=drop_last_train)
    val_loader = build_loader(ds, val_idx, batch_size, shuffle=False, drop_last=False)
    test_loader = build_loader(ds, test_idx, batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def write_csv_row(path: Path, row: dict):
    ensure_dir(path)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def save_json(path: Path, obj: dict):
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def expand_grid(grid: dict):
    keys = sorted(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def make_lr_around(best_lr: float):
    factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    lrs = sorted({float(best_lr) * f for f in factors})
    return [lr for lr in lrs if 1e-6 <= lr <= 5e-3]


def make_basech_around(best_base: int):
    cands = sorted({int(best_base), int(best_base - 8), int(best_base + 8), int(best_base - 16), int(best_base + 16)})
    return [c for c in cands if c >= 8]


def top_k_loss_names_by_best_val_mae(stage_results: list, k=2):
    best_per_loss = {}
    for r in stage_results:
        ln = r["loss_name"]
        v = float(r["val_mae"])
        if (ln not in best_per_loss) or (v < best_per_loss[ln]):
            best_per_loss[ln] = v
    ranked = sorted(best_per_loss.items(), key=lambda x: x[1])
    return [name for name, _ in ranked[:k]]


def canonicalize_config(cfg: dict):
    out = {}

    out["lr"] = float(cfg.get("lr", LR))
    out["base_ch"] = int(cfg.get("base_ch", BASE_CH))
    out["weight_decay"] = float(cfg.get("weight_decay", WEIGHT_DECAY))
    out["adam_betas"] = tuple(cfg.get("adam_betas", ADAM_BETAS))

    out["loss_name"] = str(cfg.get("loss_name", LOSS_NAME)).lower()
    out["huber_beta"] = float(cfg.get("huber_beta", HUBER_BETA))
    out["charbonnier_eps"] = float(cfg.get("charbonnier_eps", CHARBONNIER_EPS))

    out["scheduler"] = str(cfg.get("scheduler", SCHEDULER)).lower()
    out["cosine_eta_min"] = float(cfg.get("cosine_eta_min", COSINE_ETA_MIN))
    out["plateau_factor"] = float(cfg.get("plateau_factor", PLATEAU_FACTOR))
    out["plateau_patience"] = int(cfg.get("plateau_patience", PLATEAU_PATIENCE))

    out["grad_clip_norm"] = float(cfg.get("grad_clip_norm", GRAD_CLIP_NORM))
    out["batch_size"] = int(cfg.get("batch_size", BATCH_SIZE))

    # Drop conditionals
    if out["loss_name"] != "huber":
        out.pop("huber_beta", None)
    if out["loss_name"] != "charbonnier":
        out.pop("charbonnier_eps", None)

    if out["scheduler"] != "cosine":
        out.pop("cosine_eta_min", None)
    if out["scheduler"] != "plateau":
        out.pop("plateau_factor", None)
        out.pop("plateau_patience", None)

    return out

def make_optimizer(model: nn.Module, cfg: dict):
    betas = cfg.get("adam_betas", ADAM_BETAS)
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", LR),
        betas=betas,
        weight_decay=cfg.get("weight_decay", WEIGHT_DECAY),
    )


def make_scheduler(opt, cfg: dict, steps_per_epoch: int, epochs: int):
    sch = (cfg.get("scheduler", "none") or "none").lower()
    if sch == "none":
        return None, "none"

    if sch == "cosine":
        # step per batch
        t_max = max(1, steps_per_epoch * epochs)
        eta_min = float(cfg.get("cosine_eta_min", COSINE_ETA_MIN))
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min), "cosine"

    if sch == "plateau":
        factor = float(cfg.get("plateau_factor", PLATEAU_FACTOR))
        patience = int(cfg.get("plateau_patience", PLATEAU_PATIENCE))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=factor, patience=patience, verbose=False), "plateau"

    raise ValueError(f"Unknown scheduler: {sch}")


def train_for_epochs(train_loader, val_loader, cfg: dict, epochs: int, device: str, use_amp: bool, print_every: int):
    model = UNetColorizer(base=cfg["base_ch"]).to(device)
    opt = make_optimizer(model, cfg)

    loss_fn = make_loss(
        cfg["loss_name"],
        huber_beta=cfg.get("huber_beta", HUBER_BETA),
        charbonnier_eps=cfg.get("charbonnier_eps", CHARBONNIER_EPS),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and str(device).startswith("cuda")))
    steps_per_epoch = max(1, len(train_loader))
    scheduler, scheduler_name = make_scheduler(opt, cfg, steps_per_epoch=steps_per_epoch, epochs=epochs)

    amp_enabled = use_amp and str(device).startswith("cuda")
    grad_clip = float(cfg.get("grad_clip_norm", 0.0))

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for step, (L, ab) in enumerate(train_loader, start=1):
            global_step += 1
            L = L.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(L)
                loss = loss_fn(pred, ab)

            scaler.scale(loss).backward()

            if grad_clip > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(opt)
            scaler.update()

            # Scheduler step
            if scheduler is not None and scheduler_name == "cosine":
                scheduler.step()

            running += float(loss.item())
            if print_every and (step % print_every == 0):
                print(
                    f"[grid] epoch {epoch}/{epochs} step {step}/{len(train_loader)} "
                    f"loss {loss.item():.4f} avg {running/step:.4f}"
                )

        # Plateau scheduler steps per epoch on val MAE
        val_metrics = evaluate_metrics(model, val_loader, device, use_amp, thresholds=ACCURACY_THRESHOLDS)
        if scheduler is not None and scheduler_name == "plateau":
            scheduler.step(val_metrics["mae"])

    return model, val_metrics

def run_stage_grid(stage_name: str, csv_path: Path, ds, train_idx, val_idx, grid: dict, epochs: int):
    stage_results = []

    runs = list(expand_grid(grid))
    print(f"\n{stage_name}: {len(runs)} configs | train_n={len(train_idx)} val_n={len(val_idx)} | epochs={epochs}")

    best_val = float("inf")
    best_cfg = None

    for i, raw_cfg in enumerate(runs, start=1):
        cfg = canonicalize_config(raw_cfg)

        bs = int(cfg.get("batch_size", BATCH_SIZE))
        train_loader = build_loader(ds, train_idx, bs, shuffle=True, drop_last=GRID_DROP_LAST)
        val_loader = build_loader(ds, val_idx, bs, shuffle=False, drop_last=False)

        t0 = time.time()
        _, val_metrics = train_for_epochs(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            epochs=epochs,
            device=DEVICE,
            use_amp=USE_AMP,
            print_every=GRID_PRINT_EVERY,
        )
        dt = time.time() - t0

        record = {
            "stage": stage_name,
            "run_idx": i,
            "total_runs": len(runs),
            "seconds": round(dt, 3),

            # params (flattened key ones)
            "lr": cfg.get("lr"),
            "base_ch": cfg.get("base_ch"),
            "weight_decay": cfg.get("weight_decay"),
            "adam_betas": str(cfg.get("adam_betas")),
            "loss_name": cfg.get("loss_name"),
            "huber_beta": cfg.get("huber_beta", ""),
            "charbonnier_eps": cfg.get("charbonnier_eps", ""),
            "scheduler": cfg.get("scheduler"),
            "cosine_eta_min": cfg.get("cosine_eta_min", ""),
            "plateau_factor": cfg.get("plateau_factor", ""),
            "plateau_patience": cfg.get("plateau_patience", ""),
            "grad_clip_norm": cfg.get("grad_clip_norm", 0.0),
            "batch_size": cfg.get("batch_size", BATCH_SIZE),

            # metrics
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_norm_acc": val_metrics["norm_acc"],
        }

        # also store JSON config for full reproducibility
        record["config_json"] = json.dumps(cfg, sort_keys=True)

        write_csv_row(csv_path, record)

        stage_results.append({
            "loss_name": cfg.get("loss_name"),
            "val_mae": float(val_metrics["mae"]),
            "config": cfg,
        })

        print(
            f"[{i:>3}/{len(runs)}] "
            f"lr={cfg['lr']:.1e} base_ch={cfg['base_ch']} wd={cfg['weight_decay']:.1e} "
            f"betas={cfg['adam_betas']} loss={cfg['loss_name']:<11} sch={cfg['scheduler']:<7} "
            f"=> val_mae={val_metrics['mae']:.6f} val_acc={val_metrics['norm_acc']:.4f} ({dt:.1f}s)"
        )

        if float(val_metrics["mae"]) < best_val:
            best_val = float(val_metrics["mae"])
            best_cfg = dict(cfg)

    print(f"{stage_name} best: val_mae={best_val:.6f} cfg={best_cfg}")
    return best_cfg, best_val, stage_results


def build_stage2_grid(best_cfg: dict, stage1_results: list):
    best_lr = float(best_cfg["lr"])
    best_base = int(best_cfg["base_ch"])

    lr_list = make_lr_around(best_lr)
    base_list = make_basech_around(best_base)

    best_two_losses = top_k_loss_names_by_best_val_mae(stage1_results, k=2)
    if not best_two_losses:
        best_two_losses = [best_cfg.get("loss_name", "l1")]

    g = dict(GRID_STAGE2_TEMPLATE)

    # resolve special tokens
    g["lr"] = lr_list
    g["base_ch"] = base_list
    g["loss_name"] = best_two_losses

    # Ensure all non-list values are lists
    grid2 = {}
    for k, v in g.items():
        if isinstance(v, str):
            raise RuntimeError(f"Stage2 grid unresolved key {k}={v}")
        grid2[k] = v if isinstance(v, list) else list(v)

    return grid2

def train_full(ds, train_idx, val_idx, test_idx, cfg: dict):
    cfg = canonicalize_config(cfg)

    train_loader, val_loader, test_loader = build_loaders(
        ds,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=int(cfg.get("batch_size", BATCH_SIZE)),
        drop_last_train=True,
    )

    model = UNetColorizer(base=cfg["base_ch"]).to(DEVICE)
    opt = make_optimizer(model, cfg)

    loss_fn = make_loss(
        cfg["loss_name"],
        huber_beta=cfg.get("huber_beta", HUBER_BETA),
        charbonnier_eps=cfg.get("charbonnier_eps", CHARBONNIER_EPS),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.startswith("cuda")))
    steps_per_epoch = max(1, len(train_loader))
    scheduler, scheduler_name = make_scheduler(opt, cfg, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)

    amp_enabled = USE_AMP and DEVICE.startswith("cuda")
    grad_clip = float(cfg.get("grad_clip_norm", 0.0))

    best_val = float("inf")
    train_acc_curve, val_acc_curve, test_acc_curve = [], [], []

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for step, (L, ab) in enumerate(train_loader, start=1):
            global_step += 1
            L = L.to(DEVICE, non_blocking=True)
            ab = ab.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(L)
                loss = loss_fn(pred, ab)

            scaler.scale(loss).backward()

            if grad_clip > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(opt)
            scaler.update()

            if scheduler is not None and scheduler_name == "cosine":
                scheduler.step()

            running += float(loss.item())
            if PRINT_EVERY and (step % PRINT_EVERY == 0):
                print(
                    f"epoch {epoch}/{EPOCHS} step {step}/{len(train_loader)} "
                    f"train_batch_loss {loss.item():.4f} train_avg {running/step:.4f}"
                )

        train_epoch_loss = running / max(len(train_loader), 1)

        train_metrics = evaluate_metrics(model, train_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
        val_metrics = evaluate_metrics(model, val_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)
        test_metrics = evaluate_metrics(model, test_loader, DEVICE, USE_AMP, thresholds=ACCURACY_THRESHOLDS)

        if scheduler is not None and scheduler_name == "plateau":
            scheduler.step(val_metrics["mae"])

        print(
            f"epoch {epoch}/{EPOCHS} "
            f"train_loss(batchavg) {train_epoch_loss:.4f} | "
            f"train_mae {train_metrics['mae']:.4f} val_mae {val_metrics['mae']:.4f} test_mae {test_metrics['mae']:.4f} | "
            f"train_acc {train_metrics['norm_acc']:.4f} val_acc {val_metrics['norm_acc']:.4f} test_acc {test_metrics['norm_acc']:.4f}"
        )

        train_acc_curve.append(train_metrics["norm_acc"])
        val_acc_curve.append(val_metrics["norm_acc"])
        test_acc_curve.append(test_metrics["norm_acc"])

        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            torch.save(model.state_dict(), SAVE_BEST_MODEL_PATH)
            print(f"  saved best -> {SAVE_BEST_MODEL_PATH} (val_mae={best_val:.6f})")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Saved final model ->", SAVE_MODEL_PATH)

    plot_curves(train_acc_curve, val_acc_curve, test_acc_curve, PLOT_PATH)
    print("Saved plot ->", PLOT_PATH)

    # Final report on BEST checkpoint
    if os.path.exists(SAVE_BEST_MODEL_PATH):
        model.load_state_dict(torch.load(SAVE_BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded best checkpoint: {SAVE_BEST_MODEL_PATH} (best val_mae={best_val:.6f})")

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


def main():
    set_seed(SEED)
    print("Device:", DEVICE)
    print("Shard dir:", SHARD_DIR.resolve())
    print(f"BATCH_SIZE={BATCH_SIZE} workers={NUM_WORKERS} AMP={USE_AMP}")
    print(f"Split: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}")

    ds = CocoMMapCropDataset(SHARD_DIR)
    n = len(ds)
    train_idx, val_idx, test_idx = make_split_indices(n, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    print(f"Samples: total={n} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_idx_1k = make_subset_indices(train_idx, GRID_TRAIN_SUBSAMPLE_N, seed=SEED + 10)
    test_idx_1k = make_subset_indices(test_idx, GRID_TEST_SUBSAMPLE_N, seed=SEED + 30)

    train0_loader_1k = build_loader(ds, train_idx_1k, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test0_loader_1k = build_loader(ds, test_idx_1k, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    train0_1k = zero_baseline_mae(train0_loader_1k, DEVICE)
    test0_1k = zero_baseline_mae(test0_loader_1k, DEVICE)

    print(f"\nZero-pred baseline MAE (train 1K): {train0_1k:.6f}")
    print(f"Zero-pred baseline MAE (test  1K): {test0_1k:.6f}")

    if BASELINE_ON_FULL_SPLITS:
        train0_full_loader = build_loader(ds, train_idx, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test0_full_loader = build_loader(ds, test_idx, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        train0_full = zero_baseline_mae(train0_full_loader, DEVICE)
        test0_full = zero_baseline_mae(test0_full_loader, DEVICE)
        print(f"\nZero-pred baseline MAE (train full): {train0_full:.6f}")
        print(f"Zero-pred baseline MAE (test  full): {test0_full:.6f}")

    if GRID_SEARCH:
        # Subsample train/val for grid
        train_idx_g = make_subset_indices(train_idx, GRID_TRAIN_SUBSAMPLE_N, seed=SEED + 100)
        val_idx_g = make_subset_indices(val_idx, GRID_VAL_SUBSAMPLE_N, seed=SEED + 200)

        # Stage 1
        best1_cfg, best1_val, stage1_results = run_stage_grid(
            stage_name="stage1",
            csv_path=STAGE1_CSV,
            ds=ds,
            train_idx=train_idx_g,
            val_idx=val_idx_g,
            grid=GRID_STAGE1,
            epochs=GRID_EPOCHS_STAGE1,
        )

        # Stage 2 grid built around best stage1 + best loss types
        grid2 = build_stage2_grid(best1_cfg, stage1_results)
        best2_cfg, best2_val, stage2_results = run_stage_grid(
            stage_name="stage2",
            csv_path=STAGE2_CSV,
            ds=ds,
            train_idx=train_idx_g,
            val_idx=val_idx_g,
            grid=grid2,
            epochs=GRID_EPOCHS_STAGE2,
        )

        best_cfg = best2_cfg if best2_cfg is not None else best1_cfg
        save_json(BEST_JSON, {"best_config": best_cfg, "best_val_mae": float(best2_val if best2_cfg else best1_val)})
        print(f"\nSaved best config -> {BEST_JSON}")
    else:
        best_cfg = canonicalize_config({
            "lr": LR,
            "base_ch": BASE_CH,
            "weight_decay": WEIGHT_DECAY,
            "adam_betas": ADAM_BETAS,
            "loss_name": LOSS_NAME,
            "huber_beta": HUBER_BETA,
            "charbonnier_eps": CHARBONNIER_EPS,
            "scheduler": SCHEDULER,
            "cosine_eta_min": COSINE_ETA_MIN,
            "plateau_factor": PLATEAU_FACTOR,
            "plateau_patience": PLATEAU_PATIENCE,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "batch_size": BATCH_SIZE,
        })

    print("\nTraining full model with config:")
    print(json.dumps(best_cfg, indent=2, sort_keys=True))
    train_full(ds, train_idx, val_idx, test_idx, best_cfg)


if __name__ == "__main__":
    main()
