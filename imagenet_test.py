# imagenet_test_cache_eval.py
# (cache-based eval: baseline ab=0 vs model; minibatched; saves RGB mosaics)
# UPDATED: uses custom LSOS metric in addition to (optional) ΔE76

from __future__ import annotations
from pathlib import Path
import json, math, csv
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import color, io

# =========================
# CONFIG (edit only this)
# =========================
IMAGENET_TEST_CACHE_DIR = Path("data/ImageNet/test_cache_256_mmap")
OUTPUT_DIR = Path("eval_outputs_cache")

CHECKPOINT_PATHS = [
    "unet_colorizer_best.pt",
    "unet_colorizer_last.pt",
    "unet_colorizer.pt",
]
BASE_CH = 32

MAX_SAMPLES = 5000      # set None to run all crops
SEED = 42

NUM_EXAMPLES = 12
EXAMPLE_STRIDE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
PAD_MULTIPLE = 16
CHANNELS_LAST = True

RUN_ALL_CHECKPOINTS = False

BATCH_SIZE = 32         # prevents OOM / all-fail

# --- Custom LSOS (your definition) ---
# gt_chroma = sqrt(a^2 + b^2 + eps)
# pixel_weight = 1 + alpha * gt_chroma
# loss = mean(pixel_weight * |pred_ab - gt_ab|)
ALPHA = 0.05
EPS = 1e-6

# Optional: keep ΔE76 too (nice for interpretability)
COMPUTE_DELTAE76 = True
# =========================


# -------------------------
# Model (must match training)
# -------------------------
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
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# -------------------------
# AMP
# -------------------------
def autocast_ctx(enabled: bool):
    device_type = "cuda" if str(DEVICE).startswith("cuda") else "cpu"
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


# -------------------------
# Custom LSOS
# -------------------------
def lsos_loss(pred_ab: torch.Tensor, gt_ab: torch.Tensor, alpha: float, eps: float) -> torch.Tensor:
    """
    pred_ab, gt_ab: (B,2,H,W) in Lab units
    """
    gt_chroma = torch.sqrt(gt_ab[:, 0] ** 2 + gt_ab[:, 1] ** 2 + eps)  # (B,H,W)
    pixel_weight = 1.0 + alpha * gt_chroma                             # (B,H,W)
    l1 = torch.abs(pred_ab - gt_ab)                                    # (B,2,H,W)
    return torch.mean(pixel_weight.unsqueeze(1) * l1)


# -------------------------
# Checkpoint loading
# -------------------------
def _unwrap_checkpoint(obj):
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        for v in obj.values():
            if torch.is_tensor(v):
                return obj
    raise RuntimeError("Unrecognized checkpoint format.")


def load_model_from_checkpoint(model: nn.Module, ckpt_path: str, device: str) -> None:
    obj = torch.load(ckpt_path, map_location=device)
    state = _unwrap_checkpoint(obj)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"  [warn] strict load failed: {e}")
        model.load_state_dict(state, strict=False)


def choose_checkpoint_paths() -> List[str]:
    existing = [p for p in CHECKPOINT_PATHS if Path(p).exists()]
    if not existing:
        raise RuntimeError("No checkpoints found.")

    if RUN_ALL_CHECKPOINTS:
        return existing

    for p in existing:
        try:
            m = UNetColorizer(base=BASE_CH).to(DEVICE)
            if CHANNELS_LAST and str(DEVICE).startswith("cuda"):
                m = m.to(memory_format=torch.channels_last)
            load_model_from_checkpoint(m, p, str(DEVICE))
            return [p]
        except Exception as e:
            print(f"[skip] {p} failed: {e}")
    raise RuntimeError("No checkpoint could be loaded.")


# -------------------------
# Cache helpers
# -------------------------
def list_shards(cache_dir: Path) -> List[int]:
    shards = []
    for p in cache_dir.glob("shard_*_L.npy"):
        idx = int(p.stem.split("_")[1])
        shards.append(idx)
    shards.sort()
    return shards


def load_shard(cache_dir: Path, shard_idx: int):
    L = np.load(cache_dir / f"shard_{shard_idx:05d}_L.npy")                 # uint8 (N,H,W)
    ab = np.load(cache_dir / f"shard_{shard_idx:05d}_ab.npy")               # int8  (N,2,H,W)
    ids = np.load(cache_dir / f"shard_{shard_idx:05d}_ids.npy", allow_pickle=True)
    return L, ab, ids


def L_u8_to_Lstar(L_u8: np.ndarray) -> np.ndarray:
    return L_u8.astype(np.float32) * (100.0 / 255.0)


def lab_to_rgb_u8(lab: np.ndarray) -> np.ndarray:
    rgb01 = color.lab2rgb(lab).astype(np.float32)
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return (rgb01 * 255.0 + 0.5).astype(np.uint8)


def hstack_u8(imgs: List[np.ndarray], gap: int = 8) -> np.ndarray:
    h = imgs[0].shape[0]
    gap_col = np.full((h, gap, 3), 255, dtype=np.uint8)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.concatenate([out, gap_col, im], axis=1)
    return out


# -------------------------
# Inference
# -------------------------
@torch.inference_mode()
def predict_ab01_batch(model: nn.Module, L01_bhw: np.ndarray) -> np.ndarray:
    model.eval()
    amp_enabled = USE_AMP and str(DEVICE).startswith("cuda")

    x = torch.from_numpy(L01_bhw).float().unsqueeze(1).to(DEVICE)  # (B,1,H,W)
    if CHANNELS_LAST and str(DEVICE).startswith("cuda"):
        x = x.contiguous(memory_format=torch.channels_last)

    b, _, h, w = x.shape
    pad_h = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
    pad_w = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE

    pad_mode = "reflect"
    if (h < 2 and pad_h > 0) or (w < 2 and pad_w > 0):
        pad_mode = "replicate"

    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

    with autocast_ctx(enabled=amp_enabled):
        ab_pad = model(x_pad)  # (B,2,Hp,Wp)

    ab = ab_pad[:, :, :h, :w].permute(0, 2, 3, 1).contiguous()  # (B,H,W,2)
    return ab.detach().cpu().numpy().astype(np.float32)


# -------------------------
# Optional ΔE76
# -------------------------
def deltaE76_mean(lab_pred: np.ndarray, lab_gt: np.ndarray) -> float:
    d = lab_pred - lab_gt
    return float(np.mean(np.sqrt(np.sum(d * d, axis=2))))


# -------------------------
# Main
# -------------------------
def main():
    print("Device:", DEVICE)
    if not IMAGENET_TEST_CACHE_DIR.exists():
        raise RuntimeError(f"Cache dir not found: {IMAGENET_TEST_CACHE_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shard_indices = list_shards(IMAGENET_TEST_CACHE_DIR)
    if not shard_indices:
        raise RuntimeError("No shards found.")

    # compute shard sizes + global offsets
    shard_sizes: Dict[int, int] = {}
    total_available = 0
    for si in shard_indices:
        Lm = np.load(IMAGENET_TEST_CACHE_DIR / f"shard_{si:05d}_L.npy", mmap_mode="r")
        shard_sizes[si] = int(Lm.shape[0])
        total_available += int(Lm.shape[0])

    rng = np.random.default_rng(SEED)
    target = total_available if MAX_SAMPLES is None else min(int(MAX_SAMPLES), total_available)
    print(f"Shards: {len(shard_indices)} | total samples: {total_available} | evaluating: {target}")

    # sample global indices
    chosen = np.arange(total_available, dtype=np.int64)
    if target < total_available:
        chosen = rng.choice(chosen, size=target, replace=False)
    chosen.sort()

    # shard start offsets
    starts: List[Tuple[int, int]] = []
    running = 0
    for si in shard_indices:
        starts.append((si, running))
        running += shard_sizes[si]

    def locate(g: int) -> Tuple[int, int]:
        for j in range(len(starts) - 1, -1, -1):
            si, st = starts[j]
            if g >= st:
                return si, int(g - st)
        raise RuntimeError("locate failed")

    # group by shard
    by_shard: Dict[int, List[int]] = {}
    for g in chosen:
        si, ii = locate(int(g))
        by_shard.setdefault(si, []).append(ii)

    # pick examples (global indices)
    example_set = set()
    if NUM_EXAMPLES > 0:
        candidates = chosen[::max(1, EXAMPLE_STRIDE)]
        n = min(NUM_EXAMPLES, len(candidates))
        ex = rng.choice(candidates, size=n, replace=False) if n > 0 else []
        for g in ex:
            example_set.add(int(g))

    ckpt_paths = choose_checkpoint_paths()

    for ckpt_path in ckpt_paths:
        ckpt_tag = Path(ckpt_path).stem
        ckpt_out = OUTPUT_DIR / ckpt_tag
        examples_out = ckpt_out / "examples"
        ckpt_out.mkdir(parents=True, exist_ok=True)
        examples_out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Evaluating checkpoint: {ckpt_path} ===")

        model = UNetColorizer(base=BASE_CH).to(DEVICE)
        if CHANNELS_LAST and str(DEVICE).startswith("cuda"):
            model = model.to(memory_format=torch.channels_last)
        load_model_from_checkpoint(model, ckpt_path, str(DEVICE))
        model.eval()

        rows: List[Dict[str, object]] = []

        # LSOS accumulators (weighted by pixels)
        sum_px = 0
        sum_lsos_base = 0.0
        sum_lsos_model = 0.0

        # Optional ΔE accumulators (weighted by pixels)
        sum_de_base = 0.0
        sum_de_model = 0.0

        ok = fail = 0
        ex_saved = 0
        first_errors_printed = 0
        processed = 0

        for si in shard_indices:
            idxs = by_shard.get(si)
            if not idxs:
                continue

            try:
                L_u8, ab_i8, ids = load_shard(IMAGENET_TEST_CACHE_DIR, si)
            except Exception as e:
                for ii in idxs:
                    rows.append({"shard": si, "idx": int(ii), "error": f"shard_load_failed: {e}"})
                    fail += 1
                continue

            shard_start = next(st for sj, st in starts if sj == si)

            for b0 in range(0, len(idxs), BATCH_SIZE):
                bidxs = idxs[b0:b0 + BATCH_SIZE]

                try:
                    L_sel = L_u8[bidxs]   # (B,H,W) uint8
                    ab_sel = ab_i8[bidxs] # (B,2,H,W) int8
                    id_sel = ids[bidxs]
                except Exception as e:
                    for ii in bidxs:
                        rows.append({"shard": si, "idx": int(ii), "error": f"indexing_failed: {e}"})
                        fail += 1
                    continue

                B, H, W = L_sel.shape

                # Cached L* (0..100) for ΔE and RGB examples
                L_star = L_u8_to_Lstar(L_sel)  # (B,H,W)

                # Model input L in [0,1]
                L01 = (L_sel.astype(np.float32) / 255.0).astype(np.float32)

                try:
                    ab01_pred = predict_ab01_batch(model, L01)            # (B,H,W,2) in [-1,1]
                except Exception as e:
                    if first_errors_printed < 3:
                        print(f"\n[ERROR] model_failed on shard {si} batch starting {b0}: {repr(e)}")
                        first_errors_printed += 1
                    for k in range(B):
                        rows.append({"id": str(id_sel[k]), "shard": si, "idx": int(bidxs[k]), "error": f"model_failed: {e}"})
                        fail += 1
                    if str(DEVICE).startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue

                # ---- LSOS (torch) computed for whole batch (fast) ----
                # GT ab in Lab units
                gt_ab_t = torch.from_numpy(ab_sel.astype(np.float32)).to(DEVICE)        # (B,2,H,W)
                # baseline ab = 0
                base_ab_t = torch.zeros_like(gt_ab_t)                                  # (B,2,H,W)
                # model ab in Lab units
                pred_ab_t = torch.from_numpy((ab01_pred * 128.0).astype(np.float32)).to(DEVICE)  # (B,H,W,2)
                pred_ab_t = pred_ab_t.permute(0, 3, 1, 2).contiguous()                  # (B,2,H,W)

                base_lsos_t = lsos_loss(base_ab_t, gt_ab_t, alpha=ALPHA, eps=EPS)
                model_lsos_t = lsos_loss(pred_ab_t, gt_ab_t, alpha=ALPHA, eps=EPS)

                batch_px = int(B * H * W)
                sum_px += batch_px
                sum_lsos_base += float(base_lsos_t.item()) * batch_px
                sum_lsos_model += float(model_lsos_t.item()) * batch_px

                # Optional ΔE (computed per-sample on CPU since it needs Lab->RGB anyway for examples)
                if COMPUTE_DELTAE76:
                    # Build numpy GT and predictions in Lab HWC
                    ab_gt_hwc = np.transpose(ab_sel.astype(np.float32), (0, 2, 3, 1))   # (B,H,W,2)
                    ab_base_hwc = np.zeros_like(ab_gt_hwc, dtype=np.float32)
                    ab_model_hwc = (ab01_pred * 128.0).astype(np.float32)

                    for k in range(B):
                        lab_gt = np.stack([L_star[k], ab_gt_hwc[k][..., 0], ab_gt_hwc[k][..., 1]], axis=2).astype(np.float32)
                        lab_base = np.stack([L_star[k], 0.0 * L_star[k], 0.0 * L_star[k]], axis=2).astype(np.float32)
                        lab_pred = np.stack([L_star[k], ab_model_hwc[k][..., 0], ab_model_hwc[k][..., 1]], axis=2).astype(np.float32)

                        base_de = deltaE76_mean(lab_base, lab_gt)
                        pred_de = deltaE76_mean(lab_pred, lab_gt)

                        sum_de_base += float(base_de) * (H * W)
                        sum_de_model += float(pred_de) * (H * W)

                        # Save per-sample row (includes LSOS batch value duplicated for convenience)
                        rows.append({
                            "id": str(id_sel[k]),
                            "shard": int(si),
                            "idx": int(bidxs[k]),
                            "h": int(H),
                            "w": int(W),
                            "baseline_lsos": float(base_lsos_t.item()),
                            "model_lsos": float(model_lsos_t.item()),
                            "baseline_deltaE76_mean": float(base_de),
                            "model_deltaE76_mean": float(pred_de),
                        })

                        ok += 1

                        # save examples
                        g = int(shard_start + bidxs[k])
                        if (g in example_set) and (ex_saved < NUM_EXAMPLES):
                            rgb_base = lab_to_rgb_u8(lab_base)
                            rgb_pred = lab_to_rgb_u8(lab_pred)
                            rgb_gt_u8 = lab_to_rgb_u8(lab_gt)
                            mosaic = hstack_u8([rgb_base, rgb_pred, rgb_gt_u8], gap=8)
                            out_name = f"example_{ex_saved:03d}__id_{str(id_sel[k])}__base_model_gt.png"
                            io.imsave(str(examples_out / out_name), mosaic)
                            ex_saved += 1
                else:
                    # If not computing ΔE, still write per-sample LSOS info + examples
                    for k in range(B):
                        rows.append({
                            "id": str(id_sel[k]),
                            "shard": int(si),
                            "idx": int(bidxs[k]),
                            "h": int(H),
                            "w": int(W),
                            "baseline_lsos": float(base_lsos_t.item()),
                            "model_lsos": float(model_lsos_t.item()),
                        })
                        ok += 1

                        g = int(shard_start + bidxs[k])
                        if (g in example_set) and (ex_saved < NUM_EXAMPLES):
                            # Create lab_base/lab_pred/lab_gt for mosaics
                            ab_gt_hwc = np.transpose(ab_sel.astype(np.float32), (0, 2, 3, 1))
                            lab_gt = np.stack([L_star[k], ab_gt_hwc[k][..., 0], ab_gt_hwc[k][..., 1]], axis=2).astype(np.float32)
                            lab_base = np.stack([L_star[k], 0.0 * L_star[k], 0.0 * L_star[k]], axis=2).astype(np.float32)
                            lab_pred = np.stack([L_star[k], (ab01_pred[k] * 128.0)[..., 0], (ab01_pred[k] * 128.0)[..., 1]], axis=2).astype(np.float32)

                            mosaic = hstack_u8([lab_to_rgb_u8(lab_base), lab_to_rgb_u8(lab_pred), lab_to_rgb_u8(lab_gt)], gap=8)
                            out_name = f"example_{ex_saved:03d}__id_{str(id_sel[k])}__base_model_gt.png"
                            io.imsave(str(examples_out / out_name), mosaic)
                            ex_saved += 1

                processed += B
                if processed % 512 == 0 or processed >= target:
                    print(f"\r[{ckpt_tag}] processed {min(processed, target)}/{target} ok={ok} fail={fail} examples={ex_saved}",
                          end="", flush=True)

        print()

        if ok == 0 or sum_px == 0:
            raise RuntimeError(
                "No successful samples processed.\n"
                "Check the printed [ERROR] lines above; if you see CUDA OOM, lower BATCH_SIZE."
            )

        baseline_lsos = sum_lsos_base / float(sum_px)
        model_lsos = sum_lsos_model / float(sum_px)

        summary: Dict[str, object] = {
            "checkpoint": ckpt_path,
            "checkpoint_tag": ckpt_tag,
            "cache_dir": str(IMAGENET_TEST_CACHE_DIR),
            "device": str(DEVICE),
            "batch_size": int(BATCH_SIZE),
            "num_samples_target": int(target),
            "num_samples_ok": int(ok),
            "num_samples_failed": int(fail),
            "weighted_by_pixels": True,
            "lsos_params": {"alpha": float(ALPHA), "eps": float(EPS)},
            "baseline": {"lsos": float(baseline_lsos)},
            "model": {"lsos": float(model_lsos)},
            "improvement": {"lsos_delta": float(baseline_lsos - model_lsos)},
            "examples_saved": int(ex_saved),
            "examples_dir": str(examples_out),
        }

        if COMPUTE_DELTAE76:
            baseline_de = sum_de_base / float(sum_px)
            model_de = sum_de_model / float(sum_px)
            summary["baseline"] = {**summary["baseline"], "deltaE76_mean": float(baseline_de)}
            summary["model"] = {**summary["model"], "deltaE76_mean": float(model_de)}
            summary["improvement"] = {**summary["improvement"], "deltaE76_mean_delta": float(baseline_de - model_de)}

        with open(ckpt_out / "results_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        csv_path = ckpt_out / "per_sample_metrics.csv"
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())

        preferred = [
            "id", "shard", "idx", "h", "w",
            "baseline_lsos", "model_lsos",
            "baseline_deltaE76_mean", "model_deltaE76_mean",
            "error",
        ]
        cols = [k for k in preferred if k in all_keys] + sorted([k for k in all_keys if k not in preferred])

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"[{ckpt_tag}] summary: { (ckpt_out / 'results_summary.json').resolve() }")
        print(f"[{ckpt_tag}] csv:     { csv_path.resolve() }")
        print(f"[{ckpt_tag}] examples:{ examples_out.resolve() }")
        print(f"[{ckpt_tag}] Baseline LSOS={baseline_lsos:.6f} | Model LSOS={model_lsos:.6f} | Δ={baseline_lsos - model_lsos:.6f}")
        if COMPUTE_DELTAE76:
            print(f"[{ckpt_tag}] Baseline ΔE={summary['baseline']['deltaE76_mean']:.4f} | "
                  f"Model ΔE={summary['model']['deltaE76_mean']:.4f} | "
                  f"Δ={summary['improvement']['deltaE76_mean_delta']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
