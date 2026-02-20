from pathlib import Path
import os
import numpy as np
from skimage import io, color, img_as_float32
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_DIR = Path("train2017")
DST_DIR = Path("train2017_cache_256_mmap")
CHUNK_SIZE = 512
CROP = 256

def iter_images(src_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def random_crop_pair(L_u8, ab_i8, crop=CROP):
    h, w = L_u8.shape
    pad_h = max(0, crop - h)
    pad_w = max(0, crop - w)
    if pad_h or pad_w:
        L_u8 = np.pad(L_u8, ((0, pad_h), (0, pad_w)), mode="reflect")
        ab_i8 = np.pad(ab_i8, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w = L_u8.shape

    top = 0 if h == crop else np.random.randint(0, h - crop + 1)
    left = 0 if w == crop else np.random.randint(0, w - crop + 1)
    Lc = L_u8[top:top+crop, left:left+crop]
    abc = ab_i8[top:top+crop, left:left+crop, :]  # (crop,crop,2)
    return Lc, abc

def rgb_to_crop(path_str: str):
    rgb = io.imread(path_str)

    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    elif rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {rgb.shape}")

    lab = color.rgb2lab(img_as_float32(rgb)).astype(np.float32)
    L = lab[..., 0]
    ab = lab[..., 1:3]

    L_u8 = np.clip(np.round(L * 255.0 / 100.0), 0, 255).astype(np.uint8)
    ab_i8 = np.clip(np.round(ab), -128, 127).astype(np.int8)

    Lc, abc = random_crop_pair(L_u8, ab_i8, CROP)

    # Return in training-friendly layout
    # L: (256,256) uint8
    # ab: (2,256,256) int8
    ab_chw = np.transpose(abc, (2, 0, 1))  # (2,H,W)
    return Lc, ab_chw

def save_shard(shard_idx, ids, L_arr, ab_arr):
    # L_arr: (N,256,256) uint8
    # ab_arr: (N,2,256,256) int8
    DST_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DST_DIR / f"shard_{shard_idx:05d}_L.npy", L_arr)
    np.save(DST_DIR / f"shard_{shard_idx:05d}_ab.npy", ab_arr)
    np.save(DST_DIR / f"shard_{shard_idx:05d}_ids.npy", np.array(ids, dtype=object))

def main():
    if not SRC_DIR.is_dir():
        raise SystemExit(f'Expected "{SRC_DIR}" in current directory.')

    files = list(iter_images(SRC_DIR))
    total = len(files)
    if total == 0:
        print("No images found.")
        return

    cpu = os.cpu_count() or 8
    workers = max(1, cpu - 2)

    print(f"Found {total} images. Converting with {workers} workers.")
    print(f"Saving memmap-friendly shards: CHUNK_SIZE={CHUNK_SIZE}, CROP={CROP}, out={DST_DIR}")

    # Pre-allocate shard buffers for speed
    L_buf = np.empty((CHUNK_SIZE, CROP, CROP), dtype=np.uint8)
    ab_buf = np.empty((CHUNK_SIZE, 2, CROP, CROP), dtype=np.int8)
    ids = []

    shard_idx = 0
    k = 0
    ok = fail = 0
    done = 0
    bar_w = 40

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(rgb_to_crop, str(p)): p for p in files}

        for fut in as_completed(futures):
            p = futures[fut]
            done += 1
            try:
                Lc, ab_chw = fut.result()
                L_buf[k] = Lc
                ab_buf[k] = ab_chw
                ids.append(p.stem)
                k += 1
                ok += 1
            except Exception as e:
                fail += 1
                print(f"\n[FAIL] {p.name}: {e}")

            if k == CHUNK_SIZE:
                save_shard(shard_idx, ids, L_buf, ab_buf)
                shard_idx += 1
                k = 0
                ids = []

            filled = int(bar_w * done / total)
            bar = "#" * filled + "-" * (bar_w - filled)
            print(f"\r[{bar}] {done}/{total} ok={ok} fail={fail} shards={shard_idx}", end="", flush=True)

    # Flush remainder (only the filled part)
    if k > 0:
        save_shard(shard_idx, ids, L_buf[:k].copy(), ab_buf[:k].copy())
        shard_idx += 1

    print(f"\nDone. Success={ok} Failed={fail}. Wrote {shard_idx} shards to {DST_DIR.resolve()}")

if __name__ == "__main__":
    main()