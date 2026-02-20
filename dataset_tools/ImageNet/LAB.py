from pathlib import Path
import os
import random
import shutil
import numpy as np
from skimage import io, color, img_as_float32
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------
# CONFIG
# -----------------------
RAW_SRC_DIR = Path(".")                  # where nested folders live: class/000202.jpg
FLAT_DIR = Path("./flat")                # output flat folder: class_000202.jpg
SPLIT_DIR = Path("./splits")             # output splits: splits/train, splits/val, splits/test

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

# If you have multiple levels (e.g., train/class/img.jpg), you can include more path in prefix:
# prefix_mode = "parent"  -> class/img.jpg -> class_img.jpg
# prefix_mode = "twolevel"-> split/class/img.jpg -> split_class_img.jpg
PREFIX_MODE = "parent"

DO_COPY = True          # True = copy into FLAT_DIR, False = move into FLAT_DIR
SKIP_IF_EXISTS = True   # when flattening, skip if destination filename exists

# Sharding / cache settings
CHUNK_SIZE = 512
CROP = 256
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Output cache dirs (one per split)
CACHE_TRAIN = Path("train_cache_256_mmap")
CACHE_VAL   = Path("val_cache_256_mmap")
CACHE_TEST  = Path("test_cache_256_mmap")

# -----------------------
# FLATTEN
# -----------------------
def iter_images_recursive(src_dir: Path):
    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p

def make_flat_name(p: Path) -> str:
    if PREFIX_MODE == "parent":
        folder = p.parent.name
        return f"{folder}_{p.name}"
    elif PREFIX_MODE == "twolevel":
        # Use last two directory names: grandparent_parent_filename
        gp = p.parent.parent.name
        parent = p.parent.name
        return f"{gp}_{parent}_{p.name}"
    else:
        raise ValueError("PREFIX_MODE must be 'parent' or 'twolevel'")

def flatten_dataset(raw_src: Path, flat_dir: Path):
    flat_dir.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_done = 0
    n_skip = 0

    for p in iter_images_recursive(raw_src):
        n_total += 1
        out_name = make_flat_name(p)
        out_path = flat_dir / out_name

        if SKIP_IF_EXISTS and out_path.exists():
            n_skip += 1
            continue

        if DO_COPY:
            shutil.copy2(p, out_path)
        else:
            shutil.move(str(p), str(out_path))
        n_done += 1

    print(f"[FLATTEN] found={n_total} written={n_done} skipped={n_skip} -> {flat_dir.resolve()}")

# -----------------------
# SPLIT (stratified by class prefix before first "_")
# -----------------------
def class_from_flat_name(name: str) -> str:
    # class is prefix before first underscore
    return name.split("_", 1)[0]

def split_flat_dataset(flat_dir: Path, split_dir: Path):
    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

    split_train = split_dir / "train"
    split_val   = split_dir / "val"
    split_test  = split_dir / "test"
    split_train.mkdir(parents=True, exist_ok=True)
    split_val.mkdir(parents=True, exist_ok=True)
    split_test.mkdir(parents=True, exist_ok=True)

    files = [p for p in flat_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit(f"No images found in {flat_dir}")

    # group by class for stratified split
    by_class = {}
    for p in files:
        c = class_from_flat_name(p.name)
        by_class.setdefault(c, []).append(p)

    rng = random.Random(SEED)

    counts = {"train": 0, "val": 0, "test": 0}
    for c, ps in by_class.items():
        rng.shuffle(ps)
        n = len(ps)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        n_test = n - n_train - n_val  # remainder

        train_ps = ps[:n_train]
        val_ps = ps[n_train:n_train + n_val]
        test_ps = ps[n_train + n_val:]

        for p in train_ps:
            shutil.copy2(p, split_train / p.name)
            counts["train"] += 1
        for p in val_ps:
            shutil.copy2(p, split_val / p.name)
            counts["val"] += 1
        for p in test_ps:
            shutil.copy2(p, split_test / p.name)
            counts["test"] += 1

    print(f"[SPLIT] train={counts['train']} val={counts['val']} test={counts['test']} classes={len(by_class)}")
    print(f"[SPLIT] -> {split_dir.resolve()}")
    return split_train, split_val, split_test

# -----------------------
# CACHE / SHARD PIPELINE (your code, parameterized)
# -----------------------
def iter_images_flat(src_dir: Path):
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in EXTS:
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
    abc = ab_i8[top:top+crop, left:left+crop, :]
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
    ab_chw = np.transpose(abc, (2, 0, 1))  # (2,H,W)
    return Lc, ab_chw

def save_shard(dst_dir: Path, shard_idx, ids, L_arr, ab_arr):
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst_dir / f"shard_{shard_idx:05d}_L.npy", L_arr)
    np.save(dst_dir / f"shard_{shard_idx:05d}_ab.npy", ab_arr)
    np.save(dst_dir / f"shard_{shard_idx:05d}_ids.npy", np.array(ids, dtype=object))

def build_cache(src_dir: Path, dst_dir: Path):
    files = list(iter_images_flat(src_dir))
    total = len(files)
    if total == 0:
        print(f"[CACHE] No images in {src_dir}")
        return

    cpu = os.cpu_count() or 8
    workers = max(1, cpu - 2)

    print(f"[CACHE] {src_dir.name}: Found {total} images. Converting with {workers} workers.")
    print(f"[CACHE] Saving shards: CHUNK_SIZE={CHUNK_SIZE}, CROP={CROP}, out={dst_dir.resolve()}")

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
                save_shard(dst_dir, shard_idx, ids, L_buf, ab_buf)
                shard_idx += 1
                k = 0
                ids = []

            filled = int(bar_w * done / total)
            bar = "#" * filled + "-" * (bar_w - filled)
            print(f"\r[{bar}] {done}/{total} ok={ok} fail={fail} shards={shard_idx}", end="", flush=True)

    if k > 0:
        save_shard(dst_dir, shard_idx, ids, L_buf[:k].copy(), ab_buf[:k].copy())
        shard_idx += 1

    print(f"\n[CACHE] {src_dir.name}: Done. Success={ok} Failed={fail}. Wrote {shard_idx} shards.\n")

# -----------------------
# MAIN
# -----------------------
def main():
    # 1) Flatten
    flatten_dataset(RAW_SRC_DIR, FLAT_DIR)

    # 2) Split 80/10/10
    train_dir, val_dir, test_dir = split_flat_dataset(FLAT_DIR, SPLIT_DIR)

    # 3) Build caches per split
    build_cache(train_dir, CACHE_TRAIN)
    build_cache(val_dir, CACHE_VAL)
    build_cache(test_dir, CACHE_TEST)

if __name__ == "__main__":
    main()
