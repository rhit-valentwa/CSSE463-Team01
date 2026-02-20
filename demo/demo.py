from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

INPUT_DIR   = str(SCRIPT_DIR / "demo_images")
OUTPUT_DIR  = str(SCRIPT_DIR / "demo_outputs")

UNET_CHECKPOINT  = str(PROJECT_DIR / "data" / "models" / "unet_colorizer_best.pt")
VGG16_CHECKPOINT = str(PROJECT_DIR / "data" / "models" / "cnn_colorizer_best.pt")

UNET_BASE_CH = 32
PAD_MULTIPLE = 16
USE_AMP      = True
CHANNELS_LAST = True

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

        self.pool      = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.dec4 = ConvBlock(base * 16 + base * 8, base * 8)
        self.dec3 = ConvBlock(base * 8  + base * 4, base * 4)
        self.dec2 = ConvBlock(base * 4  + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2  + base,     base)

        self.out = nn.Conv2d(base, 2, 1)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._up_to(b,  e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up_to(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up_to(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up_to(d2, e1), e1], dim=1))
        return torch.tanh(self.out(d1))


class ColorizationCNN(nn.Module):
    """VGG16-based colorizer (matches CNNtest.py architecture)."""
    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        vgg      = models.vgg16_bn(pretrained=pretrained_backbone)
        features = list(vgg.features.children())

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(*features[6:13])   # ~128 ch, downsampled
        self.enc3 = nn.Sequential(*features[13:23])  # ~256 ch, downsampled
        self.enc4 = nn.Sequential(*features[23:33])  # ~512 ch, downsampled

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dec4 = nn.Conv2d(512 + 512, 512, 3, padding=1)
        self.dec3 = nn.Conv2d(512 + 256, 256, 3, padding=1)
        self.dec2 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.dec1 = nn.Conv2d(128 + 64,  64,  3, padding=1)
        self.out  = nn.Conv2d(64,        2,   1)

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


# OpenCV helpers

def _cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError as e:
        raise RuntimeError("OpenCV required: pip install opencv-python") from e


def load_rgb_u8(path: str) -> np.ndarray:
    cv2 = _cv2()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb_u8(path: str, rgb: np.ndarray) -> None:
    cv2 = _cv2()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Could not write image: {path}")


def rgb_to_L01(rgb_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 → L channel float32 in [0, 1]."""
    cv2 = _cv2()
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    return lab[..., 0].astype(np.float32) / 255.0


def lab_to_rgb_u8(L01: np.ndarray, ab_neg1_1: np.ndarray) -> np.ndarray:
    """L in [0,1] + ab in [-1,1] → RGB uint8."""
    cv2 = _cv2()
    L_u8 = np.clip(L01 * 255.0,                0, 255).astype(np.uint8)
    a_u8 = np.clip(ab_neg1_1[..., 0] * 128 + 128, 0, 255).astype(np.uint8)
    b_u8 = np.clip(ab_neg1_1[..., 1] * 128 + 128, 0, 255).astype(np.uint8)
    lab_u8 = np.stack([L_u8, a_u8, b_u8], axis=-1)
    return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)


def gray_rgb_from_L01(L01: np.ndarray) -> np.ndarray:
    g = np.clip(L01 * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def hstack_rgb(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=1)


# Checkpoint loading

def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if any(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise RuntimeError(f"Unrecognized checkpoint format: {type(obj)}")


def load_checkpoint(model: nn.Module, path: str) -> None:
    obj   = torch.load(path, map_location=DEVICE)
    state = _unwrap_state_dict(obj)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"  [warn] strict load failed ({e}); retrying with strict=False")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [warn] missing keys  : {missing[:6]}{'...' if len(missing)>6 else ''}")
        if unexpected:
            print(f"  [warn] unexpected keys: {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")


# Inference

def _autocast(enabled: bool):
    dtype = "cuda" if DEVICE.startswith("cuda") else "cpu"
    return torch.amp.autocast(dtype, enabled=enabled)


@torch.inference_mode()
def predict_ab(model: nn.Module, L01: np.ndarray) -> np.ndarray:
    """L01: HxW float32 [0,1] → ab: HxWx2 float32 [-1,1]"""
    amp = USE_AMP and DEVICE.startswith("cuda")

    x = torch.from_numpy(L01).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
    # If the model's first conv expects 3 channels, replicate the L channel
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    if first_conv.in_channels == 3:
        x = x.repeat(1, 3, 1, 1)
    if CHANNELS_LAST and DEVICE.startswith("cuda"):
        x = x.contiguous(memory_format=torch.channels_last)

    _, _, h, w = x.shape
    ph = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
    pw = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
    pad_mode = "reflect" if (h >= 2 and w >= 2) else "replicate"
    x_pad = F.pad(x, (0, pw, 0, ph), mode=pad_mode)

    with _autocast(amp):
        ab_pad = model(x_pad)

    ab = ab_pad[:, :, :h, :w].squeeze(0).permute(1, 2, 0).contiguous()
    return ab.cpu().numpy().astype(np.float32)


# Main

def main():
    print(f"Device : {DEVICE}")
    print(f"Input  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_DIR}")

    # Gather input images
    in_dir = Path(INPUT_DIR)
    if not in_dir.is_dir():
        raise SystemExit(f"ERROR: INPUT_DIR not found: {in_dir.resolve()}")

    image_paths = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_paths:
        raise SystemExit(f"ERROR: no supported images found in {in_dir.resolve()}")

    print(f"Found {len(image_paths)} image(s)\n")

    out_dir = Path(OUTPUT_DIR)
    (out_dir / "unet").mkdir(parents=True, exist_ok=True)
    (out_dir / "vgg16").mkdir(parents=True, exist_ok=True)
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)

    # Load UNet
    print(f"Loading UNet   : {UNET_CHECKPOINT}")
    unet = UNetColorizer(base=UNET_BASE_CH).to(DEVICE)
    if CHANNELS_LAST and DEVICE.startswith("cuda"):
        unet = unet.to(memory_format=torch.channels_last)
    load_checkpoint(unet, UNET_CHECKPOINT)
    unet.eval()
    print("  UNet loaded.\n")

    # Load VGG16 CNN
    print(f"Loading VGG16  : {VGG16_CHECKPOINT}")
    vgg16 = ColorizationCNN(pretrained_backbone=False).to(DEVICE)
    if CHANNELS_LAST and DEVICE.startswith("cuda"):
        vgg16 = vgg16.to(memory_format=torch.channels_last)
    load_checkpoint(vgg16, VGG16_CHECKPOINT)
    vgg16.eval()
    print("  VGG16 loaded.\n")

    # Process each image
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name}")
        try:
            rgb_in   = load_rgb_u8(str(img_path))
            L01      = rgb_to_L01(rgb_in)

            ab_unet  = predict_ab(unet,  L01)
            ab_vgg16 = predict_ab(vgg16, L01)

            rgb_unet  = lab_to_rgb_u8(L01, ab_unet)
            rgb_vgg16 = lab_to_rgb_u8(L01, ab_vgg16)
            gray      = gray_rgb_from_L01(L01)

            stem = img_path.stem

            # Individual outputs
            save_rgb_u8(str(out_dir / "unet"  / f"{stem}_unet.png"),  rgb_unet)
            save_rgb_u8(str(out_dir / "vgg16" / f"{stem}_vgg16.png"), rgb_vgg16)

            # 3-panel comparison: grayscale | unet | vgg16
            compare = hstack_rgb(gray, rgb_unet, rgb_vgg16)
            save_rgb_u8(str(out_dir / "compare" / f"{stem}_compare.png"), compare)

            print(f"  -> unet/{stem}_unet.png")
            print(f"  -> vgg16/{stem}_vgg16.png")
            print(f"  -> compare/{stem}_compare.png")

        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\nDone. Results in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()