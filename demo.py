# demo_colorize_rgb_only_noargs.py
#
# Minimal: no CLI args.
# Edit the CONFIG section only.
#
# Takes a grayscale (but RGB) image and outputs a colorized RGB image.

from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


CHECKPOINT_PATHS = [
        "unet_colorizer_best.pt",
        "unet_colorizer.pt",
        ]


#CHECKPOINT_PATH = "unet_colorizer_best.pt"
BASE_CH = 32                                   # must match training BASE_CH

INPUT_IMAGES = [
        "demo_image.png",
        "person_demo.jpg",
        ]
OUTPUT_DIR = "qualitative_outputs"

#INPUT_IMAGE = "demo_image.png"                    # RGB grayscale image path
#OUTPUT_IMAGE = "demo_image_colorized.png"              # output RGB image path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True                              # only used if DEVICE is cuda
PAD_MULTIPLE = 16                           # keep 16 unless you changed the net

# Model (must match training)
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
    def __init__(self, base=32):
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


def _import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("This demo requires OpenCV. Install with: pip install opencv-python") from e
    return cv2


def load_rgb_u8(path: str) -> np.ndarray:
    cv2 = _import_cv2()
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def save_rgb_u8(path: str, rgb: np.ndarray) -> None:
    cv2 = _import_cv2()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def rgb_to_L01(rgb_u8: np.ndarray) -> np.ndarray:
    """
    RGB uint8 -> L in [0,1] float32 using OpenCV Lab conversion.
    """
    cv2 = _import_cv2()
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)  # uint8
    return lab[..., 0].astype(np.float32) / 255.0


def lab_to_rgb_u8(L01: np.ndarray, ab01: np.ndarray) -> np.ndarray:
    """
    L in [0,1], ab in [-1,1] -> RGB uint8 using OpenCV.
    """
    cv2 = _import_cv2()
    L_u8 = np.clip(L01 * 255.0, 0, 255).astype(np.uint8)
    a_u8 = np.clip(ab01[..., 0] * 128.0 + 128.0, 0, 255).astype(np.uint8)
    b_u8 = np.clip(ab01[..., 1] * 128.0 + 128.0, 0, 255).astype(np.uint8)
    lab_u8 = np.stack([L_u8, a_u8, b_u8], axis=-1)
    return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)

def gray_rgb_from_L01(L01: np.ndarray) -> np.ndarray:
    """
    Render L01 as an RGB grayscale image (uint8) for side-by-side comparisons.
    """
    g = np.clip(L01 * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)

def hstack_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] != b.shape[0]:
        raise ValueError("Images must have same height to hstack.")
    return np.concatenate([a, b], axis=1)

@torch.no_grad()
def predict_ab(model: nn.Module, L01: np.ndarray) -> np.ndarray:
    model.eval()
    amp_enabled = USE_AMP and DEVICE.startswith("cuda")

    x = torch.from_numpy(L01).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
    _, _, h, w = x.shape

    pad_h = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
    pad_w = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        ab_pad = model(x_pad)

    ab = ab_pad[:, :, :h, :w].squeeze(0).permute(1, 2, 0).contiguous()
    return ab.cpu().numpy().astype(np.float32)


def main():
    print("Device:", DEVICE)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for ckpt_path in CHECKPOINT_PATHS:
        print("Loading checkpoint:", ckpt_path)

        model = UNetColorizer(base=BASE_CH).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        model.eval()

        ckpt_tag = Path(ckpt_path).stem  # e.g., "unet_colorizer_best"

        for in_path in INPUT_IMAGES:
            rgb_in = load_rgb_u8(in_path)          # RGB uint8
            L01 = rgb_to_L01(rgb_in)               # HxW float32 [0,1]
            ab01 = predict_ab(model, L01)          # HxWx2 float32 [-1,1]
            rgb_out = lab_to_rgb_u8(L01, ab01)     # RGB uint8

            # Save colorized output
            out_name = f"{ckpt_tag}__{Path(in_path).stem}__colorized.png"
            out_path = Path(OUTPUT_DIR) / out_name
            save_rgb_u8(str(out_path), rgb_out)
            print("Saved colorized:", out_path)

            # Save side-by-side grayscale vs colorized
            gray_rgb = gray_rgb_from_L01(L01)
            compare = hstack_rgb(gray_rgb, rgb_out)
            cmp_name = f"{ckpt_tag}__{Path(in_path).stem}__compare.png"
            cmp_path = Path(OUTPUT_DIR) / cmp_name
            save_rgb_u8(str(cmp_path), compare)
            print("Saved compare:", cmp_path)



if __name__ == "__main__":
    main()
