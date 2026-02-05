# generate_3d_surface_gifs.py
# Creates slowly-spinning 3D surface GIFs from grid_results_stage1.csv
# x = lr, y = base_ch, z = val_mse, split by loss_name x scheduler
#
# Requirements:
#   pip install pandas numpy matplotlib pillow
#
# Usage:
#   python generate_3d_surface_gifs.py \
#       --csv /path/to/grid_results_stage1.csv \
#       --out_dir ./heatmap_gifs \
#       --frames 72 \
#       --interval_ms 80 \
#       --dpi 140 \
#       --top_k 4
#
# Notes:
# - Uses log10(lr) for better spacing if lr spans orders of magnitude.
# - Builds a smooth-looking surface by interpolating onto a dense grid
#   using Triangulation + LinearTriInterpolator.

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from PIL import Image


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "", s)
    return s[:180]


def find_col(df: pd.DataFrame, candidates):
    """Return first matching column name (case-insensitive) from candidates."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def prepare_group_df(df: pd.DataFrame, lr_col, ch_col, z_col):
    g = df[[lr_col, ch_col, z_col]].copy()
    g = g.dropna()
    # Ensure numeric
    for c in [lr_col, ch_col, z_col]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna()
    # Remove non-positive lr for log transform
    g = g[g[lr_col] > 0]
    return g


def build_interpolated_grid(x, y, z, grid_n=120):
    """
    Interpolate scattered points (x,y,z) onto a dense regular grid
    via triangulation + linear interpolation for a smooth-ish surface.
    """
    triang = mtri.Triangulation(x, y)
    interp = mtri.LinearTriInterpolator(triang, z)

    xi = np.linspace(np.nanmin(x), np.nanmax(x), grid_n)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), grid_n)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = interp(Xi, Yi)

    # Mask outside convex hull
    Zi = np.ma.masked_invalid(Zi)
    return Xi, Yi, Zi


def render_frames(
    out_gif_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    z_label: str,
    Xi, Yi, Zi,
    cmap_name: str = "viridis",
    frames: int = 72,
    interval_ms: int = 80,
    dpi: int = 140,
    elev: float = 28.0,
    start_azim: float = 35.0,
    rotation_deg: float = 180.0,
    zlim=None,
):
    """
    Render a rotating 3D surface to GIF using matplotlib frame rendering + PIL.
    rotation_deg=180 means half-spin; use 360 for full.
    interval_ms controls playback speed; larger interval -> slower.
    """
    cmap = cm.get_cmap(cmap_name)

    # Determine color normalization and z-limits
    z_min = float(np.nanmin(Zi))
    z_max = float(np.nanmax(Zi))
    if zlim is None:
        zlim = (z_min, z_max)
    norm = Normalize(vmin=zlim[0], vmax=zlim[1])

    tmp_frames = []
    for i in range(frames):
        fig = plt.figure(figsize=(7.2, 5.8), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        azim = start_azim + (rotation_deg * i / max(frames - 1, 1))
        ax.view_init(elev=elev, azim=azim)

        # Surface
        surf = ax.plot_surface(
            Xi, Yi, Zi,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True,
            cmap=cmap,
            norm=norm,
        )

        # Add colorbar (smoother gradient feel)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.08)
        cbar.set_label(z_label)

        ax.set_title(title, pad=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        ax.set_zlim(*zlim)

        # Clean-ish panes
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)

        # Tight layout
        plt.tight_layout()

        # Render to image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)

        tmp_frames.append(Image.fromarray(img))

    # Save GIF
    out_gif_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_frames[0].save(
        out_gif_path,
        save_all=True,
        append_images=tmp_frames[1:],
        duration=interval_ms,  # ms per frame
        loop=0,
        optimize=False,
        disposal=2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV (e.g. grid_results_stage1.csv)")
    ap.add_argument("--out_dir", default="./heatmap_gifs", help="Output folder for GIFs")
    ap.add_argument("--frames", type=int, default=72, help="Frames per GIF")
    ap.add_argument("--interval_ms", type=int, default=80, help="ms per frame; larger = slower")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--grid_n", type=int, default=140, help="Interpolation grid density")
    ap.add_argument("--rotation_deg", type=float, default=180.0, help="Total rotation in degrees")
    ap.add_argument("--top_k", type=int, default=4, help="How many (loss,sched) combos to export")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Column detection (matches what I used)
    lr_col = find_col(df, ["lr", "learning_rate"])
    ch_col = find_col(df, ["base_ch", "base_channels", "base_chans"])
    z_col = find_col(df, ["val_mse", "valid_mse", "val_loss", "validation_mse"])

    loss_col = find_col(df, ["loss", "loss_fn", "loss_name"])
    sched_col = find_col(df, ["scheduler", "sched", "lr_scheduler"])

    missing = [("lr", lr_col), ("base_ch", ch_col), ("val_mse", z_col), ("loss", loss_col), ("scheduler", sched_col)]
    missing = [k for k, v in missing if v is None]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Columns found: {list(df.columns)}")

    # Log-transform lr for spacing
    df = df.copy()
    df["_log10_lr"] = np.log10(pd.to_numeric(df[lr_col], errors="coerce"))
    lr_plot_col = "_log10_lr"

    # Pick the most data-rich combos (what I used)
    counts = (
        df.dropna(subset=[loss_col, sched_col, lr_plot_col, ch_col, z_col])
          .groupby([loss_col, sched_col])
          .size()
          .sort_values(ascending=False)
    )
    combos = list(counts.head(args.top_k).index)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for loss_name, sched_name in combos:
        g = df[(df[loss_col] == loss_name) & (df[sched_col] == sched_name)]
        g = prepare_group_df(g, lr_plot_col, ch_col, z_col)
        if len(g) < 6:
            # Need enough points to triangulate reliably
            continue

        x = g[lr_plot_col].to_numpy(dtype=float)
        y = g[ch_col].to_numpy(dtype=float)
        z = g[z_col].to_numpy(dtype=float)

        Xi, Yi, Zi = build_interpolated_grid(x, y, z, grid_n=args.grid_n)

        title = f"val_mse surface | loss={loss_name} | sched={sched_name}"
        out_name = f"surface_loss-{sanitize_filename(loss_name)}_sched-{sanitize_filename(sched_name)}.gif"
        out_path = out_dir / out_name

        render_frames(
            out_gif_path=out_path,
            title=title,
            x_label="log10(lr)",
            y_label=ch_col,
            z_label=z_col,
            Xi=Xi, Yi=Yi, Zi=Zi,
            cmap_name=args.cmap,
            frames=args.frames,
            interval_ms=args.interval_ms,
            dpi=args.dpi,
            elev=28.0,
            start_azim=35.0,
            rotation_deg=args.rotation_deg,
            zlim=None,
        )

        print(f"Wrote: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()