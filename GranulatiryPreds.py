"""
Boundary Granularity Evaluation with Visual Output & Qualitative Analysis
UNet vs VGG-16 — shows how each refinement method changes color boundaries

Author: Temi Akinselure
Team: Temi Akinselure, Remi Schwartz, William Valentine, Stella Joung
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import ndimage
from skimage import segmentation, color as skcolor
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from CNNtest import ColorizationCNN

# ============================================================================
# Configuration
# ============================================================================

SHARD_DIR       = Path("data/coco/train2017_cache_256_mmap")
UNET_MODEL_PATH = "unet_colorizer_best.pt"
# VGG_MODEL_PATH  = "vgg16_colorizer_best.pt"   # TODO: update to your actual path

BATCH_SIZE      = 4
NUM_WORKERS     = 4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEED            = 42
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10
TEST_RATIO      = 0.10
MAX_EVAL_IMAGES = 500
NUM_VISUAL_EXAMPLES = 5
OUTPUT_DIR = Path("granularity_results")

REFINEMENT_METHODS = [
    ("No Refinement",  None,         {}),
    ("Superpixel 200", "superpixel", {"n_segments": 200}),
    ("Superpixel 400", "superpixel", {"n_segments": 400}),
    ("Superpixel 800", "superpixel", {"n_segments": 800}),
    ("Median Filter",  "median",     {"kernel_size": 5}),
    ("Combined",       "combined",   {"n_segments": 400, "kernel_size": 5}),
]

METHOD_COLORS = {
    "No Refinement":  "#e74c3c",
    "Superpixel 200": "#e67e22",
    "Superpixel 400": "#f1c40f",
    "Superpixel 800": "#2ecc71",
    "Median Filter":  "#3498db",
    "Combined":       "#9b59b6",
}


# ============================================================================
# Model Definitions
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
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
    def __init__(self, base=32):
        super().__init__()
        self.enc1=ConvBlock(1,base); self.enc2=ConvBlock(base,base*2)
        self.enc3=ConvBlock(base*2,base*4); self.enc4=ConvBlock(base*4,base*8)
        self.pool=nn.MaxPool2d(2)
        self.bottleneck=ConvBlock(base*8,base*16)
        self.dec4=ConvBlock(base*16+base*8,base*8); self.dec3=ConvBlock(base*8+base*4,base*4)
        self.dec2=ConvBlock(base*4+base*2,base*2);  self.dec1=ConvBlock(base*2+base,base)
        self.out=nn.Conv2d(base,2,1)
    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self._up(b,e4),e4],1))
        d3=self.dec3(torch.cat([self._up(d4,e3),e3],1))
        d2=self.dec2(torch.cat([self._up(d3,e2),e2],1))
        d1=self.dec1(torch.cat([self._up(d2,e1),e1],1))
        return torch.tanh(self.out(d1))


class ColorizationCNN(nn.Module):
    def __init__(self, pretrained_backbone=False):
        super().__init__()
        vgg=models.vgg16_bn(pretrained=pretrained_backbone)
        f=list(vgg.features.children())
        self.enc1=nn.Sequential(*f[:6]);  self.enc2=nn.Sequential(*f[6:13])
        self.enc3=nn.Sequential(*f[13:23]); self.enc4=nn.Sequential(*f[23:33])
        self.bottleneck=nn.Sequential(nn.Conv2d(512,512,3,padding=1),
                                      nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.dec4=nn.Conv2d(1024,512,3,padding=1); self.dec3=nn.Conv2d(768,256,3,padding=1)
        self.dec2=nn.Conv2d(384,128,3,padding=1);  self.dec1=nn.Conv2d(192,64,3,padding=1)
        self.out=nn.Conv2d(64,2,1)
    def forward(self, x):
        if x.shape[1]==1: x=x.repeat(1,3,1,1)
        e1=self.enc1(x); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3)
        b=self.bottleneck(F.max_pool2d(e4,2))
        up=lambda t,s: F.interpolate(t,scale_factor=s,mode="bilinear",align_corners=False)
        d4=F.relu(self.dec4(torch.cat([up(b,2),e4],1)))
        d3=F.relu(self.dec3(torch.cat([up(d4,2),e3],1)))
        d2=F.relu(self.dec2(torch.cat([up(d3,2),e2],1)))
        d1=F.relu(self.dec1(torch.cat([up(d2,2),e1],1)))
        return torch.tanh(self.out(d1))


# ============================================================================
# Dataset
# ============================================================================

class CocoMMapCropDataset(Dataset):
    def __init__(self, shard_dir):
        self.shard_dir=Path(shard_dir)
        self.L_paths=sorted(self.shard_dir.glob("shard_*_L.npy"))
        self.ab_paths=sorted(self.shard_dir.glob("shard_*_ab.npy"))
        if not self.L_paths: raise RuntimeError(f"No shards in {self.shard_dir}")
        self._index=[]
        for si,Lp in enumerate(self.L_paths):
            n=int(np.load(Lp,mmap_mode="r").shape[0])
            self._index.extend((si,li) for li in range(n))
        self._cached_si=None; self._L_mmap=self._ab_mmap=None
    def __len__(self): return len(self._index)
    def _load_shard(self,si):
        if self._cached_si!=si:
            self._L_mmap=np.load(self.L_paths[si],mmap_mode="r")
            self._ab_mmap=np.load(self.ab_paths[si],mmap_mode="r")
            self._cached_si=si
        return self._L_mmap,self._ab_mmap
    def __getitem__(self,idx):
        si,li=self._index[idx]; Lm,abm=self._load_shard(si)
        return (torch.from_numpy(Lm[li].astype(np.float32)/255.0).unsqueeze(0),
                torch.from_numpy(abm[li].astype(np.float32)/128.0))


def make_split_indices(n,seed,tr,vr,ter):
    rng=np.random.default_rng(seed); perm=rng.permutation(n)
    nt=int(n*tr); nv=int(n*vr)
    return perm[:nt].tolist(),perm[nt:nt+nv].tolist(),perm[nt+nv:].tolist()


# ============================================================================
# Colour Helpers
# ============================================================================

def lab_to_rgb(L_np, ab_np):
    """L in [0,1], ab in [-1,1] (either (2,H,W) or (H,W,2)) → RGB [0,1]"""
    if ab_np.ndim==3 and ab_np.shape[0]==2:
        ab_np=ab_np.transpose(1,2,0)
    Lab=np.stack([L_np*100.0, ab_np[:,:,0]*128.0, ab_np[:,:,1]*128.0], axis=-1)
    return np.clip(skcolor.lab2rgb(Lab),0,1)


def edge_map(L_np):
    mag=np.sqrt(ndimage.sobel(L_np,axis=1)**2+ndimage.sobel(L_np,axis=0)**2)
    return mag/(mag.max()+1e-8)


# ============================================================================
# Refinement
# ============================================================================

def superpixel_refine(L_batch, pred, n_segments):
    out=[]
    for i in range(L_batch.shape[0]):
        L_u8=(L_batch[i,0].cpu().numpy()*255).clip(0,255).astype(np.uint8)
        ab_np=pred[i].cpu().numpy().transpose(1,2,0)
        try:
            segs=segmentation.slic(L_u8,n_segments=n_segments,
                                   compactness=10,start_label=0,channel_axis=None)
            ref=np.zeros_like(ab_np)
            for sid in np.unique(segs):
                m=segs==sid
                ref[m,0]=ab_np[m,0].mean(); ref[m,1]=ab_np[m,1].mean()
            out.append(torch.from_numpy(ref.transpose(2,0,1)).float())
        except Exception:
            out.append(pred[i].cpu())
    return torch.stack(out)


def median_refine(L_batch, pred, kernel_size):
    from scipy.ndimage import median_filter
    out=[]
    for i in range(L_batch.shape[0]):
        L_np=L_batch[i,0].cpu().numpy(); ab_np=pred[i].cpu().numpy()
        is_edge=edge_map(L_np)>0.1
        ab_f=ab_np.copy()
        ab_f[0,~is_edge]=median_filter(ab_np[0],size=kernel_size)[~is_edge]
        ab_f[1,~is_edge]=median_filter(ab_np[1],size=kernel_size)[~is_edge]
        out.append(torch.from_numpy(ab_f).float())
    return torch.stack(out)


def apply_refinement(pred, L_batch, method, params):
    if method is None:     return pred
    if method=="superpixel": return superpixel_refine(L_batch,pred,params.get("n_segments",400)).to(pred.device)
    if method=="median":     return median_refine(L_batch,pred,params.get("kernel_size",5)).to(pred.device)
    if method=="combined":
        pred=superpixel_refine(L_batch,pred,params.get("n_segments",400)).to(pred.device)
        return median_refine(L_batch,pred,params.get("kernel_size",5)).to(pred.device)
    return pred


# ============================================================================
# Metrics
# ============================================================================

def detect_edges_batch(L_batch):
    masks=[]
    for i in range(L_batch.shape[0]):
        L_np=L_batch[i,0].cpu().numpy()
        mag=np.sqrt(ndimage.sobel(L_np,axis=1)**2+ndimage.sobel(L_np,axis=0)**2)
        mag=mag/(mag.max()+1e-8)
        masks.append(torch.from_numpy(mag>np.percentile(mag,80)))
    return torch.stack(masks)


@torch.no_grad()
def evaluate(model, loader, device, method=None, params=None, max_images=None):
    if params is None: params={}
    model.eval()
    ta=ts=te=aa=ab_=ec=0.0
    ea=ee=nea=nee=0.0; seen=0
    for L,ab in loader:
        if max_images and seen>=max_images: break
        L,ab=L.to(device),ab.to(device)
        pred=apply_refinement(model(L),L,method,params)
        err=pred-ab; ae=err.abs()
        ts+=(err*err).sum().item(); ta+=ae.sum().item(); te+=err.numel()
        aa+=ae[:,0].sum().item(); ab_+=ae[:,1].sum().item(); ec+=ae[:,0].numel()
        em=detect_edges_batch(L).to(device).unsqueeze(1).expand_as(ae)
        ea+=ae[em].sum().item(); ee+=em.sum().item()
        nea+=ae[~em].sum().item(); nee+=(~em).sum().item()
        seen+=L.shape[0]
        if seen%100==0: print(f"    {seen} images...")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    mae=ta/max(te,1); mse=ts/max(te,1)
    em_=ea/max(ee,1); nem_=nea/max(nee,1)
    return {"mae":mae,"mse":mse,"rmse":float(np.sqrt(mse)),
            "mae_a":aa/max(ec,1),"mae_b":ab_/max(ec,1),
            "edge_mae":em_,"non_edge_mae":nem_,"edge_penalty":em_/max(nem_,1e-8),
            "edge_pixels":int(ee//2),"non_edge_pixels":int(nee//2)}


# ============================================================================
# Visual Output
# ============================================================================

def render_example(L_np, ab_gt_np, preds_dict, img_idx, model_label, out_dir):
    """
    Saves one dark-themed figure per example image.
    Layout:
      Top row  : Input (greyscale)  |  Ground Truth
      Middle N : [RGB prediction  |  Error heatmap] for each method
      Bottom   : Edge map  |  Superpixel overlay
    """
    n_methods = len(preds_dict)
    total_rows = n_methods + 2
    BG = "#111111"

    fig = plt.figure(figsize=(12, 3.0 * total_rows), facecolor=BG)
    fig.suptitle(f"{model_label}  —  Test Example #{img_idx}",
                 color="white", fontsize=13, fontweight="bold", y=1.002)
    gs = gridspec.GridSpec(total_rows, 4, figure=fig,
                           hspace=0.5, wspace=0.05,
                           left=0.03, right=0.97, top=0.98, bottom=0.01)

    em = edge_map(L_np)
    rgb_gt = lab_to_rgb(L_np, ab_gt_np)

    # Row 0 — input + GT
    for col_span, img, title in [
        ((0, 2), L_np,   "Grayscale Input"),
        ((2, 4), rgb_gt, "Ground Truth"),
    ]:
        ax = fig.add_subplot(gs[0, col_span[0]:col_span[1]])
        kw = {"cmap":"gray","vmin":0,"vmax":1} if title=="Grayscale Input" else {}
        ax.imshow(img, **kw); ax.set_title(title, color="#cccccc", fontsize=9); ax.axis("off")
        ax.set_facecolor(BG)

    # Rows 1..N — methods
    for row_i, (label, ab_pred_np) in enumerate(preds_dict.items(), start=1):
        rgb_pred = lab_to_rgb(L_np, ab_pred_np)

        # ensure (2,H,W) for arithmetic
        if ab_pred_np.ndim==3 and ab_pred_np.shape[2]==2:
            ab_pred_np = ab_pred_np.transpose(2,0,1)
        if ab_gt_np.ndim==3 and ab_gt_np.shape[2]==2:
            ab_gt_arr = ab_gt_np.transpose(2,0,1)
        else:
            ab_gt_arr = ab_gt_np
        err_map = np.abs(ab_pred_np - ab_gt_arr).mean(axis=0)
        mae_val = err_map.mean()
        col = METHOD_COLORS.get(label, "white")

        # RGB prediction
        ax_rgb = fig.add_subplot(gs[row_i, 0:2])
        ax_rgb.imshow(rgb_pred)
        ax_rgb.set_title(f"{label}", color=col, fontsize=9, fontweight="bold")
        ax_rgb.axis("off")
        for sp in ax_rgb.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2.5); sp.set_visible(True)

        # Error heatmap
        ax_err = fig.add_subplot(gs[row_i, 2:4])
        ax_err.imshow(err_map, cmap="inferno", vmin=0, vmax=0.25)
        ax_err.set_title(f"Error  (MAE={mae_val:.4f})", color="#999999", fontsize=8)
        ax_err.axis("off")

    # Bottom row — edge map + superpixel
    ax_e = fig.add_subplot(gs[total_rows-1, 0:2])
    ax_e.imshow(em, cmap="hot"); ax_e.set_title("Sobel Edge Map", color="#777777", fontsize=8)
    ax_e.axis("off")

    L_u8 = (L_np*255).clip(0,255).astype(np.uint8)
    segs = segmentation.slic(L_u8,n_segments=400,compactness=10,channel_axis=None)
    base_rgb = lab_to_rgb(L_np, list(preds_dict.values())[0])
    ax_sp = fig.add_subplot(gs[total_rows-1, 2:4])
    ax_sp.imshow(segmentation.mark_boundaries(base_rgb, segs, color=(0.2,0.9,0.4)))
    ax_sp.set_title("Superpixel Boundaries (400)", color="#777777", fontsize=8)
    ax_sp.axis("off")

    save_path = out_dir / f"{model_label}_example_{img_idx:03d}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    Saved: {save_path.name}")


def render_summary_chart(all_results, out_dir):
    """Grouped bar chart: Edge MAE per method, one group per model"""
    BG = "#0d0d0d"
    method_labels = [m[0] for m in REFINEMENT_METHODS]
    colors = [METHOD_COLORS[l] for l in method_labels]
    n_models = len(all_results)

    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 5), facecolor=BG)
    if n_models == 1: axes = [axes]
    fig.suptitle("Edge MAE by Refinement Method", color="white", fontsize=13, fontweight="bold")

    for ax, (model_label, res) in zip(axes, all_results.items()):
        ax.set_facecolor("#1c1c2e")
        vals = [res[lbl]["edge_mae"] for lbl in method_labels]
        baseline_val = res["No Refinement"]["edge_mae"]
        bars = ax.bar(method_labels, vals, color=colors, alpha=0.88, width=0.6, edgecolor="#333")
        ax.axhline(baseline_val, color="#e74c3c", linestyle="--", linewidth=1.3,
                   label=f"No-refinement baseline: {baseline_val:.5f}")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0002,
                    f"{val:.5f}", ha="center", va="bottom", color="white", fontsize=7)
        best_i = int(np.argmin(vals))
        ax.annotate("★ Best", xy=(best_i, vals[best_i]),
                    xytext=(best_i, vals[best_i]+0.0018),
                    ha="center", color="#2ecc71", fontsize=9, fontweight="bold")
        ax.set_title(model_label, color="white", fontsize=11)
        ax.set_ylabel("Edge MAE  (lower = better)", color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xticklabels(method_labels, rotation=35, ha="right")
        for sp in ax.spines.values(): sp.set_color("#555")
        ax.legend(fontsize=8, labelcolor="white", facecolor="#222", edgecolor="#444")

    plt.tight_layout(rect=[0,0,1,0.93])
    p = out_dir/"summary_edge_mae.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n  Summary chart: {p.name}")


# ============================================================================
# Per-model run
# ============================================================================

@torch.no_grad()
def run_model(model, test_loader, model_label, out_dir):
    model.eval()
    results = {}

    # Quantitative metrics
    for label, method, params in REFINEMENT_METHODS:
        print(f"  [{model_label}] evaluating: {label}")
        results[label] = evaluate(model, test_loader, DEVICE,
                                  method=method, params=params, max_images=MAX_EVAL_IMAGES)

    # Visual examples
    print(f"\n  Generating {NUM_VISUAL_EXAMPLES} example images for {model_label}...")
    example_count = 0
    for L_batch, ab_gt_batch in test_loader:
        if example_count >= NUM_VISUAL_EXAMPLES: break
        L_batch   = L_batch.to(DEVICE)
        ab_gt_batch = ab_gt_batch.to(DEVICE)
        raw_pred  = model(L_batch)

        preds_dict = {}
        for label, method, params in REFINEMENT_METHODS:
            ref = apply_refinement(raw_pred.clone(), L_batch, method, params)
            preds_dict[label] = ref[0].cpu().numpy()   # (2,H,W)

        render_example(
            L_np    = L_batch[0,0].cpu().numpy(),
            ab_gt_np = ab_gt_batch[0].cpu().numpy(),
            preds_dict = preds_dict,
            img_idx  = example_count,
            model_label = model_label,
            out_dir  = out_dir,
        )
        example_count += 1
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results


# ============================================================================
# Qualitative Analysis
# ============================================================================

def print_qualitative_analysis(all_results):
    print("\n" + "="*80)
    print("QUALITATIVE ANALYSIS")
    print("="*80)

    for model_label, res in all_results.items():
        base = res["No Refinement"]
        ranked = sorted(res.items(), key=lambda x: x[1]["edge_mae"])
        best_label, best_m = ranked[0]
        ep = base["edge_penalty"]
        improvement = (base["edge_mae"] - best_m["edge_mae"]) / base["edge_mae"] * 100

        print(f"\n── {model_label} ──────────────────────────────────────────────────")
        print(f"  Baseline MAE       : {base['mae']:.6f}")
        print(f"  Baseline Edge MAE  : {base['edge_mae']:.6f}")
        print(f"  Baseline EdgePenalty: {ep:.4f}")
        print(f"  Best method        : {best_label}  (Edge MAE improvement: {improvement:+.2f}%)")
        print()

        # Interpret edge penalty
        if ep < 1.05:
            print(f"  → Edge Penalty of {ep:.4f} is excellent. The model already preserves")
            print(f"    color boundaries well — skip connections are doing their job.")
            print(f"    Post-processing offers minimal room for improvement.")
        elif ep < 1.15:
            print(f"  → Edge Penalty of {ep:.4f} indicates mild boundary errors.")
            print(f"    Post-processing refinement can provide moderate improvement.")
        else:
            print(f"  → Edge Penalty of {ep:.4f} shows notable color bleeding at edges.")
            print(f"    Post-processing refinement is strongly recommended.")
        print()

        # Per-method breakdown
        print(f"  {'Method':<20} {'Edge MAE':>10} {'Δ Edge MAE':>12}  Verdict")
        print(f"  {'─'*65}")
        for label, _, _ in REFINEMENT_METHODS:
            m = res[label]
            d = (base["edge_mae"] - m["edge_mae"]) / base["edge_mae"] * 100
            if label == "No Refinement":
                verdict = "(baseline)"
            elif d > 2.0:
                verdict = "✓ Clear improvement"
            elif d > 0.5:
                verdict = "~ Marginal improvement"
            elif d > -0.5:
                verdict = "≈ No meaningful change"
            else:
                verdict = "✗ Slight degradation — over-smoothing"
            print(f"  {label:<20} {m['edge_mae']:>10.6f} {d:>+11.2f}%  {verdict}")

    # Cross-model comparison
    if "UNet" in all_results and "VGG16" in all_results:
        u_em = all_results["UNet"]["No Refinement"]["edge_mae"]
        v_em = all_results["VGG16"]["No Refinement"]["edge_mae"]
        print("\n" + "─"*80)
        print("  CROSS-MODEL COMPARISON (no refinement)")
        print(f"  U-Net  Edge MAE : {u_em:.6f}")
        print(f"  VGG-16 Edge MAE : {v_em:.6f}")
        if v_em < u_em:
            print(f"  → VGG-16 is {(u_em-v_em)/u_em*100:.2f}% better at boundaries.")
            print("    Semantic features from the pretrained backbone help the model")
            print("    understand where object boundaries are, reducing color bleeding.")
        else:
            print(f"  → U-Net is {(v_em-u_em)/v_em*100:.2f}% better at boundaries.")
            print("    Dense skip connections preserve fine spatial detail better than")
            print("    VGG-16's deeper but spatially coarser features.")

    print()
    print("  KEY TAKEAWAY FOR YOUR REPORT")
    print("  ─────────────────────────────────────────────────────────────────────")
    print("  The granularity analysis shows that boundary quality is primarily")
    print("  determined by the model architecture (skip connections in U-Net,")
    print("  semantic pretraining in VGG-16), not by post-processing. This means")
    print("  the best path to sharper color boundaries is architectural: either")
    print("  denser skip connections, higher base_ch, or better pretrained features.")


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*80)
    print("BOUNDARY GRANULARITY EVALUATION WITH VISUAL OUTPUT")
    print("="*80)
    print(f"Device: {DEVICE}  |  Max eval images: {MAX_EVAL_IMAGES}")
    print(f"Visual examples per model: {NUM_VISUAL_EXAMPLES}")
    print(f"Outputs: {OUTPUT_DIR.resolve()}\n")

    ds = CocoMMapCropDataset(SHARD_DIR)
    _, _, test_idx = make_split_indices(len(ds), SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, drop_last=False)
    print(f"Test set: {len(test_idx)} images\n")

    all_results = {}

    # # U-Net
    # print("="*80); print("U-NET"); print("="*80)
    # unet = UNetColorizer(base=32)
    # checkpoint = torch.load(UNET_MODEL_PATH, map_location=DEVICE)
    # unet.load_state_dict(checkpoint["model"])
    # unet = unet.to(DEVICE); print(f"✓ Loaded {UNET_MODEL_PATH}\n")
    # all_results["UNet"] = run_model(unet, test_loader, "UNet", OUTPUT_DIR)
    # del unet
    # if torch.cuda.is_available(): torch.cuda.empty_cache()

    # VGG-16
    print("\n" + "="*80); print("VGG-16"); print("="*80)
    try:
        vgg = ColorizationCNN(pretrained_backbone=True)
        # vgg.load_state_dict(torch.load(VGG_MODEL_PATH, map_location=DEVICE))
        vgg = vgg.to(DEVICE)
        all_results["VGG16"] = run_model(vgg, test_loader, "VGG16", OUTPUT_DIR)
        del vgg
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except FileNotFoundError:
        # print(f"⚠  {VGG_MODEL_PATH} not found — skipping VGG-16.\n")
        print("   Update VGG_MODEL_PATH at the top of the script.")

    # Summary chart
    render_summary_chart(all_results, OUTPUT_DIR)

    # Printed summary table
    print("\n" + "="*80); print("SUMMARY TABLE"); print("="*80)
    for model_label, res in all_results.items():
        base = res["No Refinement"]
        print(f"\n  {model_label}")
        print(f"  {'Method':<20} {'MAE':>10} {'Edge MAE':>11} {'EdgePenalty':>13} {'Δ Edge MAE':>12}")
        print("  " + "─"*70)
        for label, _, _ in REFINEMENT_METHODS:
            m = res[label]
            d = (base["edge_mae"] - m["edge_mae"]) / base["edge_mae"] * 100
            tag = "  ← baseline" if label=="No Refinement" else ""
            print(f"  {label:<20} {m['mae']:>10.6f} {m['edge_mae']:>11.6f} "
                  f"{m['edge_penalty']:>13.4f} {d:>+11.2f}%{tag}")

    print_qualitative_analysis(all_results)
    print("\n" + "="*80)
    print(f"DONE. All outputs in: {OUTPUT_DIR.resolve()}")
    print("="*80)


if __name__ == "__main__":
    main()