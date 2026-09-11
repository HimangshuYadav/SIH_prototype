"""
train_esrgan.py — SIH Sentinel-2 ESRGAN Training & Refinement Script
=====================================================================
Refined 4-band ESRGAN super-resolution model for Sentinel-2 satellite data.

Refinements:
  1. Multi-tile training across all cached Sentinel-2 scenes (urban, agri, water, desert).
  2. Pure float32 bicubic downsampling (Wald Protocol) preserving 32-bit spectral reflectance.
  3. Combined Multi-Task Loss:
     - L1 pixel loss (base reconstruction)
     - EdgeLoss (Sobel gradient L1 for sharp parcel/building boundaries)
     - SAMLoss (Spectral Angle Mapper for precise NDVI, CIR, and NDWI band ratios)
     - Perceptual Loss (VGG19 feature consistency)
     - Adversarial GAN Loss (PatchDiscriminator for natural textures)
  4. Fine-tuning capability from existing best checkpoint.

Usage:
  python3 train_esrgan.py                     # refine model for 20 epochs
  python3 train_esrgan.py --epochs 30         # full refinement
  python3 train_esrgan.py --quick             # quick 10-epoch demo
"""

import argparse
import csv
import logging
import random
import ssl
import time
from pathlib import Path

# Allow torchvision to access pretrained weights on macOS without SSL cert errors
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import rasterio

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("esrgan-train")

# Directories
CACHE_DIR  = Path("data/cache")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Hyperparameters
HR_SIZE      = 128
LR_SIZE      = HR_SIZE // 4   # = 32
N_BANDS      = 4
N_RRDB       = 8
FEAT_CH      = 64
L1_WEIGHT    = 1.0
EDGE_WEIGHT  = 0.50   # Boosted Sobel gradient loss for crisp building footprints and road corridors
SAM_WEIGHT   = 0.10
COLOR_WEIGHT = 0.40
PERC_WEIGHT  = 0.12   # Boosted perceptual loss for sharp high-frequency texture synthesis
GAN_WEIGHT   = 5e-3   # Relativistic GAN discriminator weight for micro-detail generation


# ============================================================
#  MODEL DEFINITIONS
# ============================================================

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_ch=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth_ch, 3, 1, 1, bias=True)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.d1 = DenseLayer(nf,        gc)
        self.d2 = DenseLayer(nf + gc,   gc)
        self.d3 = DenseLayer(nf + 2*gc, gc)
        self.d4 = DenseLayer(nf + 3*gc, gc)
        self.d5 = nn.Conv2d(nf + 4*gc,  nf, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(torch.cat([x,  x1], dim=1))
        x3 = self.d3(torch.cat([x,  x1, x2], dim=1))
        x4 = self.d4(torch.cat([x,  x1, x2, x3], dim=1))
        x5 = self.d5(torch.cat([x,  x1, x2, x3, x4], dim=1))
        return x5 * self.res_scale + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class ESRGANGenerator(nn.Module):
    """
    4-band ESRGAN Generator.
    Input : [B, 4, 32, 32]   (LR Sentinel-2)
    Output: [B, 4, 128, 128] (SR Sentinel-2)
    """
    def __init__(self, in_ch=N_BANDS, out_ch=N_BANDS, nf=FEAT_CH, n_rrdb=N_RRDB):
        super().__init__()
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf) for _ in range(n_rrdb)])
        self.body_tail = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.up1 = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        feat = self.head(x)
        body_out = self.body_tail(self.body(feat)) + feat
        out = self.up2(self.up1(body_out))
        return self.tail(out)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=N_BANDS):
        super().__init__()
        def block(ic, oc, stride, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, 64,  2, norm=False),
            *block(64,   128,  2),
            *block(128,  256,  2),
            *block(256,  512,  1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
#  DATASET
# ============================================================

class SentinelPatchDataset(Dataset):
    def __init__(self, cache_dir, patch_size=HR_SIZE,
                 min_tile_size=64, augment=True, max_patches_per_tile=150):
        self.patch_size = patch_size
        self.lr_size    = patch_size // 4
        self.augment    = augment
        self.patches    = []

        tif_files = sorted(Path(cache_dir).glob("*.tif"))
        log.info(f"Scanning {len(tif_files)} tiles for patches ...")

        for tif in tif_files:
            try:
                with rasterio.open(tif) as src:
                    arr = src.read().astype(np.float32)
            except Exception:
                continue

            C, H, W = arr.shape
            if H < min_tile_size or W < min_tile_size:
                continue

            # Pad small tiles up to patch_size
            if H < patch_size or W < patch_size:
                ph = max(0, patch_size - H)
                pw = max(0, patch_size - W)
                arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="edge")
                H, W = arr.shape[1], arr.shape[2]

            step = patch_size // 2
            n_rows = max(1, (H - patch_size) // step + 1)
            n_cols = max(1, (W - patch_size) // step + 1)
            positions = [(r * step, c * step)
                         for r in range(n_rows)
                         for c in range(n_cols)]
            random.shuffle(positions)
            positions = positions[:max_patches_per_tile]

            for r, c in positions:
                self.patches.append((arr, r, c))

        log.info(f"Total training patches: {len(self.patches)} across {len(tif_files)} tiles")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        arr, r, c = self.patches[idx]
        ps = self.patch_size

        hr = arr[:, r:r+ps, c:c+ps].copy()
        hr = np.clip(hr, 0.0, 1.0)

        if self.augment:
            if random.random() > 0.5:
                hr = hr[:, :, ::-1].copy()
            if random.random() > 0.5:
                hr = hr[:, ::-1, :].copy()
            k = random.randint(0, 3)
            if k > 0:
                hr = np.rot90(hr, k, axes=(1, 2)).copy()

        hr_t = torch.tensor(hr, dtype=torch.float32)

        # True float32 anti-aliased bicubic downsampling (preserving 32-bit spectral reflectance)
        lr_t = F.interpolate(
            hr_t.unsqueeze(0),
            size=(self.lr_size, self.lr_size),
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze(0).clamp(0.0, 1.0)

        return lr_t, hr_t


# ============================================================
#  SPECIALIZED LOSSES FOR REMOTE SENSING
# ============================================================

class EdgeLoss(nn.Module):
    """Sobel gradient loss to sharpen building edges and road networks."""
    def __init__(self, device):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=device).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32, device=device).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, sr, hr):
        gx_sr = F.conv2d(sr, self.kx, padding=1, groups=4)
        gy_sr = F.conv2d(sr, self.ky, padding=1, groups=4)
        gx_hr = F.conv2d(hr, self.kx, padding=1, groups=4)
        gy_hr = F.conv2d(hr, self.ky, padding=1, groups=4)
        return F.l1_loss(gx_sr, gx_hr) + F.l1_loss(gy_sr, gy_hr)


class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss to ensure multi-spectral band ratios (NDVI/NDWI) are preserved."""
    def forward(self, sr, hr):
        cos_sim = F.cosine_similarity(sr, hr, dim=1, eps=1e-6)
        return (1.0 - cos_sim).mean()


class ColorLoss(nn.Module):
    """
    Low-frequency color & spectral consistency loss.
    Enforces that macro-level color distribution and channel means match exactly,
    preventing chromatic drift, blue/green skew, and radiometric bias.
    """
    def __init__(self, device, kernel_size=15, sigma=2.5):
        super().__init__()
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
        grid = coords.repeat(kernel_size, 1)
        gauss = torch.exp(-(grid**2 + grid.t()**2) / (2 * sigma**2))
        kernel = (gauss / gauss.sum()).view(1, 1, kernel_size, kernel_size).repeat(4, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.pad = kernel_size // 2

    def forward(self, sr, hr):
        sr_blur = F.conv2d(sr, self.kernel, padding=self.pad, groups=4)
        hr_blur = F.conv2d(hr, self.kernel, padding=self.pad, groups=4)
        l1_color = F.l1_loss(sr_blur, hr_blur)
        mean_sr = sr.mean(dim=(2, 3))
        mean_hr = hr.mean(dim=(2, 3))
        l1_mean = F.l1_loss(mean_sr, mean_hr)
        return l1_color + 2.0 * l1_mean


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features)[:18]).to(device)
        for p in self.features.parameters():
            p.requires_grad = False
        self.device = device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def _to_rgb(self, x):
        return torch.stack([x[:,2], x[:,1], x[:,0]], dim=1)  # B04,B03,B02

    def forward(self, sr, hr):
        sr_n = (torch.clamp(self._to_rgb(sr), 0, 1) - self.mean) / self.std
        hr_n = (torch.clamp(self._to_rgb(hr), 0, 1) - self.mean) / self.std
        return F.mse_loss(self.features(sr_n), self.features(hr_n))


# ============================================================
#  METRICS
# ============================================================

def compute_psnr_batch(sr, hr):
    mse = F.mse_loss(sr.clamp(0,1), hr.clamp(0,1)).item()
    return 100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)

def compute_ssim_simple(sr, hr):
    C1, C2 = 0.01**2, 0.03**2
    mu_sr = sr.mean(dim=[2,3], keepdim=True)
    mu_hr = hr.mean(dim=[2,3], keepdim=True)
    sig_sr   = ((sr - mu_sr)**2).mean(dim=[2,3], keepdim=True)
    sig_hr   = ((hr - mu_hr)**2).mean(dim=[2,3], keepdim=True)
    sig_cross= ((sr - mu_sr)*(hr - mu_hr)).mean(dim=[2,3], keepdim=True)
    ssim = ((2*mu_sr*mu_hr + C1)*(2*sig_cross + C2)) / \
           ((mu_sr**2 + mu_hr**2 + C1)*(sig_sr + sig_hr + C2))
    return float(ssim.mean().item())


# ============================================================
#  TRAINING LOOP
# ============================================================

def train(args):
    device_str = "mps"  if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log.info(f"Training on: {device}")

    max_patches = 80 if args.quick else 150
    dataset = SentinelPatchDataset(CACHE_DIR, max_patches_per_tile=max_patches)

    if len(dataset) == 0:
        log.error("No training patches! Check data/cache/ has .tif files.")
        return

    n_val   = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    log.info(f"Train: {n_train}  |  Val: {n_val}")

    G = ESRGANGenerator().to(device).float()
    D = PatchDiscriminator().to(device).float()
    log.info(f"Generator: {sum(p.numel() for p in G.parameters())/1e6:.2f}M params")

    # Fine-tuning: check for existing checkpoint
    best_psnr = 0.0
    resume_path = MODELS_DIR / "esrgan_sentinel2_best.pth"
    if args.resume and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            G.load_state_dict(ckpt["generator_state"])
            if "discriminator_state" in ckpt:
                try:
                    D.load_state_dict(ckpt["discriminator_state"])
                except Exception:
                    pass
            best_psnr = float(ckpt.get("val_psnr", 29.0))
            log.info(f"Refining from existing checkpoint {resume_path.name} (initial PSNR={best_psnr:.2f}dB) ✓")
        except Exception as e:
            log.warning(f"Could not load checkpoint ({e}) — starting fresh")

    criterion_pixel = nn.L1Loss()
    criterion_edge  = EdgeLoss(device)
    criterion_sam   = SAMLoss()
    criterion_color = ColorLoss(device)
    criterion_gan   = nn.BCEWithLogitsLoss()

    try:
        criterion_perc = PerceptualLoss(device)
        use_perc = True
        log.info("Perceptual loss (VGG19): enabled")
    except Exception as e:
        log.warning(f"Perceptual loss disabled: {e}")
        use_perc = False

    lr = args.lr
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.999))
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, args.epochs, eta_min=1e-6)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, args.epochs, eta_min=1e-6)

    csv_path = MODELS_DIR / "training_log.csv"
    mode = "a" if (args.resume and csv_path.exists()) else "w"
    if mode == "w":
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","loss_G","loss_D",
                                     "loss_pixel","val_psnr","val_ssim"])

    start_epoch = 31 if (args.resume and csv_path.exists()) else 1

    for ep_idx in range(1, args.epochs + 1):
        epoch = start_epoch + ep_idx - 1
        G.train(); D.train()
        t0 = time.time()
        stats = dict(G=0.0, D=0.0, pix=0.0, edge=0.0, sam=0.0, col=0.0, n=0)

        for lr_b, hr_b in train_loader:
            lr_b = lr_b.to(device).float()
            hr_b = hr_b.to(device).float()
            sr_b = G(lr_b)

            # Discriminator
            opt_D.zero_grad()
            rl = D(hr_b); fl = D(sr_b.detach())
            loss_D = 0.5 * (criterion_gan(rl, torch.ones_like(rl)) +
                             criterion_gan(fl, torch.zeros_like(fl)))
            loss_D.backward(); opt_D.step()
            loss_D_val = loss_D.item()

            # Generator
            opt_G.zero_grad()
            loss_pix   = criterion_pixel(sr_b, hr_b)
            loss_edge  = criterion_edge(sr_b, hr_b)
            loss_sam   = criterion_sam(sr_b, hr_b)
            loss_color = criterion_color(sr_b, hr_b)

            loss_total = (L1_WEIGHT * loss_pix +
                          EDGE_WEIGHT * loss_edge +
                          SAM_WEIGHT * loss_sam +
                          COLOR_WEIGHT * loss_color)

            if use_perc:
                loss_total = loss_total + PERC_WEIGHT * criterion_perc(sr_b, hr_b)

            fl_g = D(sr_b)
            loss_total = loss_total + GAN_WEIGHT * criterion_gan(
                fl_g, torch.ones_like(fl_g))

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            stats["G"]    += loss_total.item()
            stats["D"]    += loss_D_val
            stats["pix"]  += loss_pix.item()
            stats["edge"] += loss_edge.item()
            stats["sam"]  += loss_sam.item()
            stats["col"]  += loss_color.item()
            stats["n"]    += 1

        sched_G.step(); sched_D.step()

        G.eval()
        vp = vs = vn = 0.0
        with torch.no_grad():
            for lr_v, hr_v in val_loader:
                lr_v = lr_v.to(device).float()
                hr_v = hr_v.to(device).float()
                sr_v = G(lr_v)
                vp += compute_psnr_batch(sr_v, hr_v)
                vs += compute_ssim_simple(sr_v, hr_v)
                vn += 1
        val_psnr = vp / max(vn,1)
        val_ssim = vs / max(vn,1)

        nb = max(stats["n"], 1)
        log.info(f"Epoch {epoch:3d}  "
                 f"G={stats['G']/nb:.4f}  pix={stats['pix']/nb:.4f}  "
                 f"edge={stats['edge']/nb:.4f}  sam={stats['sam']/nb:.4f}  col={stats['col']/nb:.4f}  "
                 f"PSNR={val_psnr:.2f}dB  SSIM={val_ssim:.4f}  "
                 f"({time.time()-t0:.1f}s)")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(stats["G"]/nb,6),
                round(stats["D"]/nb,6), round(stats["pix"]/nb,6),
                round(val_psnr,4), round(val_ssim,6)])

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "generator_state": G.state_dict(),
                "discriminator_state": D.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "config": {"in_ch": N_BANDS, "out_ch": N_BANDS,
                           "nf": FEAT_CH, "n_rrdb": N_RRDB}
            }, MODELS_DIR / "esrgan_sentinel2_best.pth")
            log.info(f"  ★ New best refined checkpoint saved (PSNR={val_psnr:.2f}dB, SSIM={val_ssim:.4f})")

    torch.save({
        "epoch": epoch,
        "generator_state": G.state_dict(),
        "val_psnr": best_psnr,
        "config": {"in_ch": N_BANDS, "out_ch": N_BANDS,
                   "nf": FEAT_CH, "n_rrdb": N_RRDB}
    }, MODELS_DIR / "esrgan_sentinel2.pth")

    log.info(f"Refinement complete! Best PSNR: {best_psnr:.2f} dB")
    log.info(f"Weights: {MODELS_DIR}/esrgan_sentinel2_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=3e-5)
    parser.add_argument("--quick",      action="store_true",
                        help="Fewer patches for quick training")
    parser.add_argument("--resume",     action="store_true", default=True)
    args = parser.parse_args()
    log.info("=" * 60)
    log.info("  SIH ESRGAN Refinement Training — Sentinel-2 4-Band SR")
    log.info("=" * 60)
    train(args)
