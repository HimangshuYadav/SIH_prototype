"""
train_esrgan_real.py
====================
Train 4-band ESRGAN (RRDBNet) completely from scratch with random weight
initialization, using only real paired high-resolution satellite imagery.
NO teacher distillation data. NO pre-trained checkpoint loading.

Architecture: ESRGANGenerator (in_ch=4, out_ch=4, nf=64, n_rrdb=8) -- UNCHANGED
Input:  [B, 4, 32, 32]   (10m Sentinel-2)
Output: [B, 4, 128, 128] (2.5m SR output)

Data sources (real paired HR):
  1. Sen2Venµs: 10m S2 LR  →  5m VENµS HR (2× pairing; model predicts 4×, then 2x-pooled for loss)
  2. WorldStrat: 10m LR  →  2.5m panchromatic (4× pairing, structure-only losses)

Checkpoint saved to: models/esrgan_sentinel2_realdata_scratch_v1.pth
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("real-train")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = Path(".training_progress.json")

# ============================================================
#  IMPORT ARCHITECTURE & LOSSES FROM train_esrgan.py  (unchanged)
# ============================================================
from train_esrgan import (
    ESRGANGenerator,
    PatchDiscriminator,
    EdgeLoss,
    SAMLoss,
    ColorLoss,
    PerceptualLoss,
    compute_psnr_batch,
    compute_ssim_simple,
)
# Import Laplacian & Contrast from distill_esrgan.py (reused loss modules, not data)
from distill_esrgan import LaplacianLoss, ContrastLoss

# Import the real paired dataset
from real_paired_dataset import RealPairedPatchDataset

# Hyperparameters
N_BANDS     = 4
N_RRDB      = 8
FEAT_CH     = 64

# Loss weights
L1_WEIGHT      = 1.00
EDGE_WEIGHT    = 0.50
LAP_WEIGHT     = 0.30
CONTRAST_WEIGHT= 0.20
SAM_WEIGHT     = 0.10   # multispectral only
COLOR_WEIGHT   = 0.30   # multispectral only
PERC_WEIGHT    = 0.12
GAN_WEIGHT     = 5e-3


def compute_ergas(sr: torch.Tensor, hr: torch.Tensor, scale: float) -> float:
    """ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse), lower is better."""
    ergas_sum = 0.0
    B = hr.shape[1]
    for b in range(B):
        rmse = torch.sqrt(F.mse_loss(sr[:, b], hr[:, b])).item()
        mean_hr = hr[:, b].mean().item()
        if mean_hr > 1e-6:
            ergas_sum += (rmse / mean_hr) ** 2
    return float(100 / scale * np.sqrt(ergas_sum / B))


def compute_sam_mean(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """Mean Spectral Angle Mapper in degrees, lower is better."""
    cos_sim = F.cosine_similarity(sr.clamp(0, 1), hr.clamp(0, 1), dim=1, eps=1e-6)
    sam_rad = torch.acos(cos_sim.clamp(-1.0, 1.0))
    return float(sam_rad.mean().item() * 180.0 / np.pi)


def compute_ndvi_mae(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Mean absolute error of NDVI.
    Bands: B02=ch0, B03=ch1, B04=ch2, B08=ch3
    NDVI = (B08 - B04) / (B08 + B04)
    """
    def ndvi(t):
        b8, b4 = t[:, 3], t[:, 2]
        denom = (b8 + b4).clamp(min=1e-6)
        return (b8 - b4) / denom
    return float(F.l1_loss(ndvi(sr.clamp(0, 1)), ndvi(hr.clamp(0, 1))).item())


def compute_composite_loss(
    sr_4x: torch.Tensor,
    hr_batch: torch.Tensor,
    metas: list,
    device,
    criterion_pixel: nn.Module,
    criterion_edge:  nn.Module,
    criterion_lap:   nn.Module,
    criterion_cont:  nn.Module,
    criterion_sam:   nn.Module,
    criterion_color: nn.Module,
    criterion_perc:  nn.Module,
    use_perc: bool,
) -> dict:
    """
    Compute multi-task loss, routing spectral losses correctly:
    - Sen2Venµs (2x): Downsample 4x SR to 2x for L1/SAM/Color/Sobel vs 64x64 target.
    - WorldStrat (4x, panchromatic): Use only structural losses on luminance.
    Returns dict of all loss scalars and the total loss tensor.
    """
    is_pan_flags = [m["is_panchromatic"] for m in metas]
    all_same_source = (all(is_pan_flags) or not any(is_pan_flags))

    # Split batch into multispectral (Sen2Venµs) and panchromatic (WorldStrat)
    ms_indices  = [i for i, f in enumerate(is_pan_flags) if not f]
    pan_indices = [i for i, f in enumerate(is_pan_flags) if f]

    total_loss = torch.zeros(1, device=device, requires_grad=True)
    stats = {k: 0.0 for k in ["l1", "edge", "lap", "contrast", "sam", "color", "perc"]}

    # ---- Multispectral (Sen2Venµs) ----
    if ms_indices:
        idx = torch.tensor(ms_indices, device=device)
        sr_ms = sr_4x[idx]     # [N, 4, 128, 128]
        hr_ms = hr_batch[idx]  # [N, 4, 64, 64] (5m VENµS)

        # Downsample 4x SR prediction to 2x (64x64) for comparison with 5m VENµS target
        sr_ms_2x = F.interpolate(sr_ms, size=(64, 64), mode="area")

        l_pix   = criterion_pixel(sr_ms_2x, hr_ms)
        l_edge  = criterion_edge(sr_ms_2x, hr_ms)
        l_lap   = criterion_lap(sr_ms_2x, hr_ms)
        l_cont  = criterion_cont(sr_ms_2x, hr_ms)
        l_sam   = criterion_sam(sr_ms_2x, hr_ms)
        l_color = criterion_color(sr_ms_2x, hr_ms)

        ms_loss = (L1_WEIGHT * l_pix + EDGE_WEIGHT * l_edge + LAP_WEIGHT * l_lap
                   + CONTRAST_WEIGHT * l_cont + SAM_WEIGHT * l_sam + COLOR_WEIGHT * l_color)

        if use_perc:
            l_perc = criterion_perc(sr_ms_2x, hr_ms)
            ms_loss = ms_loss + PERC_WEIGHT * l_perc
            stats["perc"] += l_perc.item()

        total_loss = total_loss + ms_loss
        stats["l1"]      += l_pix.item()
        stats["edge"]    += l_edge.item()
        stats["lap"]     += l_lap.item()
        stats["contrast"]+= l_cont.item()
        stats["sam"]     += l_sam.item()
        stats["color"]   += l_color.item()

    # ---- Panchromatic (WorldStrat, structural losses only) ----
    if pan_indices:
        idx = torch.tensor(pan_indices, device=device)
        sr_pan_all = sr_4x[idx]    # [N, 4, 128, 128]
        hr_pan = hr_batch[idx]     # [N, 1, 128, 128] (2.5m panchromatic)

        # Synthesize luminance from 4-band SR using standard weights B04=R, B03=G, B02=B
        # Bands: B02=ch0, B03=ch1, B04=ch2
        sr_lum = (0.299 * sr_pan_all[:, 2:3] + 0.587 * sr_pan_all[:, 1:2] + 0.114 * sr_pan_all[:, 0:1])

        l_edge_p  = criterion_edge(sr_lum.repeat(1, 4, 1, 1), hr_pan.repeat(1, 4, 1, 1))
        l_lap_p   = criterion_lap(sr_lum.repeat(1, 4, 1, 1), hr_pan.repeat(1, 4, 1, 1))
        l_cont_p  = criterion_cont(sr_lum, hr_pan)

        pan_loss = (EDGE_WEIGHT * l_edge_p + LAP_WEIGHT * l_lap_p + CONTRAST_WEIGHT * l_cont_p)
        total_loss = total_loss + 0.5 * pan_loss  # lower weight for structural-only supervision

    return {"total": total_loss, **stats}


def train(args):
    device_str = ("mps"  if torch.backends.mps.is_available() else
                  "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    log.info(f"Training on: {device}")

    # ---- Datasets ----
    max_patches = 4 if args.quick else 12
    train_ds = RealPairedPatchDataset(
        "data/real_data", split="train", augment=True,
        max_patches_per_tile=max_patches
    )
    val_ds = RealPairedPatchDataset(
        "data/real_data", split="val", augment=False,
        max_patches_per_tile=max_patches
    )

    if len(train_ds) == 0:
        log.error("No real training patches found! Run download_real_datasets.py first.")
        return

    log.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    src_count = Counter(s["source"] for s in train_ds.samples)
    log.info(f"Train sources: { {k: v for k, v in src_count.items()} }")

    # ---- Batching strategy: keep Sen2Venµs & WorldStrat in same batch (mixed allowed) ----
    # Need a custom collate since HR shapes differ (64x64 vs 128x128)
    def collate_mixed(batch):
        lrs = torch.stack([b[0] for b in batch])
        hrs = [b[1] for b in batch]
        metas = [b[2] for b in batch]
        same_hr = all(h.shape == hrs[0].shape for h in hrs)
        if same_hr:
            hrs_t = torch.stack(hrs)
        else:
            hrs_t = hrs  # list – handled per-sample
        return lrs, hrs_t, metas

    # For simplicity in batching, separate loaders by source type
    ms_samples  = [s for s in train_ds.samples if not s["is_panchromatic"]]
    pan_samples = [s for s in train_ds.samples if s["is_panchromatic"]]

    class SubDataset(torch.utils.data.Dataset):
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.parent[self.indices[idx]]

    ms_idx  = [i for i, s in enumerate(train_ds.samples) if not s["is_panchromatic"]]
    pan_idx = [i for i, s in enumerate(train_ds.samples) if s["is_panchromatic"]]

    ms_loader  = DataLoader(SubDataset(train_ds, ms_idx),  batch_size=args.batch_size,
                            shuffle=True, num_workers=0, drop_last=True)
    pan_loader = DataLoader(SubDataset(train_ds, pan_idx), batch_size=args.batch_size,
                            shuffle=True, num_workers=0, drop_last=False)

    val_ms_idx  = [i for i, s in enumerate(val_ds.samples) if not s["is_panchromatic"]]
    val_loader  = DataLoader(SubDataset(val_ds, val_ms_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # ---- Model: fresh random initialization ----
    G = ESRGANGenerator(in_ch=N_BANDS, out_ch=N_BANDS, nf=FEAT_CH, n_rrdb=N_RRDB).to(device).float()
    D = PatchDiscriminator(in_ch=N_BANDS).to(device).float()
    log.info(f"Generator: {sum(p.numel() for p in G.parameters())/1e6:.2f}M params (FRESH INIT, NO distillation weights)")
    log.info(f"Discriminator: {sum(p.numel() for p in D.parameters())/1e6:.2f}M params")

    # ---- Losses ----
    criterion_pixel = nn.L1Loss()
    criterion_edge  = EdgeLoss(device)
    criterion_lap   = LaplacianLoss(device)
    criterion_cont  = ContrastLoss()
    criterion_sam   = SAMLoss()
    criterion_color = ColorLoss(device)
    criterion_gan   = nn.BCEWithLogitsLoss()

    try:
        criterion_perc = PerceptualLoss(device)
        use_perc = True
        log.info("PerceptualLoss (VGG19): enabled")
    except Exception as e:
        log.warning(f"PerceptualLoss disabled: {e}")
        use_perc = False

    # ---- Optimizers & Schedulers ----
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, args.epochs, eta_min=1e-6)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, args.epochs, eta_min=1e-6)

    # ---- CSV logging ----
    csv_path = MODELS_DIR / "realdata_training_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "loss_G", "loss_D", "loss_pix", "loss_edge", "loss_lap",
            "loss_sam", "loss_color", "val_psnr", "val_ssim", "val_sam_deg",
            "val_ndvi_mae", "overfit_flag"
        ])

    best_psnr  = 0.0
    val_hist   = []  # for overfitting detection
    pan_iter = iter(pan_loader)

    log.info("=" * 60)
    log.info("  REAL-DATA ESRGAN TRAINING FROM SCRATCH — No Teacher Data")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        stats = {k: 0.0 for k in ["G", "D", "pix", "edge", "lap", "sam", "color", "n"]}

        # Alternate: one MS batch + occasional PAN batch
        pbar = tqdm(enumerate(ms_loader), total=len(ms_loader), desc=f"Epoch {epoch:2d}/{args.epochs}", dynamic_ncols=True, leave=True)
        for step, ms_batch in pbar:
            lr_ms, hr_ms_batch, ms_metas = ms_batch
            lr_ms  = lr_ms.to(device).float()
            hr_ms  = hr_ms_batch.to(device).float()  # [B, 4, 64, 64]

            sr_ms_4x = G(lr_ms)  # [B, 4, 128, 128]

            # ---- Discriminator step on multispectral ----
            opt_D.zero_grad()
            sr_2x = F.interpolate(sr_ms_4x.detach(), size=(64, 64), mode="area")
            rl = D(hr_ms); fl = D(sr_2x.detach())
            loss_D = 0.5 * (criterion_gan(rl, torch.ones_like(rl)) +
                            criterion_gan(fl, torch.zeros_like(fl)))
            loss_D.backward(); opt_D.step()

            # ---- Generator step on multispectral ----
            opt_G.zero_grad()
            sr_2x = F.interpolate(sr_ms_4x, size=(64, 64), mode="area")
            l_pix   = criterion_pixel(sr_2x, hr_ms)
            l_edge  = criterion_edge(sr_2x, hr_ms)
            l_lap   = criterion_lap(sr_2x, hr_ms)
            l_cont  = criterion_cont(sr_2x, hr_ms)
            l_sam   = criterion_sam(sr_2x, hr_ms)
            l_color = criterion_color(sr_2x, hr_ms)
            fl_g    = D(sr_2x)

            loss_G = (L1_WEIGHT * l_pix + EDGE_WEIGHT * l_edge + LAP_WEIGHT * l_lap
                      + CONTRAST_WEIGHT * l_cont + SAM_WEIGHT * l_sam + COLOR_WEIGHT * l_color
                      + GAN_WEIGHT * criterion_gan(fl_g, torch.ones_like(fl_g)))

            if use_perc:
                loss_G = loss_G + PERC_WEIGHT * criterion_perc(sr_2x, hr_ms)

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            # ---- Occasional PAN structural step (every 4 MS steps) ----
            if pan_idx and step % 4 == 0:
                try:
                    pan_batch = next(pan_iter)
                except StopIteration:
                    pan_iter = iter(pan_loader)
                    pan_batch = next(pan_iter)

                lr_pan, hr_pan_batch, _ = pan_batch
                lr_pan = lr_pan.to(device).float()
                hr_pan = hr_pan_batch.to(device).float()  # [B, 1, 128, 128]

                opt_G.zero_grad()
                sr_pan_4x = G(lr_pan)  # [B, 4, 128, 128]
                sr_lum = (0.299 * sr_pan_4x[:, 2:3] + 0.587 * sr_pan_4x[:, 1:2] + 0.114 * sr_pan_4x[:, 0:1])
                l_pan = (EDGE_WEIGHT  * criterion_edge(sr_lum.expand(-1, 4, -1, -1), hr_pan.expand(-1, 4, -1, -1))
                       + LAP_WEIGHT   * criterion_lap(sr_lum.expand(-1, 4, -1, -1), hr_pan.expand(-1, 4, -1, -1))
                       + CONTRAST_WEIGHT * criterion_cont(sr_lum, hr_pan))
                (0.5 * l_pan).backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_G.step()

            stats["G"]    += loss_G.item()
            stats["D"]    += loss_D.item()
            stats["pix"]  += l_pix.item()
            stats["edge"] += l_edge.item()
            stats["lap"]  += l_lap.item()
            stats["sam"]  += l_sam.item()
            stats["color"]+= l_color.item()
            stats["n"]    += 1

            if step % 5 == 0 or step == len(ms_loader) - 1:
                pbar.set_postfix({
                    "G": f"{loss_G.item():.3f}",
                    "pix": f"{l_pix.item():.4f}",
                    "sam": f"{l_sam.item():.4f}",
                    "edge": f"{l_edge.item():.4f}",
                })
                try:
                    with open(PROGRESS_FILE, "w") as pf:
                        json.dump({
                            "epoch": epoch,
                            "total_epochs": args.epochs,
                            "step": step + 1,
                            "total_steps": len(ms_loader),
                            "pct": round((step + 1) / len(ms_loader) * 100, 1),
                            "loss_G": round(loss_G.item(), 4),
                            "loss_pix": round(l_pix.item(), 4),
                            "loss_sam": round(l_sam.item(), 4),
                            "loss_edge": round(l_edge.item(), 4),
                            "updated_at": time.time(),
                        }, pf)
                except Exception:
                    pass

        sched_G.step(); sched_D.step()

        # ---- Validation (multispectral only, against real 5m VENµS) ----
        G.eval()
        vp = vs = vn = v_sam = v_ndvi = 0.0
        val_pbar = tqdm(val_loader, desc="  🔍 Validating vs Real 5m VENµS", dynamic_ncols=True, leave=False)
        with torch.no_grad():
            for lr_v, hr_v, metas_v in val_pbar:
                lr_v = lr_v.to(device).float()
                hr_v = hr_v.to(device).float()  # [B, 4, 64, 64]
                sr_v = G(lr_v)
                sr_v_2x = F.interpolate(sr_v, size=(64, 64), mode="area")
                vp    += compute_psnr_batch(sr_v_2x, hr_v)
                vs    += compute_ssim_simple(sr_v_2x, hr_v)
                v_sam += compute_sam_mean(sr_v_2x, hr_v)
                v_ndvi+= compute_ndvi_mae(sr_v_2x, hr_v)
                vn += 1

        val_psnr      = vp    / max(vn, 1)
        val_ssim      = vs    / max(vn, 1)
        val_sam_deg   = v_sam / max(vn, 1)
        val_ndvi_mae  = v_ndvi/ max(vn, 1)

        # ---- Overfitting detection ----
        val_hist.append(val_psnr)
        overfit_flag = ""
        if len(val_hist) >= 4:
            recent = val_hist[-3:]
            if all(recent[i] <= recent[i-1] for i in range(1, len(recent))):
                overfit_flag = "⚠ VAL_PLATEAU"
                log.warning(f"  ⚠ Validation PSNR has plateaued for 3 epochs ({recent}).")

        nb = max(stats["n"], 1)
        log.info(f"Epoch {epoch:3d}/{args.epochs}  "
                 f"G={stats['G']/nb:.4f}  pix={stats['pix']/nb:.4f}  "
                 f"edge={stats['edge']/nb:.4f}  sam={stats['sam']/nb:.4f}  "
                 f"| Val PSNR={val_psnr:.2f}dB  SSIM={val_ssim:.4f}  "
                 f"SAM={val_sam_deg:.2f}°  NDVI-MAE={val_ndvi_mae:.4f}  "
                 f"({time.time()-t0:.1f}s) {overfit_flag}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                round(stats["G"]/nb, 6),   round(stats["D"]/nb, 6),
                round(stats["pix"]/nb, 6),  round(stats["edge"]/nb, 6),
                round(stats["lap"]/nb, 6),  round(stats["sam"]/nb, 6),
                round(stats["color"]/nb, 6),
                round(val_psnr, 4),   round(val_ssim, 6),
                round(val_sam_deg, 4), round(val_ndvi_mae, 6),
                overfit_flag
            ])

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "generator_state": G.state_dict(),
                "discriminator_state": D.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_sam_deg": val_sam_deg,
                "val_ndvi_mae": val_ndvi_mae,
                "config": {"in_ch": N_BANDS, "out_ch": N_BANDS, "nf": FEAT_CH, "n_rrdb": N_RRDB},
                "training_mode": "real_data_from_scratch",
                "data_sources": ["sen2venus", "worldstrat"],
            }, MODELS_DIR / "esrgan_sentinel2_realdata_scratch_v1.pth")
            log.info(f"  ★ New best real-data checkpoint (PSNR={val_psnr:.2f}dB, SAM={val_sam_deg:.2f}°, NDVI-MAE={val_ndvi_mae:.4f})")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "generator_state": G.state_dict(),
                "val_psnr": val_psnr,
                "config": {"in_ch": N_BANDS, "out_ch": N_BANDS, "nf": FEAT_CH, "n_rrdb": N_RRDB},
            }, MODELS_DIR / f"esrgan_realdata_epoch{epoch:03d}.pth")

    # Final save
    torch.save({
        "epoch": args.epochs,
        "generator_state": G.state_dict(),
        "val_psnr": best_psnr,
        "config": {"in_ch": N_BANDS, "out_ch": N_BANDS, "nf": FEAT_CH, "n_rrdb": N_RRDB},
        "training_mode": "real_data_from_scratch",
    }, MODELS_DIR / "esrgan_sentinel2_realdata_scratch_final.pth")

    log.info(f"Training complete! Best PSNR (vs real 5m VENµS): {best_psnr:.2f} dB")
    log.info(f"Weights saved → {MODELS_DIR}/esrgan_sentinel2_realdata_scratch_v1.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESRGAN Real-Data Scratch Training")
    parser.add_argument("--epochs",      type=int,   default=25)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--quick",       action="store_true",
                        help="Fewer patches (for smoke-test / fast validation)")
    args = parser.parse_args()
    train(args)
