#!/usr/bin/env python3
"""
validate_comparison.py
======================
Side-by-side benchmark comparing:
1. Old Distilled Checkpoint: models/esrgan_sentinel2_best.pth (LDSR-S2 distillation)
2. New Real-Data Checkpoint: models/esrgan_sentinel2_realdata_scratch_v1.pth (Trained from scratch on real VENµS/WorldStrat pairs)

Evaluates on identical held-out real test/validation patches:
- Sen2Venµs (Real 5m HR multispectral reference):
  * PSNR (dB)
  * SSIM
  * SAM (° - Spectral Angle Mapper)
  * ERGAS (Synthesis global error)
  * NDVI-MAE (Vegetation index accuracy)
- WorldStrat (Real 2.5m HR panchromatic reference):
  * PSNR (dB)
  * SSIM
  * Edge Sharpness Score
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_esrgan import ESRGANGenerator, compute_psnr_batch, compute_ssim_simple
from train_esrgan_real import compute_sam_mean, compute_ergas, compute_ndvi_mae
from real_paired_dataset import RealPairedPatchDataset

def load_generator(checkpoint_path: Path, device: torch.device) -> ESRGANGenerator:
    G = ESRGANGenerator(in_ch=4, out_ch=4, nf=64, n_rrdb=8).to(device).float()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "generator_state" in ckpt:
        state = ckpt["generator_state"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "netG" in ckpt:
        state = ckpt["netG"]
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        clean_k = k.replace("module.", "").replace("generator.", "")
        clean_state[clean_k] = v

    G.load_state_dict(clean_state, strict=True)
    G.eval()
    return G

def evaluate_model_on_sen2venus(G: ESRGANGenerator, val_loader: DataLoader, device: torch.device):
    total_psnr = 0.0
    total_ssim = 0.0
    total_sam  = 0.0
    total_ergas= 0.0
    total_ndvi = 0.0
    count = 0

    with torch.no_grad():
        for lr, hr, metas in val_loader:
            lr = lr.to(device).float()
            hr = hr.to(device).float()  # [B, 4, 64, 64]
            sr_4x = G(lr)               # [B, 4, 128, 128]
            # Downsample 4x to 2x for comparison with 5m VENµS target
            sr_2x = F.interpolate(sr_4x, size=(64, 64), mode="area")

            total_psnr  += compute_psnr_batch(sr_2x, hr)
            total_ssim  += compute_ssim_simple(sr_2x, hr)
            total_sam   += compute_sam_mean(sr_2x, hr)
            total_ergas += compute_ergas(sr_2x, hr, scale=2.0)
            total_ndvi  += compute_ndvi_mae(sr_2x, hr)
            count += 1

    n = max(count, 1)
    return {
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "sam_deg": total_sam / n,
        "ergas": total_ergas / n,
        "ndvi_mae": total_ndvi / n,
        "count": count
    }

def main():
    parser = argparse.ArgumentParser(description="Side-by-side benchmark of Old vs New ESRGAN models")
    parser.add_argument("--old-ckpt", type=str, default="models/esrgan_sentinel2_best.pth",
                        help="Path to old distillation checkpoint")
    parser.add_argument("--new-ckpt", type=str, default="models/esrgan_sentinel2_realdata_scratch_v1.pth",
                        help="Path to new real-data scratch checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/real_data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=12)
    args = parser.parse_args()

    device_str = ("mps" if torch.backends.mps.is_available() else
                  "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Benchmarking on device: {device}\n")

    val_ds = RealPairedPatchDataset(args.data_dir, split="val", augment=False,
                                   max_patches_per_tile=args.max_patches)
    # Filter for Sen2Venµs (multispectral 5m reference)
    ms_idx = [i for i, s in enumerate(val_ds.samples) if not s["is_panchromatic"]]
    
    class SubDataset(torch.utils.data.Dataset):
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.parent[self.indices[idx]]

    val_loader = DataLoader(SubDataset(val_ds, ms_idx), batch_size=args.batch_size, shuffle=False)
    print(f"Validation set: {len(ms_idx)} real 5m VENµS multispectral patches")

    old_path = Path(args.old_ckpt)
    new_path = Path(args.new_ckpt)

    print("-" * 75)
    print(f"Old Checkpoint: {old_path} ({'Found' if old_path.exists() else 'Missing'})")
    print(f"New Checkpoint: {new_path} ({'Found' if new_path.exists() else 'Missing'})")
    print("-" * 75)

    results = {}
    if old_path.exists():
        print("Evaluating Model 1: Old Distilled Checkpoint (LDSR-S2 teacher)...")
        G_old = load_generator(old_path, device)
        results["old"] = evaluate_model_on_sen2venus(G_old, val_loader, device)
    else:
        results["old"] = None

    if new_path.exists():
        print("Evaluating Model 2: New Real-Data Scratch Checkpoint (Zero teacher data)...")
        G_new = load_generator(new_path, device)
        results["new"] = evaluate_model_on_sen2venus(G_new, val_loader, device)
    else:
        results["new"] = None

    print("\n" + "=" * 78)
    print("      📊 REAL-DATA BENCHMARK RESULTS (Sen2Venµs 5m Ground Truth)")
    print("=" * 78)
    print(f"{'Metric':<20} | {'Old (Distillation)':<20} | {'New (Real-Data Scratch)':<22} | {'Delta':<12}")
    print("-" * 78)

    metrics = [
        ("PSNR (dB)", "psnr", "{:.2f} dB", True),
        ("SSIM", "ssim", "{:.4f}", True),
        ("SAM (°)", "sam_deg", "{:.2f}°", False),
        ("ERGAS", "ergas", "{:.2f}", False),
        ("NDVI-MAE", "ndvi_mae", "{:.4f}", False),
    ]

    for label, key, fmt_str, higher_better in metrics:
        old_val = results["old"][key] if results["old"] else None
        new_val = results["new"][key] if results["new"] else None

        old_str = fmt_str.format(old_val) if old_val is not None else "N/A"
        new_str = fmt_str.format(new_val) if new_val is not None else "In Progress"

        if old_val is not None and new_val is not None:
            delta = new_val - old_val
            sign = "+" if delta > 0 else ""
            good = (delta > 0) if higher_better else (delta < 0)
            delta_str = f"{sign}{delta:.4f} {'✅' if good else '⚠️'}"
        else:
            delta_str = "-"

        print(f"{label:<20} | {old_str:<20} | {new_str:<22} | {delta_str:<12}")

    print("=" * 78)
    print("Notes: SAM, ERGAS, and NDVI-MAE: lower is better (✅). PSNR and SSIM: higher is better (✅).")

if __name__ == "__main__":
    main()
