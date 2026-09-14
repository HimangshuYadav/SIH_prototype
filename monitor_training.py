#!/usr/bin/env python3
"""
monitor_training.py
===================
Real-time CLI dashboard for tracking real-data ESRGAN training progress.
Run with:
    python3 monitor_training.py          (single snapshot)
    python3 monitor_training.py --watch  (live updating dashboard every 1s)
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path

CSV_PATH = Path("models/realdata_training_log.csv")
MODEL_PATH = Path("models/esrgan_sentinel2_realdata_scratch_v1.pth")
PROGRESS_PATH = Path(".training_progress.json")

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

def get_process_info():
    try:
        res = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        lines = [l for l in res.stdout.splitlines() if "train_esrgan_real.py" in l and "grep" not in l]
        if lines:
            parts = lines[0].split()
            cpu = parts[2] if len(parts) > 2 else "?"
            mem = parts[3] if len(parts) > 3 else "?"
            time_cpu = parts[9] if len(parts) > 9 else "?"
            return True, f"PID {parts[1]} (CPU: {cpu}%, MEM: {mem}%, Active Time: {time_cpu})"
        return False, "Not running"
    except Exception:
        return False, "Unknown"

def render_dashboard(spin_idx=0):
    is_running, proc_str = get_process_info()
    spin = SPINNER[spin_idx % len(SPINNER)]

    print("\033[H\033[J", end="")  # Clear terminal screen cleanly
    print("=" * 78)
    print(f"      🛰️  REAL-DATA ESRGAN LIVE TRAINING MONITOR (Sentinel-2 SR)")
    print("=" * 78)

    status_icon = f"🟢 {spin} ACTIVE" if is_running else "⚪ IDLE / COMPLETED"
    print(f"Status:      {status_icon} | {proc_str}")

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        mtime = os.path.getmtime(MODEL_PATH)
        t_str = time.strftime('%H:%M:%S', time.localtime(mtime))
        print(f"Checkpoint:  ✅ {MODEL_PATH.name} ({size_mb:.1f} MB, latest update at {t_str})")
    else:
        print(f"Checkpoint:  ⏳ Initializing scratch weights...")

    # Check live per-batch progress file
    if is_running and PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r") as pf:
                prog = json.load(pf)
            pct = prog.get("pct", 0)
            step = prog.get("step", 0)
            total = prog.get("total_steps", 1)
            bar_len = 25
            filled = int(bar_len * (pct / 100.0))
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"Live Batch:  [{bar}] {pct:.1f}% ({step}/{total}) | Epoch {prog.get('epoch')}/{prog.get('total_epochs')}")
            print(f"Live Losses: Loss_G: {prog.get('loss_G', 0):.4f} | L1: {prog.get('loss_pix', 0):.4f} | SAM: {prog.get('loss_sam', 0):.4f} | Edge: {prog.get('loss_edge', 0):.4f}")
        except Exception:
            pass
    elif is_running:
        print(f"Live Step:   {spin} Epoch in progress, validating on real satellite patches...")

    print("-" * 78)

    if not CSV_PATH.exists():
        print(f"Log file '{CSV_PATH}' will appear once epoch 1 metrics are recorded.")
        print("=" * 78)
        return

    with open(CSV_PATH, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) <= 1:
        print("⏳ Epoch 1 in progress — metrics log at end of each epoch.")
        print("=" * 78)
        return

    rows = [l.split(",") for l in lines[1:]]

    fmt = "{:<6} | {:<8} | {:<8} | {:<8} | {:<12} | {:<9} | {:<10} | {:<10}"
    print(fmt.format("Epoch", "Loss_G", "Loss_pix", "Loss_SAM", "Val_PSNR(dB)", "Val_SSIM", "Val_SAM(°)", "NDVI_MAE"))
    print("-" * 78)

    for r in rows:
        if len(r) >= 12:
            ep = r[0]
            lg = r[1][:6]
            lp = r[3][:6]
            lsam = r[6][:6]
            v_psnr = f"{float(r[8]):.2f} dB" if r[8] else "-"
            v_ssim = f"{float(r[9]):.4f}" if r[9] else "-"
            v_sam  = f"{float(r[10]):.2f}°" if r[10] else "-"
            v_ndvi = f"{float(r[11]):.4f}" if r[11] else "-"
            flag   = f" {r[12]}" if len(r) > 12 and r[12] else ""
            print(fmt.format(ep, lg, lp, lsam, v_psnr, v_ssim, v_sam, v_ndvi) + flag)

    print("=" * 78)
    if is_running:
        print("💡 Press Ctrl+C to exit monitor (training continues running uninterrupted).")

def main():
    parser = argparse.ArgumentParser(description="Live ESRGAN training monitor")
    parser.add_argument("--watch", "-w", action="store_true", help="Live auto-refresh dashboard mode")
    parser.add_argument("--interval", type=float, default=1.5, help="Refresh interval in seconds")
    args = parser.parse_args()

    if not args.watch:
        render_dashboard(0)
        return

    spin_idx = 0
    try:
        while True:
            render_dashboard(spin_idx)
            spin_idx += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")

if __name__ == "__main__":
    main()
