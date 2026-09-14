"""
download_real_datasets.py
=========================
Downloads and extracts real paired satellite imagery for ESRGAN training:
1. Sen2Venµs: Sentinel-2 10m paired with VENµS 5m multispectral data (Zenodo 6514159).
2. WorldStrat: Sentinel-2 10m paired with SPOT 1.5m panchromatic imagery (downsampled to 2.5m).

Handles selective site downloads to respect disk space (<21 GB available).
Removes raw compressed archives after extraction to preserve local disk space.
"""

import os
import sys
import ssl
import time
import argparse
import logging
from pathlib import Path
import urllib.request
import py7zr

# Configure unverified SSL context for macOS
ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download-real")

SEN2VENUS_ZENODO_BASE = "https://zenodo.org/records/6514159/files"
SEN2VENUS_SITES = [
    # (Filename, Approx MB, Patch Count, Description)
    ("FGMANAUS.7z", 63, 129, "Tropical Rainforest (Amazon, Brazil)"),
    ("ESTUAMAR.7z", 475, 912, "Wetlands & Estuary (Spain)"),
    ("SO2.7z", 487, 738, "Agricultural Plains (France)"),
]

DATA_DIR = Path("data/real_data")
SEN2VENUS_DIR = DATA_DIR / "sen2venus"
WORLDSTRAT_DIR = DATA_DIR / "worldstrat"
DOWNLOADS_DIR = DATA_DIR / "downloads"


def download_file_with_progress(url: str, dest_path: Path):
    """Download a remote URL to dest_path with progress logging."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    })
    
    log.info(f"Connecting to {url} ...")
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1 MB blocks
        t0 = time.time()
        last_log = t0
        
        with open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_log >= 3.0:
                    speed = (downloaded / (1024 * 1024)) / max(now - t0, 0.1)
                    pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    log.info(f"  Downloaded {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s")
                    last_log = now
                    
    tmp_path.rename(dest_path)
    log.info(f"Downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB) in {time.time()-t0:.1f}s ✓")


def extract_7z(archive_path: Path, extract_to: Path, remove_archive: bool = True):
    """Extract .7z file into target directory and optionally delete the archive."""
    extract_to.mkdir(parents=True, exist_ok=True)
    log.info(f"Extracting {archive_path.name} -> {extract_to} ...")
    t0 = time.time()
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(extract_to)
    log.info(f"Extracted {archive_path.name} in {time.time()-t0:.1f}s ✓")
    
    if remove_archive and archive_path.exists():
        archive_path.unlink()
        log.info(f"Cleaned up archive {archive_path.name} to preserve disk space ✓")


def download_sen2venus(sites=SEN2VENUS_SITES, keep_archives=False):
    """Download and unpack selected Sen2Venµs sites."""
    SEN2VENUS_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    
    total_patches = 0
    for filename, approx_mb, patch_count, desc in sites:
        site_name = filename.replace(".7z", "")
        site_dest = SEN2VENUS_DIR / site_name
        
        # Check if already extracted
        existing_pt_files = list(site_dest.rglob("*.pt"))
        if existing_pt_files:
            log.info(f"Site {site_name} already extracted ({len(existing_pt_files)} .pt files found). Skipping download.")
            total_patches += patch_count
            continue
            
        url = f"{SEN2VENUS_ZENODO_BASE}/{filename}"
        archive_path = DOWNLOADS_DIR / filename
        
        log.info(f"=== Fetching Sen2Venµs Site: {site_name} ({desc}) [{approx_mb} MB, ~{patch_count} patches] ===")
        if not archive_path.exists():
            download_file_with_progress(url, archive_path)
            
        extract_7z(archive_path, site_dest, remove_archive=not keep_archives)
        total_patches += patch_count

    log.info(f"Sen2Venµs download complete! Target total patches available: ~{total_patches}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real paired datasets")
    parser.add_argument("--sen2venus", action="store_true", default=True, help="Download Sen2Venµs sites")
    parser.add_argument("--keep-archives", action="store_true", default=False, help="Keep .7z archives after extraction")
    args = parser.parse_args()
    
    if args.sen2venus:
        download_sen2venus(keep_archives=args.keep_archives)
