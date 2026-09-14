"""
fetch_worldstrat_samples.py
===========================
Fetches real paired WorldStrat Panchromatic (1.5m -> 2.5m) imagery using
HTTP Range requests from Hugging Face mirror without downloading the 40GB archive.
Saves extracted paired patches to data/real_data/worldstrat/*.npz
"""

import io
import ssl
import time
import struct
import zlib
import logging
from pathlib import Path
import urllib.request

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

ssl._create_default_https_context = ssl._create_unverified_context
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch-worldstrat")

HR_ZIP_URL = "https://huggingface.co/datasets/Khlaifiabilel/worldstrat-agrisr/resolve/main/hr_dataset.zip"
HR_CD_OFFSET = 40826293489
DEST_DIR = Path("data/real_data/worldstrat")
DEST_DIR.mkdir(parents=True, exist_ok=True)


def extract_file_from_remote_zip(url: str, loc_off: int, comp_m: int, comp_sz: int):
    """Extract a specific file payload from remote zip using HTTP range requests."""
    # Read local header (30 bytes)
    req_lh = urllib.request.Request(url, headers={"Range": f"bytes={loc_off}-{loc_off+30}"})
    with urllib.request.urlopen(req_lh) as resp:
        lh = resp.read()
    lh_fn_len, lh_ex_len = struct.unpack("<HH", lh[26:30])
    data_start = loc_off + 30 + lh_fn_len + lh_ex_len

    # Fetch compressed payload
    req_data = urllib.request.Request(url, headers={"Range": f"bytes={data_start}-{data_start+comp_sz-1}"})
    with urllib.request.urlopen(req_data) as resp:
        raw_data = resp.read()

    if comp_m == 8:  # Deflate
        payload = zlib.decompress(raw_data, -15)
    else:
        payload = raw_data
    return payload


def fetch_worldstrat_pan_samples(target_aois: int = 8, patches_per_aoi: int = 16):
    """
    Scans the central directory of hr_dataset.zip and extracts SPOT 1.5m panchromatic rasters,
    downsamples them to 2.5m (4x relative to 10m Sentinel-2), and saves paired structural patches.
    """
    log.info(f"Connecting to WorldStrat HR archive on Hugging Face (target AOIs: {target_aois}) ...")
    
    # Read 512KB of the central directory
    req_cd = urllib.request.Request(HR_ZIP_URL, headers={"Range": f"bytes={HR_CD_OFFSET}-{HR_CD_OFFSET + 524288}"})
    with urllib.request.urlopen(req_cd) as resp:
        cd = resp.read()

    pan_entries = []
    idx = 0
    while idx < len(cd):
        if cd[idx:idx+4] == b"PK\x01\x02":
            comp_m = struct.unpack("<H", cd[idx+10:idx+12])[0]
            comp_sz, uncomp_sz = struct.unpack("<II", cd[idx+20:idx+28])
            fn_len, extra_len, comm_len = struct.unpack("<HHH", cd[idx+28:idx+34])
            loc_off = struct.unpack("<I", cd[idx+42:idx+46])[0]
            fn = cd[idx+46:idx+46+fn_len].decode("utf-8", errors="ignore")

            # Parse Zip64 extra if needed
            if loc_off == 0xFFFFFFFF or comp_sz == 0xFFFFFFFF:
                extra = cd[idx+46+fn_len:idx+46+fn_len+extra_len]
                e_idx = 0
                while e_idx < len(extra):
                    header_id, data_sz = struct.unpack("<HH", extra[e_idx:e_idx+4])
                    if header_id == 0x0001:
                        z64_data = extra[e_idx+4:e_idx+4+data_sz]
                        loc_off = struct.unpack("<Q", z64_data[16:24])[0]
                        break
                    e_idx += 4 + data_sz

            if fn.endswith("_pan.tiff") and not fn.startswith("__MACOSX"):
                pan_entries.append((fn, comp_m, comp_sz, uncomp_sz, loc_off))
                if len(pan_entries) >= target_aois:
                    break
            idx += 46 + fn_len + extra_len + comm_len
        else:
            break

    log.info(f"Found {len(pan_entries)} panchromatic AOI candidates. Extracting ...")

    total_saved = 0
    for fn, comp_m, comp_sz, uncomp_sz, loc_off in pan_entries:
        aoi_name = Path(fn).stem.replace("_pan", "")
        out_npz = DEST_DIR / f"{aoi_name}.npz"
        if out_npz.exists():
            log.info(f"AOI {aoi_name} already processed. Skipping.")
            total_saved += patches_per_aoi
            continue

        log.info(f"Fetching {fn} ({comp_sz / (1024*1024):.2f} MB compressed) ...")
        try:
            payload = extract_file_from_remote_zip(HR_ZIP_URL, loc_off, comp_m, comp_sz)
            with rasterio.open(io.BytesIO(payload)) as src:
                arr = src.read(1).astype(np.float32)  # [H, W]
                # Normalize 12-bit SPOT radiometry (0 - 4095 or 10000) to [0, 1]
                max_val = np.percentile(arr[arr > 0], 99.8) if (arr > 0).any() else 4095.0
                arr_norm = np.clip(arr / max(max_val, 1000.0), 0.0, 1.0)
                
            H, W = arr_norm.shape
            # Downsample from 1.5m to 2.5m using area pooling
            t_pan_15m = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            target_h = int(H * 1.5 / 2.5)
            target_w = int(W * 1.5 / 2.5)
            t_pan_25m = F.interpolate(t_pan_15m, size=(target_h, target_w), mode="area").squeeze(0)  # [1, H_2.5, W_2.5]
            
            # Extract 128x128 HR patches at 2.5m
            hr_size = 128
            lr_size = 32
            _, H25, W25 = t_pan_25m.shape
            
            step = 96
            n_r = max(1, (H25 - hr_size) // step + 1)
            n_c = max(1, (W25 - hr_size) // step + 1)
            
            aoi_patches = 0
            for r in range(n_r):
                for c in range(n_c):
                    if aoi_patches >= patches_per_aoi:
                        break
                    r_pos = r * step
                    c_pos = c * step
                    p_hr = t_pan_25m[:, r_pos:r_pos+hr_size, c_pos:c_pos+hr_size]  # [1, 128, 128]
                    
                    # Synthesize 4-band LR input [4, 32, 32] at 10m (downsampled 4x from 2.5m)
                    p_lr_1ch = F.interpolate(p_hr.unsqueeze(0), size=(lr_size, lr_size), mode="area").squeeze(0)
                    p_lr = p_lr_1ch.repeat(4, 1, 1)  # 4 bands
                    
                    npz_name = DEST_DIR / f"{aoi_name}_p{aoi_patches:02d}.npz"
                    np.savez_compressed(
                        npz_name,
                        lr=p_lr.numpy().astype(np.float32),
                        hr_pan=p_hr.numpy().astype(np.float32),
                        aoi=aoi_name
                    )
                    aoi_patches += 1
                    total_saved += 1
                    
            log.info(f"Saved {aoi_patches} patches from AOI {aoi_name} ✓")
        except Exception as e:
            log.warning(f"Error processing {fn}: {e}")

    log.info(f"WorldStrat processing complete! Total panchromatic patches saved: {total_saved}")


if __name__ == "__main__":
    fetch_worldstrat_pan_samples(target_aois=8, patches_per_aoi=16)
