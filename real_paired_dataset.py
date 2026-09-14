"""
real_paired_dataset.py
======================
PyTorch Dataset for real paired satellite super-resolution:
1. Sen2Venµs: Real paired Sentinel-2 (10m, B02/B03/B04/B08) and VENµS (5m, B02/B03/B04/B08)
   Native 2x pairing: LR is 32x32 (10m), HR reference is 64x64 (5m).
2. WorldStrat: Real paired Sentinel-2 (10m, 4-band) and SPOT Panchromatic (2.5m, 1-band).
   Target 4x pairing: LR is 32x32 (10m), HR structural reference is 128x128 (2.5m).

Includes strong data augmentations:
- Random H-flip, V-flip
- Random 90/180/270 degree rotations
- Physically-plausible scalar reflectance jitter (illumination variation)
Strict scene-level train/validation split to avoid data leakage.
"""

import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

log = logging.getLogger("real-dataset")


class RealPairedPatchDataset(Dataset):
    """
    Unified Dataset for real paired satellite imagery.
    Yields:
      lr_t: Tensor [4, 32, 32] (float32, [0, 1]) - 10m Sentinel-2 (B02, B03, B04, B08)
      hr_t: Tensor [C, H_hr, W_hr] (float32, [0, 1])
            - Sen2Venµs: [4, 64, 64] (5m VENµS multispectral)
            - WorldStrat: [1, 128, 128] (2.5m panchromatic)
      meta: Dict containing:
            'source': 'sen2venus' or 'worldstrat'
            'is_panchromatic': bool
            'scale': 2.0 or 4.0
            'site': str
    """
    def __init__(
        self,
        real_data_dir: str = "data/real_data",
        split: str = "train",
        val_ratio: float = 0.15,
        augment: bool = True,
        max_patches_per_tile: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(real_data_dir)
        self.split = split
        self.augment = augment and (split == "train")
        self.max_patches = max_patches_per_tile
        self.samples = []
        
        random.seed(seed)
        
        # 1. Index Sen2Venµs data
        self._index_sen2venus(val_ratio=val_ratio, seed=seed)
        
        # 2. Index WorldStrat data
        self._index_worldstrat(val_ratio=val_ratio, seed=seed)
        
        log.info(f"[{split.upper()}] RealPairedPatchDataset loaded {len(self.samples)} total patches "
                 f"(Augmentation={'ON' if self.augment else 'OFF'})")

    def _index_sen2venus(self, val_ratio: float, seed: int):
        s2v_dir = self.data_dir / "sen2venus"
        if not s2v_dir.exists():
            log.warning(f"Sen2Venµs directory {s2v_dir} does not exist yet.")
            return

        # Find all 10m tensor files
        files_10m = sorted(s2v_dir.rglob("*10m_b2b3b4b8.pt"))
        if not files_10m:
            log.warning(f"No Sen2Venµs 10m tensor files found in {s2v_dir}.")
            return

        log.info(f"Indexing Sen2Venµs files across {len(files_10m)} acquisition scenes ...")

        # Group by acquisition scene for scene-level splitting
        scenes = []
        for f10 in files_10m:
            # Matching 05m file name
            f05_name = f10.name.replace("10m_b2b3b4b8.pt", "05m_b2b3b4b8.pt")
            f05 = f10.parent / f05_name
            if f05.exists():
                scenes.append((f10, f05))

        # Deterministic shuffle by scene to avoid spatial data leakage
        rng = random.Random(seed)
        rng.shuffle(scenes)

        n_val = max(1, int(len(scenes) * val_ratio))
        val_scenes = scenes[:n_val]
        train_scenes = scenes[n_val:]

        active_scenes = train_scenes if self.split == "train" else val_scenes

        # Extract patch coordinates
        for f10, f05 in active_scenes:
            site_name = f10.parent.name
            try:
                t10 = torch.load(f10, map_location="cpu", weights_only=True)
                t05 = torch.load(f05, map_location="cpu", weights_only=True)
            except Exception:
                try:
                    t10 = torch.load(f10, map_location="cpu")
                    t05 = torch.load(f05, map_location="cpu")
                except Exception as e:
                    log.warning(f"Could not load pair ({f10.name}): {e}")
                    continue

            num_patches = t10.shape[0]  # shape: [N, 4, 128, 128]
            H10, W10 = t10.shape[2], t10.shape[3]

            # We need 32x32 LR patches from 10m, and corresponding 64x64 HR from 5m (since 10m->5m is 2x)
            lr_size = 32
            hr_size = 64
            scale = 2  # 10m -> 5m

            step_lr = 24
            rows = list(range(0, H10 - lr_size + 1, step_lr))
            cols = list(range(0, W10 - lr_size + 1, step_lr))
            coords = [(r, c) for r in rows for c in cols]

            for patch_idx in range(num_patches):
                p_lr_full = t10[patch_idx].float() / 10000.0
                p_hr_full = t05[patch_idx].float() / 10000.0

                rng.shuffle(coords)
                selected_coords = coords[:self.max_patches]

                for r, c in selected_coords:
                    r_hr = r * scale
                    c_hr = c * scale
                    
                    sub_lr = p_lr_full[:, r:r+lr_size, c:c+lr_size].clamp(0.0, 1.0)
                    sub_hr = p_hr_full[:, r_hr:r_hr+hr_size, c_hr:c_hr+hr_size].clamp(0.0, 1.0)

                    # Quick quality filter: discard nodata / empty patches
                    if sub_lr.mean() < 0.005 or sub_hr.mean() < 0.005:
                        continue

                    self.samples.append({
                        "lr": sub_lr,
                        "hr": sub_hr,
                        "source": "sen2venus",
                        "is_panchromatic": False,
                        "scale": 2.0,
                        "site": site_name,
                    })

    def _index_worldstrat(self, val_ratio: float, seed: int):
        ws_dir = self.data_dir / "worldstrat"
        if not ws_dir.exists():
            return

        ws_files = sorted(ws_dir.glob("*.npz"))
        if not ws_files:
            return

        rng = random.Random(seed)
        shuffled = ws_files.copy()
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_ratio))
        active_files = shuffled[n_val:] if self.split == "train" else shuffled[:n_val]

        for f in active_files:
            try:
                data = np.load(f)
                lr = torch.from_numpy(data["lr"]).float()       # [4, 32, 32]
                hr_pan = torch.from_numpy(data["hr_pan"]).float() # [1, 128, 128]
                self.samples.append({
                    "lr": lr.clamp(0.0, 1.0),
                    "hr": hr_pan.clamp(0.0, 1.0),
                    "source": "worldstrat",
                    "is_panchromatic": True,
                    "scale": 4.0,
                    "site": f.stem,
                })
            except Exception as e:
                log.warning(f"Failed loading WorldStrat sample {f.name}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        lr = item["lr"].clone()
        hr = item["hr"].clone()
        is_pan = item["is_panchromatic"]

        if self.augment:
            # 1. Random Horizontal Flip
            if random.random() > 0.5:
                lr = torch.flip(lr, dims=[2])
                hr = torch.flip(hr, dims=[2])

            # 2. Random Vertical Flip
            if random.random() > 0.5:
                lr = torch.flip(lr, dims=[1])
                hr = torch.flip(hr, dims=[1])

            # 3. Random 90-degree rotations
            k = random.randint(0, 3)
            if k > 0:
                lr = torch.rot90(lr, k, dims=[1, 2])
                hr = torch.rot90(hr, k, dims=[1, 2])

            # 4. Physically-plausible illumination scaling & offset
            # Same scalar factor across all bands preserves spectral angles (SAM)
            scale_jitter = random.uniform(0.96, 1.04)
            offset_jitter = random.uniform(-0.015, 0.015)

            lr = torch.clamp(lr * scale_jitter + offset_jitter, 0.0, 1.0)
            hr = torch.clamp(hr * scale_jitter + offset_jitter, 0.0, 1.0)

        meta = {
            "source": item["source"],
            "is_panchromatic": is_pan,
            "scale": item["scale"],
            "site": item["site"],
        }
        return lr, hr, meta


def collate_real_batch(batch):
    """
    Custom collate function separating multispectral and panchromatic items
    if mixed in a batch, or standard stacking.
    """
    lrs = torch.stack([b[0] for b in batch], dim=0)
    hrs = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    
    # Check if all HR have identical shape
    same_hr_shape = all(h.shape == hrs[0].shape for h in hrs)
    if same_hr_shape:
        hrs_t = torch.stack(hrs, dim=0)
    else:
        hrs_t = hrs  # list of tensors
        
    return lrs, hrs_t, metas
