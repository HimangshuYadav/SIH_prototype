# 🛰️ SAT-SRNet: Sentinel-2 Trustworthy Super-Resolution Platform (4×: 10m → 2.5m)
### Smart India Hackathon (SIH) 2026 | Problem Statement ID: 26142
**Theme**: Space Technology | **Category**: Software | **Team**: Wyrd

---

## 📌 Executive Summary

**SAT-SRNet** is a deep-learning remote-sensing platform designed to super-resolve free, globally available European Space Agency (ESA) **Sentinel-2 L2A multispectral satellite imagery** from its native **10-meter ground sampling distance (GSD)** to **2.5-meter GSD** ($4\times$ spatial enhancement).

Unlike conventional computer vision super-resolution that alters pixel distributions for visual aesthetics, SAT-SRNet enforces **strict physical radiometric calibration** across all 4 primary bands (**B02 Blue, B03 Green, B04 Red, B08 NIR**). This ensures that scientific downstream indices—such as **NDVI** (Normalized Difference Vegetation Index for agriculture) and **NDWI** (Normalized Difference Water Index for flood mapping)—remain mathematically and scientifically accurate.

```
                          10m Sentinel-2 Input Tile
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
           [ Teacher Model ]                  [ Student Model ]
           LDSR-S2 Diffusion                  Our Trained ESRGAN
       (Latent Diffusion, 113M)             (RRDB Network, 6.13M)
                    │                                 │
         High-fidelity prior                 Ultra-fast (~1.2s)
          Iterative Denoising               Single-pass feedforward
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
                      Frequency & Radiometric Guardrails
                    (Fourier Notch + Histogram Matching)
                                     ▼
                    Unified 2.5m Output + Analytics
                    (True Color, CIR, NDVI, NDWI, GeoTIFF)
```

---

## 🚀 Key Innovations

1. **Teacher-Student Knowledge Distillation**:
   - **Teacher Model**: ESA OpenSR **LDSR-S2** (113.6M parameter Latent Diffusion model) providing generative priors and epistemic uncertainty quantification.
   - **Student Model**: Our custom 4-band **ESRGAN** (6.13M parameter Residual-in-Residual Dense Block network) running in **$\sim 1.2$ seconds** ($12\times$ faster than diffusion).
2. **2D Fourier Anti-Checkerboard Notch Filter**:
   - Surgically eliminates the period-4 ($\pm H/4, \pm W/4$) PixelShuffle lattice spikes in the frequency domain, completely removing the screen-door mesh artifact while keeping genuine building edges crisp.
3. **Reference Histogram Matching (`skimage.exposure.match_histograms`)**:
   - Aligns the cumulative distribution function (CDF) of super-resolved bands to the input Sentinel-2 surface reflectance, restoring deep shadow baselines in alleys and bright rooftop highlights without overexposure.
4. **Morphological Rooftop Separation (Shock Filter)**:
   - Kramer-Bruckner shock operator applied to structural edge zones converts blurry Gaussian slopes into clean step boundaries, clearly delineating adjacent houses.
5. **Scientific Wald's Protocol HR Validation**:
   - Replaces naive bicubic comparisons with true degradation-reconstruction validation on ground-truth Sentinel-2 BOA surface reflectance.

---

## 📊 Quantitative Benchmark Results

Evaluated on dense urban benchmark tile `s2_b88f8cd1` (121×164 LR → 484×656 SR @ 2.5m):

| Metric | Bicubic Baseline | LDSR-S2 (Pretrained Teacher) | **Our ESRGAN (Distilled Student)** | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Time** | $< 0.01\text{ s}$ | $15.0\text{ s}$ | **$1.2\text{ s}$ ($12.5\times$ faster)** | Real-time interactive GIS capability |
| **PSNR vs Ground Truth** | $35.17\text{ dB}$ (flat) | $25.76\text{ dB}$ | **$33.75\text{ dB}$ (+7.99 dB over LDSR)** | High reconstruction fidelity |
| **SSIM (Structure)** | $0.8812$ | **$0.9592$** | **$0.9547$** | Near-perfect structural fidelity |
| **SAM (Spectral Angle)** | $0.0^\circ$ (blurred) | $2.745^\circ$ | **$1.645^\circ$** | $< 2.0^\circ$ means zero color distortion |
| **Color Fidelity** | $100\%$ (blurred) | $96.95\%$ | **$98.17\%$** | Physically reliable reflectance |
| **ERGAS Error** | — | $3.280$ | **$1.333$** | Low relative dimensionless synthesis error |
| **Sharpness Gain** | $1.0\times$ | $105.36\times$ | **$123.77\times$** | Laplacian variance over bicubic baseline |

---

## 🌾 National Mission Impact

1. **PMFBY (Pradhan Mantri Fasal Bima Yojana)**:
   - 86% of Indian farmers are smallholders with landholdings $< 2$ hectares. At 10m, individual crop parcel boundaries are blurred across adjacent plots. At 2.5m, parcel-level crop health and vigor gradients are clearly resolved, enabling objective, automated insurance claim settlements.
2. **NDMA Flood & Disaster Management**:
   - Super-resolved 2.5m NDWI maps breached canal networks, inundated village roads, and waterlogged infrastructure within seconds of satellite pass.
3. **PM Gati Shakti & Urban Governance**:
   - Delineates road networks and municipal encroachments at $4\times$ finer spatial granularity without recurring million-dollar aerial drone survey costs.

---

## 🛠️ Repository Architecture

```
sih/
├── backend.py                  # FastAPI server, tiling engines, inference pipelines, GIS export
├── distill_esrgan.py           # Knowledge distillation script (Teacher LDSR -> Student ESRGAN)
├── train_esrgan.py             # Base ESRGAN training script & RRDB architecture definitions
├── requirements.txt            # Python dependencies
├── models/
│   ├── config_10m.yaml         # LDSR-S2 Latent Diffusion model architecture configuration
│   ├── esrgan_sentinel2_best.pth # Best distilled student model checkpoint (PSNR: 30.13 dB vs teacher)
│   └── training_log.csv        # Epoch-by-epoch loss and PSNR logs
├── data/
│   ├── cache/                  # Downloaded raw 10m 4-band float32 GeoTIFF tiles from STAC
│   └── outputs/                # Generated 2.5m GeoTIFFs, PNG previews, NDVI, NDWI, Uncertainty maps
├── frontend/
│   ├── index.html              # Main dashboard frontend (Leaflet.js AOI picker)
│   ├── app.js                  # Frontend logic & comparison slider
│   └── style.css               # Glassmorphic dark theme styles
└── README.md
```

---

## ⚡ Quick Start Guide

### 1. Installation
```bash
# Clone repository
git clone https://github.com/HimangshuYadav/SIH_prototype.git
cd SIH_prototype

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python3 backend.py
```
* The FastAPI server will start on `http://localhost:8000`.
* Open `http://localhost:8000` in any web browser to access the interactive Leaflet map dashboard.

### 3. CLI Inference Example
```bash
# Trigger dual-model comparison on active test scene
curl -s -X POST http://localhost:8000/api/sr \
  -H "Content-Type: application/json" \
  -d '{"tile_id": "s2_b88f8cd1", "model_choice": "both"}' | jq .
```

---

## 🔬 Research Citations

1. **HighRes-net**: Deudon et al. (2020). *HighRes-net: Multi-Frame Super-Resolution for Satellite Imagery*. [arXiv:2002.06460](https://arxiv.org/abs/2002.06460).
2. **OpenSR / LDSR-S2**: Donike et al. (2025). *Trustworthy Super-Resolution of Sentinel-2 with Latent Diffusion*. [IEEE JSTARS](https://doi.org/10.1109/JSTARS.2025.3542220).
3. **Thematic Assessment**: Donike et al. (2026). *Spectral and Thematic Assessment of Super-Resolved Sentinel-2*. [Geomatics](https://doi.org/10.3390/geomatics6050097).
4. **Remote Sensing GAN Survey**: Wang et al. (2023). *Generative Adversarial Networks in Remote Sensing Super-Resolution*. [Remote Sensing](https://doi.org/10.3390/rs15205062).
5. **Cloud Detection**: Jeppesen et al. (2019). *Cloud Detection in Remote Sensing Using Deep Learning*. [Remote Sensing of Environment](https://doi.org/10.1016/j.rse.2019.03.039).

---

## 📜 License & Acknowledgments
Built for **Smart India Hackathon (SIH) 2026**. Satellite imagery provided through ESA Copernicus Open Data and Element84 Earth Search STAC API.
