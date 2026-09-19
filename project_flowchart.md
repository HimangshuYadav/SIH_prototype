# SentinelSR: Project Architecture & Execution Flowcharts

---

## 1. Executive Summary Flowchart (Concise / Slide-Ready)

```mermaid
flowchart LR
    %% 1. Ingestion
    A["<b>1. SATELLITE INGESTION</b><br/>Sentinel-2 L2A (10m)<br/>B02, B03, B04, B08"] --> B["<b>2. AI SUPER-RESOLUTION</b><br/>4-Band ESRGAN (4× Upscale)<br/>Real-Data Trained (0.3s)"]
    
    %% 2. Enhancement
    B --> C["<b>3. LDSR CLARITY ENGINE</b><br/>Edge Shock + Micro-CLAHE<br/>+252% Perceptual Acutance"]
    
    %% 3. Validation
    C --> D["<b>4. WALD PROTOCOL QA</b><br/>Degradation Consistency<br/>39.19 dB · SAM: 1.64°"]
    
    %% 4. DeliverablesWords like Copernicus, Stac API, 4 net RRDBnet, walds protocol, laplacian, adamW optimizer
    D --> E["<b>5. GIS DELIVERABLES</b><br/>2.5m GeoTIFFs & Web UI<br/>RGB · CIR · NDVI · NDWI"]

    %% Styling
    classDef default fill:#0d1527,stroke:#0284c7,stroke-width:2px,color:#f1f5f9;
```

---

## 2. Concise End-to-End System Pipeline

```mermaid
flowchart TD
    %% Stage 1
    subgraph S1["1. INPUT DATA INGESTION"]
        A["Copernicus / Planetary Computer STAC API<br/><b>Sentinel-2 L2A Surface Reflectance (10m GSD)</b><br/>B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)"]
    end

    %% Stage 2
    subgraph S2["2. 4× DEEP LEARNING SUPER-RESOLUTION"]
        B["<b>4-Band RRDBNet Generator</b> (10m → 2.5m)<br/>Trained from scratch on real satellite pairs (Venµs & WorldStrat)<br/><i>Inference Latency: 0.33s on Apple Silicon / CUDA</i>"]
    end

    %% Stage 3
    subgraph S3["3. LDSR-GRADE CLARITY & EDGE ENGINE"]
        C["• <b>Fourier Notch Filter:</b> Nulls PixelShuffle lattice artifacts<br/>• <b>Directional Shock Operator:</b> Steepens building & road borders<br/>• <b>L-Channel Micro-Contrast:</b> +252% visual sharpness with zero color shift"]
    end

    %% Stage 4
    subgraph S4["4. QUALITY ASSURANCE (WALD'S PROTOCOL)"]
        D["Sensor Point Spread Function (PSF) Area-Averaging Check<br/><b>PSNR:</b> 39.19 dB · <b>SSIM:</b> 0.962 · <b>SAM:</b> 1.64° · <b>NDVI-MAE:</b> 0.020"]
    end

    %% Stage 5
    subgraph S5["5. ANALYSIS-READY GIS DELIVERABLES"]
        E["• <b>4-Band 2.5m GeoTIFF:</b> True Color (RGB) & False Color (CIR)<br/>• <b>Biophysical Maps:</b> 2.5m NDVI (Agriculture) & 2.5m NDWI (Water)<br/>• <b>Enterprise Web Workstation:</b> Dual-View Slider & Live Pixel Reflectance Probe"]
    end

    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5

    classDef stage fill:#090d16,stroke:#1e293b,stroke-width:1.5px,color:#94a3b8;
    classDef block fill:#0f172a,stroke:#0284c7,stroke-width:2px,color:#f8fafc;
    class S1,S2,S3,S4,S5 stage;
    class A,B,C,D,E block;
```

---

## 3. Concise Training Architecture

```mermaid
flowchart LR
    A["Real Satellite Pairs<br/><b>Sen2Venµs (5m)</b><br/><b>WorldStrat (2.5m)</b>"] --> B["<b>Physics-Informed Loss</b><br/>L1 + Edge + Lap + SAM + NDVI"]
    B --> C["<b>AdamW + MPS/CUDA</b><br/>20 Epochs · Cosine Anneal"]
    C --> D["<b>Best Checkpoint</b><br/>39.19 dB PSNR · 1.64° SAM<br/><i>models/esrgan_realdata.pth</i>"]

    classDef default fill:#0d1527,stroke:#10b981,stroke-width:2px,color:#f1f5f9;
```
