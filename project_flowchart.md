# SentinelSR: Project Architecture & Execution Flowcharts

This document details the complete end-to-end technical flow of **SentinelSR**—from Sentinel-2 multispectral satellite ingestion, through the deep-learning super-resolution engines, radiometric guardrails, Wald synthesis protocol verification, and downstream GIS products.

---

## 1. End-to-End System Architecture Flowchart

```mermaid
flowchart TD
    subgraph Ingestion["1. SATELLITE DATA INGESTION"]
        A["Planetary Computer / Copernicus STAC API"] --> B["Multi-threaded Band Fetching (ThreadPoolExecutor)"]
        B --> C["4-Band Sentinel-2 L2A BOA Surface Reflectance<br/>B02 (Blue, 490nm), B03 (Green, 560nm), B04 (Red, 665nm), B08 (NIR, 842nm)"]
        C --> D["Dynamic Windowed COG Reading & Local Cache<br/>(Native 10m Ground Sampling Distance)"]
    end

    subgraph CoreEngine["2. AI SUPER-RESOLUTION ENGINES"]
        D --> E{"Model Selector"}
        E -->|"Primary Model (Ours)"| F["4-Band RRDBNet ESRGAN<br/>(6.13M params · 0.33s GPU Latency)"]
        E -->|"Diffusion Baseline"| G["LDSR-S2 Latent Diffusion<br/>(113.6M params · Iterative DDIM)"]
        E -->|"Benchmarking"| H["Side-by-Side Dual Model Evaluation"]
        
        F --> I["4x Sub-Pixel Resolution Synthesis<br/>10m/px → 2.5m/px GSD"]
        G --> I
        H --> I
    end

    subgraph Guardrails["3. RADIOMETRIC & EDGE GUARDRAILS"]
        I --> J["High-Frequency Gradient Shock Filtering & Unsharp Masking"]
        J --> K["Joint Photometric Scaling (Earth Chromaticity Invariant)"]
        K --> L["Epistemic Uncertainty Mapping (Stochastic Sampling Variance)"]
    end

    subgraph Validation["4. ACCURACY ASSESSMENT (WALD'S PROTOCOL)"]
        L --> M["Sensor Point Spread Function (PSF) Area-Averaging Degradation"]
        M --> N["Mathematical Consistency Verification against Sensor Reference"]
        N --> O["Quantitative Metrics:<br/>PSNR: 39.19 dB · SSIM: 0.962<br/>SAM: 1.51° · ERGAS: 10.40"]
    end

    subgraph Downstream["5. DOWNSTREAM GIS & EO PRODUCTS"]
        O --> P["Precision Agriculture: 2.5m NDVI Canopy Map"]
        O --> Q["Hydrology & Disaster: 2.5m NDWI Surface Water Map"]
        O --> R["Analysis-Ready Cloud-Optimized GeoTIFFs<br/>(Preserved EPSG:4326/UTM Affine Geotransform)"]
    end

    subgraph WebWorkstation["6. OPERATIONAL WEB WORKSTATION"]
        R --> S["Interactive Leaflet Map with Live Lat/Lon Telemetry"]
        R --> T["Dual-View Split Slider (10m Original vs 2.5m Enhanced)"]
        R --> U["Real-Time Pixel Reflectance Probe (B02, B03, B04, B08, NDVI)"]
        R --> V["Direct GIS Product Downloads (QGIS/ArcGIS Ingestion)"]
    end
```

---

## 2. Model Training Pipeline Flowchart (From Scratch on Real Satellite Pairs)

```mermaid
flowchart TD
    subgraph DataAcquisition["1. REAL SATELLITE PAIRED DATASETS"]
        D1["Sen2Venµs Dataset<br/>(10m Sentinel-2 + 5m VENµS Ground Truth)"] --> P1["Spatial Co-Registration & Reflectance Calibration"]
        D2["WorldStrat Dataset<br/>(10m Sentinel-2 + 2.5m SPOT Panchromatic GT)"] --> P1
        P1 --> P2["7,645 Training Patches<br/>811 Validation Patches (128x128 px)"]
    end

    subgraph Architecture["2. 4-BAND RRDBNet GENERATOR ARCHITECTURE"]
        P2 --> M1["Input Tensor: 4 Channels [B02, B03, B04, B08] (Float32)"]
        M1 --> M2["First Convolution Layer (64 filters, 3x3 kernel)"]
        M2 --> M3["8 Residual-in-Residual Dense Blocks (RRDB)<br/>(Dense connections across 3 sub-blocks each)"]
        M3 --> M4["Trunk Convolution with Residual Skip Connection"]
        M4 --> M5["Sub-Pixel Convolutional Upsampler (PixelShuffle 4x)"]
        M5 --> M6["Final Reconstruction Conv (4 Channels @ 2.5m GSD)"]
    end

    subgraph MultiLoss["3. PHYSICS-INFORMED MULTI-OBJECTIVE LOSS"]
        M6 --> L1["Pixel L1 Charbonnier Loss (Spatial Fidelity)"]
        M6 --> L2["Sobel High-Frequency Edge Loss (Building Boundaries)"]
        M6 --> L3["Spectral Angle Mapper Loss (SAM Color Invariance)"]
        M6 --> L4["Bio-Physical Loss (NDVI Radiometric Conservation)"]
        
        L1 & L2 & L3 & L4 --> LTotal["Combined Loss Function:<br/>L = L1 + 0.1*Edge + 0.05*SAM + 0.05*NDVI"]
    end

    subgraph Optimization["4. TRAINING HARDWARE & CHECKPOINTING"]
        LTotal --> OPT["AdamW Optimizer (lr=2e-4, Cosine Annealing)"]
        OPT --> HARD["Apple Silicon MPS (Metal Performance Shaders) / CUDA"]
        HARD --> VAL["Epoch Validation on Real 5m VENµS Reference"]
        VAL --> BEST["Peak Checkpoint Saved:<br/>esrgan_sentinel2_realdata_scratch_v1.pth<br/>PSNR: 39.19 dB · SSIM: 0.9624 · SAM: 1.51°"]
    end
```

---

## 3. Real-Time Inference & Verification Execution Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as GIS Specialist / Operator
    participant UI as Frontend Web Workstation
    participant API as FastAPI Backend (backend.py)
    participant STAC as Planetary Computer STAC API
    participant Engine as ESRGAN / LDSR Inference Engine
    participant Wald as Wald Protocol Module
    participant FS as Local Tile Cache & Storage

    User->>UI: Selects Bounding Area (AOI) on Leaflet Map
    UI->>API: POST /api/fetch (bbox, dates, max_cloud)
    API->>STAC: Query Sentinel-2 L2A Collection
    STAC-->>API: Signed Asset URLs (B02, B03, B04, B08)
    API->>API: Multi-threaded COG window read (rasterio)
    API->>FS: Cache raw 10m GeoTIFF & generate PNG preview
    API-->>UI: Return tile_id, dimensions, preview URL
    UI-->>User: Renders 10m overlay & updates status bar

    User->>UI: Clicks "Run Super-Resolution" (Model: ESRGAN)
    UI->>API: POST /api/sr (tile_id, model="esrgan")
    API->>FS: Load 4-band Float32 raster
    API->>Engine: Run RRDBNet feedforward pass (0.33s)
    Engine-->>API: 2.5m Super-Resolved 4-Band Array
    API->>API: Apply Sub-pixel Unsharp Mask & Joint Scaling
    API->>API: Compute 2.5m NDVI & 2.5m NDWI rasters

    API->>Wald: Execute Wald's Protocol Verification
    Wald->>Wald: Area-averaging PSF degradation to 10m
    Wald->>Wald: Calculate PSNR, SSIM, SAM, ERGAS vs input
    Wald-->>API: Metric Results (PSNR: 39.19 dB, SSIM: 0.962, SAM: 1.51°)

    API->>FS: Save 2.5m Georeferenced GeoTIFF (affine: a/4, e/4)
    API->>FS: Generate True Color, CIR, NDVI, NDWI PNG previews
    API-->>UI: Return 2.5m previews, metrics & download endpoints
    UI-->>User: Displays Dual-View Split Slider & live Pixel Probe
```

---

## 4. Downstream Earth Observation & GIS Export Pipeline

```mermaid
flowchart LR
    subgraph InputData["Inference Output"]
        SR["4-Band 2.5m Reflectance Tensor<br/>[B02, B03, B04, B08]"]
    end

    subgraph AnalyticalPipelines["Domain Analytics Engines"]
        SR --> AGRI["Precision Agriculture Engine"]
        SR --> HYDRO["Hydrological & Disaster Engine"]
        SR --> UNC["Epistemic Reliability Engine"]
        SR --> GIS["Cartographic Export Engine"]
    end

    subgraph MathematicalFormulas["Processing Operations"]
        AGRI --> F_NDVI["NDVI = (B08 - B04) / (B08 + B04)<br/>Dense Canopy / Moderate / Bare Soil Pct"]
        HYDRO --> F_NDWI["NDWI = (B03 - B08) / (B03 + B08)<br/>Surface Water Inundation Delineation"]
        UNC --> F_UNC["Stochastic Multi-pass Variance Mapping<br/>Per-pixel Confidence Score [0.0 - 1.0]"]
        GIS --> F_AFFINE["Affine Transformation Update:<br/>Scale: a=a/4, e=e/4<br/>CRS: WGS 84 / UTM Zone intact"]
    end

    subgraph Deliverables["Analysis-Ready Deliverables"]
        F_NDVI --> OUT_NDVI["2.5m NDVI GeoTIFF & Heatmap"]
        F_NDWI --> OUT_NDWI["2.5m NDWI GeoTIFF & Water Mask"]
        F_UNC --> OUT_UNC["2.5m Epistemic Variance GeoTIFF"]
        F_AFFINE --> OUT_SR["4-Band 2.5m Reflectance GeoTIFF<br/>(Direct QGIS / ArcGIS Ingestion)"]
    end
```
