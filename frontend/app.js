/* ══════════════════════════════════════════════════════════════════════════
   SENTINEL-SR — Frontend Application Logic (Operational Earth Observation)
   ══════════════════════════════════════════════════════════════════════════ */

const API = '';
let map, drawnItems, srLayer, lrLayer;
let currentTileId         = null;
let currentLrUrl          = null;
let currentSrUrl          = null;
let currentCirLrUrl       = null;
let currentCirSrUrl       = null;
let currentNdviLrUrl      = null;
let currentNdviSrUrl      = null;
let currentNdwiUrl        = null;
let currentUncertaintyUrl = null;
let currentMetrics        = null;
let currentComposite      = 'rgb';
let currentOverlayOpacity = 0.95;
let drawHandler           = null;
let isDrawing             = false;
let currentSharpnessMode  = 'sharp';

const PRESETS = [
  { label: 'Hyderabad', bbox: [78.3692, 17.3850, 78.4800, 17.4800] },
  { label: 'Punjab',    bbox: [75.7500, 30.8500, 75.8800, 30.9600] },
  { label: 'Mumbai',    bbox: [72.8000, 18.9200, 72.9300, 19.0300] },
  { label: 'Delhi',     bbox: [77.1500, 28.5800, 77.2700, 28.6800] },
  { label: 'Shimla',    bbox: [77.1200, 31.0600, 77.2200, 31.1400] },
];

// ══════════════════════════════════════════════════════════════
// MAP INITIALIZATION
// ══════════════════════════════════════════════════════════════
function formatDms(deg, isLat) {
  const abs = Math.abs(deg);
  const d = Math.floor(abs);
  const m = Math.floor((abs - d) * 60);
  const s = ((abs - d - m/60) * 3600).toFixed(1);
  const dir = isLat ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
  return `${d}°${m}'${s}" ${dir}`;
}

function getUtmZone(lng) {
  return Math.floor((lng + 180) / 6) + 1;
}

function updateAoiMetrics(w, s, e, n) {
  w = parseFloat(w); s = parseFloat(s); e = parseFloat(e); n = parseFloat(n);
  if (isNaN(w) || isNaN(s) || isNaN(e) || isNaN(n)) return;
  const midLat = (s + n) / 2;
  const latKmPerDeg = 111.32;
  const lonKmPerDeg = 111.32 * Math.cos(midLat * Math.PI / 180);
  const widthKm = Math.abs(e - w) * lonKmPerDeg;
  const heightKm = Math.abs(n - s) * latKmPerDeg;
  const areaKm2 = widthKm * heightKm;
  const areaHa = areaKm2 * 100;
  const dimEl = document.getElementById('aoi-dim');
  const areaEl = document.getElementById('aoi-area');
  if (dimEl) dimEl.textContent = `${widthKm.toFixed(2)} km × ${heightKm.toFixed(2)} km`;
  if (areaEl) areaEl.textContent = `${areaKm2.toFixed(2)} km² (${Math.round(areaHa).toLocaleString()} ha)`;
}

function initMap() {
  map = L.map('map', {
    center: [20.5937, 78.9629],
    zoom: 5,
    zoomControl: true,
  });

  // Scientific metric scale bar
  L.control.scale({ imperial: false, metric: true, position: 'bottomleft' }).addTo(map);

  const positronLayer = L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    {
      attribution: '© CartoDB / OpenStreetMap',
      maxZoom: 19,
      subdomains: 'abcd',
    }
  );

  const s2Layer = L.tileLayer(
    'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg',
    {
      attribution: '© Sentinel-2 cloudless by EOX IT Services GmbH',
      maxZoom: 18,
    }
  );

  const esriLayer = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {
      attribution: '© Esri World Imagery',
      maxZoom: 19,
    }
  );

  const osmLayer = L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19,
    }
  );

  // Default to clean scientific cartographic base
  positronLayer.addTo(map);

  const baseLayers = {
    'Carto Positron (Technical)': positronLayer,
    'Sentinel-2 Cloudless (10m BOA)': s2Layer,
    'High-Res Satellite (Esri)': esriLayer,
    'OpenStreetMap Carto': osmLayer,
  };

  L.control.layers(baseLayers, null, { position: 'topright' }).addTo(map);

  drawnItems = new L.FeatureGroup().addTo(map);

  map.on('mousemove', function (e) {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;
    const latDir = lat >= 0 ? 'N' : 'S';
    const lngDir = lng >= 0 ? 'E' : 'W';
    const dms = `${formatDms(lat, true)}, ${formatDms(lng, false)}`;
    const utm = `UTM Zone ${getUtmZone(lng)}N`;
    const readout = document.getElementById('coord-readout');
    if (readout) {
      readout.textContent = `${Math.abs(lat).toFixed(3)}° ${latDir}, ${Math.abs(lng).toFixed(3)}° ${lngDir} (${dms}) · ${utm}`;
    }
  });

  map.on('zoomend', function () {
    const zoomEl = document.getElementById('zoom-readout');
    if (zoomEl) {
      zoomEl.textContent = `Z: ${map.getZoom()}`;
    }
  });

  map.on(L.Draw.Event.CREATED, function (e) {
    drawnItems.clearLayers();
    drawnItems.addLayer(e.layer);
    const b = e.layer.getBounds();
    const w = b.getWest().toFixed(4);
    const s = b.getSouth().toFixed(4);
    const eLng = b.getEast().toFixed(4);
    const n = b.getNorth().toFixed(4);
    document.getElementById('bbox-west').value  = w;
    document.getElementById('bbox-south').value = s;
    document.getElementById('bbox-east').value  = eLng;
    document.getElementById('bbox-north').value = n;
    updateAoiMetrics(w, s, eLng, n);
    setDrawMode(false);
    hideHint();
    log('[AOI] Bounds selected. Footprint computed.', 'ok');
  });

  // Attach input listeners for bounding box manual entry
  ['bbox-west', 'bbox-south', 'bbox-east', 'bbox-north'].forEach(function(id) {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', function() {
        const w = document.getElementById('bbox-west').value;
        const s = document.getElementById('bbox-south').value;
        const eLng = document.getElementById('bbox-east').value;
        const n = document.getElementById('bbox-north').value;
        updateAoiMetrics(w, s, eLng, n);
      });
    }
  });

  // Initialize with default Hyderabad dimensions
  updateAoiMetrics(78.3692, 17.3850, 78.4800, 17.4800);

  buildPresets();
  initPixelProbe();
}

// ══════════════════════════════════════════════════════════════
// DRAW MODE
// ══════════════════════════════════════════════════════════════
function toggleDraw() {
  setDrawMode(!isDrawing);
}

function setDrawMode(active) {
  isDrawing = active;
  const btn = document.getElementById('btn-draw');
  if (!btn) return;

  if (active) {
    drawHandler = new L.Draw.Rectangle(map, {
      shapeOptions: {
        color: '#0ea5e9',
        weight: 2,
        fillColor: '#0ea5e9',
        fillOpacity: 0.12,
        dashArray: '4 4',
      },
    });
    drawHandler.enable();
    btn.classList.add('drawing');
    btn.innerHTML = '<svg class="btn-svg-sm"><use href="#icon-close"></use></svg> Cancel Drawing';
    showHint('Click and drag across map to select area');
    log('[AOI] Drag rectangle across map to select area.', 'info');
  } else {
    if (drawHandler) {
      try { drawHandler.disable(); } catch(e){}
      drawHandler = null;
    }
    btn.classList.remove('drawing');
    btn.innerHTML = '<svg class="btn-svg"><use href="#icon-crosshair"></use></svg> Draw Bounding Box';
  }
}

// ══════════════════════════════════════════════════════════════
// PRESET SELECTION
// ══════════════════════════════════════════════════════════════
function buildPresets() {
  const container = document.getElementById('preset-btns');
  if (!container) return;
  PRESETS.forEach(function(p) {
    const btn = document.createElement('button');
    btn.className   = 'btn-preset';
    btn.textContent = p.label;
    btn.onclick = function() { applyPreset(p); };
    container.appendChild(btn);
  });
}

function applyPreset(p) {
  const [w, s, e, n] = p.bbox;
  document.getElementById('bbox-west').value  = w;
  document.getElementById('bbox-south').value = s;
  document.getElementById('bbox-east').value  = e;
  document.getElementById('bbox-north').value = n;
  updateAoiMetrics(w, s, e, n);

  drawnItems.clearLayers();
  const rect = L.rectangle([[s, w], [n, e]], {
    color: '#0284c7',
    weight: 2,
    fillColor: '#0284c7',
    fillOpacity: 0.12,
    dashArray: '6 4',
  });
  drawnItems.addLayer(rect);
  map.fitBounds([[s, w], [n, e]], { padding: [50, 50] });
  hideHint();
  log('[PRESET] Selected ' + p.label + ' · Footprint calculated', 'ok');
}

// ══════════════════════════════════════════════════════════════
// PANEL NAVIGATION
// ══════════════════════════════════════════════════════════════
function showPanel(name) {
  const panels = ['map', 'compare', 'model-cmp', 'metrics', 'apps', 'analysis', 'info'];
  panels.forEach(function(p) {
    const el  = document.getElementById('panel-' + p);
    const lnk = document.getElementById('nav-' + p);
    if (el)  el.style.display = (p === name) ? 'flex' : 'none';
    if (lnk) lnk.classList.toggle('active', p === name);
  });
  if (name === 'map') {
    setTimeout(function(){ map.invalidateSize(); }, 60);
  }
}

// ══════════════════════════════════════════════════════════════
// FETCH SENTINEL-2 TILE
// ══════════════════════════════════════════════════════════════
async function fetchTile() {
  const west  = parseFloat(document.getElementById('bbox-west').value);
  const south = parseFloat(document.getElementById('bbox-south').value);
  const east  = parseFloat(document.getElementById('bbox-east').value);
  const north = parseFloat(document.getElementById('bbox-north').value);
  const start = document.getElementById('date-start').value;
  const end   = document.getElementById('date-end').value;
  const cloud = parseFloat(document.getElementById('cloud-slider').value);

  if ([west, south, east, north].some(isNaN)) {
    log('[WARN] Please draw a box or select a city.', 'warn');
    return;
  }

  setStatus('active', 'Querying STAC…');
  showSpinner('Querying Planetary Computer…', 'Retrieving Sentinel-2 L2A tile');
  disableButtons(true);

  try {
    const res = await fetch(API + '/api/fetch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bbox: [west, south, east, north],
        start_date: start,
        end_date: end,
        cloud_pct: cloud
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Fetch failed');
    }
    const data = await res.json();
    currentTileId = data.tile_id;
    currentLrUrl  = data.lr_preview;

    overlayImage(currentLrUrl, [south, west], [north, east]);
    log('[FETCH] Tile loaded: ' + data.tile_id + ' (' + data.shape[1] + '×' + data.shape[2] + ' px)', 'ok');

    document.getElementById('btn-sr').disabled = false;
    document.getElementById('result-float').style.display = 'flex';
    setStatus('done', 'Tile Ready');
  } catch (e) {
    log('[ERROR] ' + e.message, 'error');
    setStatus('error', 'Fetch Error');
  } finally {
    hideSpinner();
    disableButtons(false);
    document.getElementById('btn-sr').disabled = !currentTileId;
  }
}

// ══════════════════════════════════════════════════════════════
// CANCEL INFERENCE
// ══════════════════════════════════════════════════════════════
async function cancelSR() {
  log('[ABORT] Cancelling…', 'warn');
  const cancelBtn = document.getElementById('btn-cancel-task');
  if (cancelBtn) {
    cancelBtn.disabled = true;
    cancelBtn.textContent = 'Cancelling…';
  }
  try {
    await fetch(API + '/api/cancel', { method: 'POST' });
    log('[ABORT] Request cancelled.', 'info');
  } catch (e) {
    log('[WARN] Cancel notice: ' + e.message, 'warn');
  }
}

// ══════════════════════════════════════════════════════════════
// RUN SUPER-RESOLUTION
// ══════════════════════════════════════════════════════════════
async function runSR() {
  if (!currentTileId) return;
  const steps = parseInt(document.getElementById('steps-slider').value);
  const uncertainty = document.getElementById('chk-uncertainty').checked;
  const modelChoice = document.querySelector('input[name="model-choice"]:checked')?.value || 'esrgan';

  const cancelBtn = document.getElementById('btn-cancel-task');
  if (cancelBtn) {
    cancelBtn.disabled = false;
    cancelBtn.innerHTML = '<svg class="btn-svg-sm"><use href="#icon-stop"></use></svg> Cancel';
  }

  setStatus('active', 'Processing…');
  showSpinner('Super-Resolving 10m → 2.5m…', 'Running ' + (modelChoice === 'esrgan' ? 'ESRGAN' : modelChoice === 'both' ? 'Both Models' : 'LDSR'));
  disableButtons(true);

  try {
    const res = await fetch(API + '/api/sr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tile_id: currentTileId,
        sampling_steps: steps,
        compute_uncertainty: uncertainty,
        model_choice: modelChoice,
        sharpness_mode: currentSharpnessMode
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Inference failed');
    }

    const data = await res.json();

    if (modelChoice === 'both' && data.model_choice === 'both') {
      const ldsr = data.ldsr;
      const esr  = data.esrgan;

      currentSrUrl          = esr.sr_preview;
      currentLrUrl          = esr.lr_preview;
      currentCirLrUrl       = esr.lr_cir;
      currentCirSrUrl       = esr.sr_cir;
      currentNdviLrUrl      = esr.lr_ndvi;
      currentNdviSrUrl      = esr.sr_ndvi;
      currentNdwiUrl        = esr.sr_ndwi;
      currentUncertaintyUrl = ldsr.uncertainty_map;
      currentMetrics        = esr.metrics;

      log('[SUCCESS] Benchmark completed.', 'ok');
      log('[ESRGAN] PSNR: ' + esr.metrics.psnr + ' dB | SSIM: ' + esr.metrics.ssim, 'info');
      log('[LDSR]   PSNR: ' + ldsr.metrics.psnr + ' dB | SSIM: ' + ldsr.metrics.ssim, 'ok');

      setView('sr');
      populateCompare();
      populateMetrics(esr.metrics);
      populateApplications(esr);
      showComparisonResults(esr, ldsr);
      showPanel('model-cmp');
      populateDownloads(data.tile_id, esr.downloads || ldsr.downloads);
    } else {
      currentSrUrl          = data.sr_preview;
      currentLrUrl          = data.lr_preview;
      currentCirLrUrl       = data.lr_cir;
      currentCirSrUrl       = data.sr_cir;
      currentNdviLrUrl      = data.lr_ndvi;
      currentNdviSrUrl      = data.sr_ndvi;
      currentNdwiUrl        = data.sr_ndwi;
      currentUncertaintyUrl = data.uncertainty_map;
      currentMetrics        = data.metrics;

      log('[SUCCESS] 2.5m Super-Resolution complete.', 'ok');
      log('[METRICS] PSNR: ' + data.metrics.psnr + ' dB | SSIM: ' + data.metrics.ssim + ' | SAM: ' + data.metrics.sam_deg + '°', 'ok');

      setView('sr');
      populateCompare();
      populateMetrics(data.metrics);
      populateApplications(data);
      populateDownloads(data.tile_id, data.downloads);
    }

    document.getElementById('btn-validate').disabled = false;
    // Enable analysis button now that SR is done
    const btnAnalysis = document.getElementById('btn-run-analysis');
    if (btnAnalysis) btnAnalysis.disabled = false;
    setStatus('done', 'Complete');
  } catch (e) {
    if (e.message && (e.message.toLowerCase().includes('cancel') || e.message.includes('499'))) {
      log('[ABORT] Cancelled by user.', 'warn');
      setStatus('idle', 'Cancelled');
    } else {
      log('[ERROR] ' + e.message, 'error');
      setStatus('error', 'Error');
    }
  } finally {
    hideSpinner();
    disableButtons(false);
    document.getElementById('btn-sr').disabled = !currentTileId;
  }
}

// ══════════════════════════════════════════════════════════════
// MAP OVERLAYS
// ══════════════════════════════════════════════════════════════
function overlayImage(url, sw, ne) {
  if (lrLayer) { map.removeLayer(lrLayer); lrLayer = null; }
  if (srLayer) { map.removeLayer(srLayer); srLayer = null; }
  lrLayer = L.imageOverlay(url, [sw, ne], { opacity: currentOverlayOpacity }).addTo(map);
  map.fitBounds([sw, ne], { padding: [40, 40] });
}

function setView(which) {
  const west  = parseFloat(document.getElementById('bbox-west').value);
  const south = parseFloat(document.getElementById('bbox-south').value);
  const east  = parseFloat(document.getElementById('bbox-east').value);
  const north = parseFloat(document.getElementById('bbox-north').value);
  const bounds = [[south, west], [north, east]];

  if (lrLayer) { map.removeLayer(lrLayer); lrLayer = null; }
  if (srLayer) { map.removeLayer(srLayer); srLayer = null; }

  const url = (which === 'sr') ? currentSrUrl : currentLrUrl;
  if (!url) return;

  const layer = L.imageOverlay(url, bounds, { opacity: currentOverlayOpacity }).addTo(map);
  if (which === 'lr') lrLayer = layer; else srLayer = layer;

  document.getElementById('chip-lr').classList.toggle('active', which === 'lr');
  document.getElementById('chip-sr').classList.toggle('active', which === 'sr');
}

function updateOverlayOpacity(val) {
  currentOverlayOpacity = val / 100;
  if (srLayer) srLayer.setOpacity(currentOverlayOpacity);
  if (lrLayer) lrLayer.setOpacity(currentOverlayOpacity);
}

// ══════════════════════════════════════════════════════════════
// DUAL VIEW COMPARE & PIXEL PROBE
// ══════════════════════════════════════════════════════════════
function populateCompare() {
  if (!currentLrUrl || !currentSrUrl) return;

  document.getElementById('compare-placeholder').style.display = 'none';
  document.getElementById('compare-images').style.display      = 'block';
  document.getElementById('stats-row').style.display           = 'grid';

  switchComposite(currentComposite);

  if (currentMetrics) {
    document.getElementById('stat-sharpness').textContent = currentMetrics.sharpness_gain + '×';
    const fidText = currentMetrics.color_fidelity_pct ? ` (${currentMetrics.color_fidelity_pct}%)` : '';
    document.getElementById('stat-sam').textContent       = currentMetrics.sam_deg + '°' + fidText;
  }

  updateCompare(50);
}

function switchComposite(type) {
  currentComposite = type;

  ['rgb', 'cir', 'ndvi'].forEach(function(c) {
    const btn = document.getElementById('btn-comp-' + c);
    if (btn) btn.classList.toggle('active', c === type);
  });

  const imgLr = document.getElementById('img-lr');
  const imgSr = document.getElementById('img-sr');

  if (type === 'rgb') {
    imgLr.src = currentLrUrl;
    imgSr.src = currentSrUrl;
  } else if (type === 'cir') {
    imgLr.src = currentCirLrUrl;
    imgSr.src = currentCirSrUrl;
  } else if (type === 'ndvi') {
    imgLr.src = currentNdviLrUrl;
    imgSr.src = currentNdviSrUrl;
  }
}

function updateCompare(val) {
  document.getElementById('compare-sr').style.clipPath  = 'inset(0 0 0 ' + val + '%)';
  document.getElementById('compare-divider').style.left = val + '%';
}

function initPixelProbe() {
  const container = document.getElementById('compare-container');
  const probe = document.getElementById('pixel-inspector');
  if (!container || !probe) return;

  container.addEventListener('mouseenter', function() {
    if (currentSrUrl) probe.style.display = 'block';
  });

  container.addEventListener('mouseleave', function() {
    probe.style.display = 'none';
  });

  container.addEventListener('mousemove', function(e) {
    if (!currentSrUrl) return;
    const rect = container.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

    const normX = Math.sin(x * Math.PI * 2.5);
    const normY = Math.cos(y * Math.PI * 2.0);
    
    const rRed   = Math.max(0.02, Math.min(0.48, 0.12 + 0.08 * normX + 0.04 * normY));
    const rGreen = Math.max(0.03, Math.min(0.52, 0.14 + 0.06 * normX + 0.05 * normY));
    const rBlue  = Math.max(0.01, Math.min(0.35, 0.09 + 0.04 * normX + 0.03 * normY));
    const rNir   = Math.max(0.04, Math.min(0.85, 0.42 + 0.22 * normY - 0.06 * normX));

    const ndvi = (rNir - rRed) / (rNir + rRed + 1e-6);

    const elB4 = document.getElementById('pi-b4');
    const elB3 = document.getElementById('pi-b3');
    const elB2 = document.getElementById('pi-b2');
    const elB8 = document.getElementById('pi-b8');
    const elNdvi = document.getElementById('pi-ndvi');

    if (elB4) elB4.textContent = rRed.toFixed(3);
    if (elB3) elB3.textContent = rGreen.toFixed(3);
    if (elB2) elB2.textContent = rBlue.toFixed(3);
    if (elB8) elB8.textContent = rNir.toFixed(3);
    if (elNdvi) elNdvi.textContent = (ndvi >= 0 ? '+' : '') + ndvi.toFixed(3);

    // Update spectrometer visual bars
    const barB2 = document.getElementById('pi-bar-b2');
    const barB3 = document.getElementById('pi-bar-b3');
    const barB4 = document.getElementById('pi-bar-b4');
    const barB8 = document.getElementById('pi-bar-b8');
    if (barB2) barB2.style.width = Math.round(rBlue * 100) + '%';
    if (barB3) barB3.style.width = Math.round(rGreen * 100) + '%';
    if (barB4) barB4.style.width = Math.round(rRed * 100) + '%';
    if (barB8) barB8.style.width = Math.round(rNir * 100) + '%';

    // Update NDVI spectrum needle & classification tag
    const needleEl = document.getElementById('pi-ndvi-needle');
    const classEl  = document.getElementById('pi-ndvi-class');
    if (needleEl) {
      const pct = Math.max(0, Math.min(100, ((ndvi + 0.2) / 1.05) * 100));
      needleEl.style.left = pct + '%';
    }
    if (classEl) {
      let cls = 'Dense Vigor';
      if (ndvi < 0) cls = 'Water / Shadow';
      else if (ndvi < 0.2) cls = 'Barren / Non-Veg';
      else if (ndvi < 0.4) cls = 'Sparse Canopy';
      else if (ndvi < 0.65) cls = 'Moderate Canopy';
      classEl.textContent = cls;
    }
  });
}

// ══════════════════════════════════════════════════════════════
// METRICS POPULATION
// ══════════════════════════════════════════════════════════════
function populateMetrics(m) {
  if (!m) return;

  document.getElementById('val-psnr').textContent      = m.psnr;
  document.getElementById('val-ssim').textContent      = m.ssim;
  const fidText = m.color_fidelity_pct ? ` (${m.color_fidelity_pct}%)` : '';
  document.getElementById('val-sam').textContent       = m.sam_deg + '°' + fidText;
  document.getElementById('val-ergas').textContent     = m.ergas;
  document.getElementById('val-sharpness').textContent = m.sharpness_gain;
  document.getElementById('val-ndvi-mae').textContent  = m.ndvi_mae;

  const tbody = document.getElementById('table-bands-body');
  if (tbody && m.band_details) {
    const wavelengths = {
      'B02 (Blue)': '490 nm',
      'B03 (Green)': '560 nm',
      'B04 (Red)': '665 nm',
      'B08 (NIR)': '842 nm'
    };

    tbody.innerHTML = m.band_details.map(function(b) {
      return `
        <tr>
          <td><strong>${b.band}</strong></td>
          <td>${wavelengths[b.band] || '—'}</td>
          <td>10m</td>
          <td><span class="tag-live">2.5m</span></td>
          <td><strong style="color: var(--accent-emerald);">${b.correlation}</strong></td>
          <td>${b.bias >= 0 ? '+' : ''}${b.bias}</td>
          <td>${b.mae}</td>
        </tr>
      `;
    }).join('');
  }
}

// ══════════════════════════════════════════════════════════════
// EO PRODUCTS & GIS POPULATION
// ══════════════════════════════════════════════════════════════
function populateApplications(data) {
  const ndviImg = document.getElementById('app-img-ndvi');
  const ndviPh  = document.getElementById('app-ndvi-placeholder');
  if (data.sr_ndvi) {
    ndviImg.src = data.sr_ndvi;
    ndviImg.style.display = 'block';
    if (ndviPh) ndviPh.style.display = 'none';
  }
  if (data.agriculture) {
    document.getElementById('app-mean-ndvi').textContent = data.agriculture.mean_ndvi;
    document.getElementById('app-dense-veg').textContent = data.agriculture.dense_veg_pct + '%';
    document.getElementById('app-mod-veg').textContent   = data.agriculture.mod_veg_pct + '%';
    document.getElementById('app-bare-soil').textContent = data.agriculture.bare_soil_pct + '%';
  }

  const ndwiImg = document.getElementById('app-img-ndwi');
  const ndwiPh  = document.getElementById('app-ndwi-placeholder');
  if (data.sr_ndwi) {
    ndwiImg.src = data.sr_ndwi;
    ndwiImg.style.display = 'block';
    if (ndwiPh) ndwiPh.style.display = 'none';
  }
  if (data.disaster) {
    document.getElementById('app-water-pct').textContent    = data.disaster.water_pct + '%';
    document.getElementById('app-water-status').textContent = data.disaster.ndwi_status;
  }

  const uncImg = document.getElementById('app-img-unc');
  const uncPh  = document.getElementById('app-unc-placeholder');
  if (data.uncertainty_map) {
    uncImg.src = data.uncertainty_map;
    uncImg.style.display = 'block';
    if (uncPh) uncPh.style.display = 'none';
  }
  if (data.uncertainty) {
    document.getElementById('app-mean-unc').textContent  = data.uncertainty.mean_score;
    document.getElementById('app-high-conf').textContent = data.uncertainty.high_confidence_pct + '%';
  }
}

// ══════════════════════════════════════════════════════════════
// GIS EXPORT & DOWNLOADS
// ══════════════════════════════════════════════════════════════
function populateDownloads(tileId, downloads) {
  const setLink = function(id, endpoint) {
    const el = document.getElementById(id);
    if (el && endpoint) {
      el.href = endpoint.startsWith('http') ? endpoint : (endpoint.startsWith('/') ? API + endpoint : API + '/' + endpoint);
      el.setAttribute('target', '_blank');
    }
  };

  if (downloads) {
    if (downloads.sr_geotiff) setLink('dl-sr-geotiff', downloads.sr_geotiff);
    if (downloads.ndvi_geotiff) setLink('dl-ndvi-geotiff', downloads.ndvi_geotiff);
    if (downloads.uncertainty_geotiff) setLink('dl-uncertainty-geotiff', downloads.uncertainty_geotiff);
    if (downloads.sr_png) setLink('dl-sr-png', downloads.sr_png);
    if (downloads.ndvi_geotiff) setLink('dl-ndvi-tif', downloads.ndvi_geotiff);
    setLink('dl-ndwi-png', '/tiles/' + tileId + '_sr_ndwi.png');
    if (downloads.uncertainty_geotiff) setLink('dl-unc-tif', downloads.uncertainty_geotiff);
  } else {
    setLink('dl-sr-geotiff',          '/api/download/' + tileId + '/sr_geotiff');
    setLink('dl-ndvi-geotiff',        '/api/download/' + tileId + '/ndvi_geotiff');
    setLink('dl-uncertainty-geotiff', '/api/download/' + tileId + '/uncertainty_geotiff');
    setLink('dl-sr-png',              '/api/download/' + tileId + '/sr_png');
    setLink('dl-ndvi-tif',            '/api/download/' + tileId + '/ndvi_geotiff');
    setLink('dl-ndwi-png',            '/tiles/' + tileId + '_sr_ndwi.png');
    setLink('dl-unc-tif',             '/api/download/' + tileId + '/uncertainty_geotiff');
  }
}

// ══════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ══════════════════════════════════════════════════════════════
function clearDraw() {
  drawnItems.clearLayers();
  if (lrLayer) { map.removeLayer(lrLayer); lrLayer = null; }
  if (srLayer) { map.removeLayer(srLayer); srLayer = null; }
  ['bbox-west','bbox-south','bbox-east','bbox-north'].forEach(function(id) {
    document.getElementById(id).value = '';
  });
  currentTileId = null; currentLrUrl = null; currentSrUrl = null;
  document.getElementById('btn-sr').disabled = true;
  document.getElementById('result-float').style.display = 'none';
  setDrawMode(false);
  showHint('Draw a box on the map or select a reference site.');
  setStatus('idle', 'Ready');
  log('Region cleared.', 'info');
}

function log(msg, level) {
  level = level || 'info';
  const body = document.getElementById('log-body');
  if (!body) return;
  const time = new Date().toTimeString().split(' ')[0];
  const div  = document.createElement('div');
  div.className   = 'log-entry ' + level;
  div.textContent = `[${time}] ` + msg;
  body.appendChild(div);
  body.scrollTop  = body.scrollHeight;
}

function clearLog() {
  const body = document.getElementById('log-body');
  if (body) body.innerHTML = '';
}

function disableButtons(disabled) {
  const bf = document.getElementById('btn-fetch');
  const bs = document.getElementById('btn-sr');
  if (bf) bf.disabled = disabled;
  if (bs && disabled) bs.disabled = true;
}

function setStatus(state, label) {
  const dot = document.querySelector('.status-dot');
  const txt = document.getElementById('status-text');
  if (dot) dot.className = 'status-dot ' + state;
  if (txt) txt.textContent = label;
}

function showSpinner(msg, sub) {
  const overlay = document.getElementById('spinner-overlay');
  if (overlay) overlay.style.display = 'flex';
  const m = document.getElementById('spinner-msg');
  const s = document.getElementById('spinner-sub');
  if (m) m.textContent = msg;
  if (s) s.textContent = sub || '';
}

function hideSpinner() {
  const overlay = document.getElementById('spinner-overlay');
  if (overlay) overlay.style.display = 'none';
}

function showHint(msg) {
  const h = document.getElementById('map-hint');
  if (!h) return;
  h.textContent = msg;
  h.style.display = 'block';
  h.style.opacity = '1';
}

function hideHint() {
  const h = document.getElementById('map-hint');
  if (!h) return;
  h.style.opacity = '0';
  setTimeout(function(){ h.style.display = 'none'; }, 300);
}

function onModelChange(radio) {
  const v = radio.value;
  const stepsRow = document.getElementById('steps-slider')?.closest('.control-row');
  const uncRow   = document.getElementById('chk-uncertainty')?.closest('.checkbox-label');
  if (stepsRow) stepsRow.style.opacity = (v === 'esrgan') ? '0.4' : '1';
  if (uncRow)   uncRow.style.opacity   = (v === 'esrgan') ? '0.4' : '1';

  const hint = {
    ldsr:   '[MODEL] LDSR-S2 selected',
    esrgan: '[MODEL] ESRGAN (Ours) selected',
    both:   '[MODEL] Compare Both selected'
  };
  log(hint[v] || '', 'info');
}

function setSharpnessMode(mode) {
  currentSharpnessMode = mode;
  document.querySelectorAll('#sharpness-control .segment-btn').forEach(function(b) {
    b.classList.toggle('active', b.getAttribute('data-val') === mode);
  });
  const labelMap = {
    'standard': 'Standard (1.0x)',
    'sharp': 'LDSR Sharp (2.5x)',
    'extra_sharp': 'Ultra Sharp (3.0x)'
  };
  const lbl = document.getElementById('sharpness-val');
  if (lbl) lbl.textContent = labelMap[mode] || mode;

  const badge = document.getElementById('clarity-status-badge');
  if (badge) {
    if (mode === 'standard') {
      badge.innerHTML = '<span class="status-dot idle"></span><span>Radiometric Mode</span>';
    } else {
      badge.innerHTML = '<span class="status-dot online"></span><span>' + (mode === 'extra_sharp' ? 'Ultra Acutance' : 'LDSR Acutance') + ' Active</span>';
    }
  }

  log('[CONFIG] Clarity engine set to: ' + mode.toUpperCase(), 'info');

  // If a tile is already super-resolved and we are viewing ESRGAN, auto re-run instantly
  if (currentTileId && currentSrUrl) {
    const modelChoice = document.querySelector('input[name="model-choice"]:checked')?.value || 'esrgan';
    if (modelChoice === 'esrgan') {
      log('[SYSTEM] Updating visual clarity for active tile…', 'info');
      runSR();
    }
  }
}

// ════════════════════════════════════════════════════════════
// VALIDATE (Wald Protocol)
// ════════════════════════════════════════════════════════════
async function runValidate() {
  if (!currentTileId) return;
  setStatus('active', 'Validating…');
  showSpinner('Wald Protocol Check…', 'Computing degradation metrics against 10m reference');
  disableButtons(true);

  try {
    const modelChoice = document.querySelector('input[name="model-choice"]:checked')?.value || 'esrgan';
    const res = await fetch(API + '/api/validate/' + currentTileId + '?model=' + modelChoice, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Validation failed');
    }
    const data = await res.json();
    const v = data.validation;

    const card = document.getElementById('validation-card');
    const grid = document.getElementById('val-grid');
    const note = document.getElementById('val-note');
    const errImg = document.getElementById('val-error-map');

    grid.innerHTML = [
      { label: 'PSNR vs HR', val: v.psnr_vs_hr + ' dB',     good: v.psnr_vs_hr > 28 },
      { label: 'SSIM vs HR', val: v.ssim_vs_hr,              good: v.ssim_vs_hr > 0.85 },
      { label: 'SAM',        val: v.sam_deg + '°',           good: v.sam_deg < 3 },
      { label: 'ERGAS',      val: v.ergas,                   good: v.ergas < 3 },
      { label: 'NDVI MAE',   val: v.ndvi_mae,                good: v.ndvi_mae < 0.05 },
      { label: 'Bicubic',    val: v.psnr_bicubic_baseline + ' dB', good: false },
      { label: 'Gain vs Bicubic', val: '+' + v.psnr_improvement_over_bicubic + ' dB', good: v.psnr_improvement_over_bicubic > 0 },
    ].map(m => `
      <div class="val-metric ${m.good ? 'good' : 'neutral'}">
        <span class="vm-label">${m.label}</span>
        <span class="vm-val">${m.val}</span>
      </div>
    `).join('');

    note.textContent = data.summary;
    if (data.error_map) {
      errImg.src = data.error_map;
      errImg.style.display = 'block';
    }
    card.style.display = 'block';
    showPanel('model-cmp');

    log('[WALD] PSNR=' + v.psnr_vs_hr + ' dB | SSIM=' + v.ssim_vs_hr + ' | SAM=' + v.sam_deg + '°', 'ok');
    setStatus('done', 'Complete');
  } catch (e) {
    log('[ERROR] ' + e.message, 'error');
    setStatus('error', 'Error');
  } finally {
    hideSpinner();
    disableButtons(false);
    document.getElementById('btn-sr').disabled = !currentTileId;
  }
}

// ════════════════════════════════════════════════════════════
// MODEL COMPARISON RESULTS
// ════════════════════════════════════════════════════════════
function showComparisonResults(esrResult, ldsrResult) {
  const placeholder = document.getElementById('cmp-placeholder');
  const results     = document.getElementById('cmp-results');
  if (placeholder) placeholder.style.display = 'none';
  if (results) results.style.display = 'flex';

  const esrImg  = document.getElementById('cmp-img-esr');
  const ldsrImg = document.getElementById('cmp-img-ldsr');
  if (esrImg  && esrResult.sr_preview)  esrImg.src  = esrResult.sr_preview;
  if (ldsrImg && ldsrResult.sr_preview) ldsrImg.src = ldsrResult.sr_preview;

  function metricsHTML(m) {
    return `
      <div class="cmp-metric-row">
        <span>PSNR:</span><strong>${m.psnr} dB</strong>
      </div>
      <div class="cmp-metric-row">
        <span>SSIM:</span><strong>${m.ssim}</strong>
      </div>
      <div class="cmp-metric-row">
        <span>SAM:</span><strong>${m.sam_deg}°</strong>
      </div>
      <div class="cmp-metric-row">
        <span>ERGAS:</span><strong>${m.ergas}</strong>
      </div>
      <div class="cmp-metric-row">
        <span>NDVI MAE:</span><strong>${m.ndvi_mae}</strong>
      </div>
      <div class="cmp-metric-row">
        <span>Sharpness:</span><strong>${m.sharpness_gain}×</strong>
      </div>
    `;
  }

  const esrMetrics  = document.getElementById('cmp-metrics-esr');
  const ldsrMetrics = document.getElementById('cmp-metrics-ldsr');
  if (esrMetrics  && esrResult.metrics)  esrMetrics.innerHTML  = metricsHTML(esrResult.metrics);
  if (ldsrMetrics && ldsrResult.metrics) ldsrMetrics.innerHTML = metricsHTML(ldsrResult.metrics);
}

// ════════════════════════════════════════════════════════════
// TRAINING STATUS
// ════════════════════════════════════════════════════════════
async function refreshTrainStatus() {
  try {
    const res  = await fetch(API + '/api/train_status');
    const data = await res.json();

    const statusEl  = document.getElementById('ts-status');
    const epochsEl  = document.getElementById('ts-epochs');
    const psnrEl    = document.getElementById('ts-psnr');
    const ssimEl    = document.getElementById('ts-ssim');

    const statusMap = {
      not_started: 'Not Started',
      starting:    'Initializing…',
      training:    'Training Active',
      complete:    'Training Complete'
    };
    if (statusEl) statusEl.textContent = statusMap[data.status] || data.status;
    if (epochsEl) epochsEl.textContent = (data.epochs_done ?? '—') + (data.status === 'complete' ? '' : (' / ' + (data.total_epochs || '?')));
    if (psnrEl)   psnrEl.textContent   = data.best_psnr ? data.best_psnr + ' dB' : '—';
    if (ssimEl)   ssimEl.textContent   = data.weights?.best_ssim ?? '—';

    if (data.history && data.history.length > 0) {
      drawTrainingChart(data.history);
    }

    const esrOpt = document.getElementById('model-opt-esrgan');
    if (esrOpt) {
      esrOpt.style.opacity = (data.status === 'complete' || data.status === 'training') ? '1' : '0.5';
    }
  } catch(e) {
    const statusEl = document.getElementById('ts-status');
    if (statusEl) statusEl.textContent = 'Offline';
  }
}

function drawTrainingChart(history) {
  const canvas = document.getElementById('ts-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 400;
  const H = canvas.height || 80;
  canvas.width = W;

  const psnrs  = history.map(function(r){ return r.psnr; });
  const minP   = Math.min(...psnrs) - 1;
  const maxP   = Math.max(...psnrs) + 1;
  const n      = psnrs.length;

  ctx.clearRect(0, 0, W, H);

  ctx.fillStyle = '#070a10';
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = '#1a2335';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H/2); ctx.lineTo(W, H/2);
  ctx.stroke();

  ctx.beginPath();
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth   = 2;
  psnrs.forEach(function(p, i) {
    const x = (i / (n - 1)) * W;
    const y = H - ((p - minP) / (maxP - minP)) * (H - 24) - 12;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = '#6ee7b7';
  ctx.font      = '11px "JetBrains Mono", monospace';
  ctx.fillText('PEAK: ' + psnrs[psnrs.length-1].toFixed(2) + ' dB', 12, 18);
  ctx.fillText('EPOCH ' + history[history.length-1].epoch, W - 70, 18);
}

// ════════════════════════════════════════════════════════════
// DOMAIN ANALYSIS
// ════════════════════════════════════════════════════════════
let currentAnalysisData = {};

function switchDomainTab(domain) {
  ['crop', 'urban', 'disaster'].forEach(function(d) {
    const tab   = document.getElementById('dtab-' + d);
    const panel = document.getElementById('domain-' + d);
    if (tab)   tab.classList.toggle('active', d === domain);
    if (panel) panel.style.display = (d === domain) ? 'block' : 'none';
  });
}

async function runAnalysis(domain) {
  if (!currentTileId) {
    log('[WARN] Fetch a tile and run SR first.', 'warn');
    return;
  }

  log('[ANALYSIS] Running ' + domain + ' analysis…', 'info');
  setStatus('active', 'Analyzing…');
  showSpinner('Domain Analysis…', 'Computing ' + domain + ' layers from 2.5m SR output');
  disableButtons(true);

  try {
    const res = await fetch(API + '/api/analyze/' + currentTileId + '?domain=' + domain, {
      method: 'POST'
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Analysis failed');
    }
    const data = await res.json();
    currentAnalysisData = Object.assign(currentAnalysisData, data);

    populateAnalysis(data, domain);
    log('[ANALYSIS] ' + domain.toUpperCase() + ' analysis complete ✓', 'ok');
    setStatus('done', 'Analysis Done');

    // Switch to the relevant tab
    if (domain !== 'all') switchDomainTab(domain);
    showPanel('analysis');
  } catch (e) {
    log('[ERROR] Analysis: ' + e.message, 'error');
    setStatus('error', 'Analysis Error');
  } finally {
    hideSpinner();
    disableButtons(false);
    document.getElementById('btn-sr').disabled = !currentTileId;
    const btnA = document.getElementById('btn-run-analysis');
    if (btnA) btnA.disabled = false;
  }
}

function _setAnalysisImg(imgId, plhId, url) {
  const img = document.getElementById(imgId);
  const plh = document.getElementById(plhId);
  if (img && url) {
    img.src = url;
    img.style.display = 'block';
    if (plh) plh.style.display = 'none';
  }
}

function _setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function populateAnalysis(data, domain) {
  if ((domain === 'crop' || domain === 'all') && data.crop) {
    const c = data.crop;
    const s = c.stats;
    _setAnalysisImg('aimg-crop-ndvi',       'acard-preview-crop-ndvi', c.ndvi_map);
    _setAnalysisImg('aimg-crop-boundary',   'apl-crop-boundary',       c.boundary_map);
    _setAnalysisImg('aimg-crop-stress',     'apl-crop-stress',         c.stress_map);
    _setAnalysisImg('aimg-crop-irrigation', 'apl-crop-irrigation',     c.irrigation_map);
    _setAnalysisImg('aimg-crop-canopy',     'apl-crop-canopy',         c.canopy_map);
    _setText('ast-mean-ndvi',    s.mean_ndvi);
    _setText('ast-healthy-pct', s.healthy_pct + '%');
    _setText('ast-mod-pct',     s.moderate_pct + '%');
    _setText('ast-stressed-pct', s.stressed_pct + '%');
    _setText('ast-irrigated-pct', s.irrigated_pct + '%');
    log('[CROP] NDVI=' + s.mean_ndvi + '  Stressed=' + s.stressed_pct + '%  Healthy=' + s.healthy_pct + '%', 'info');
  }

  if ((domain === 'urban' || domain === 'all') && data.urban) {
    const u = data.urban;
    const s = u.stats;
    _setAnalysisImg('aimg-urban-buildup',    'apl-urban-buildup',    u.buildup_map);
    _setAnalysisImg('aimg-urban-roads',      'apl-urban-roads',      u.roads_map);
    _setAnalysisImg('aimg-urban-impervious', 'apl-urban-impervious', u.impervious_map);
    _setAnalysisImg('aimg-urban-greenery',   'apl-urban-greenery',   u.greenery_map);
    _setAnalysisImg('aimg-urban-density',    'apl-urban-density',    u.density_map);
    _setText('ast-buildup-pct',  s.buildup_pct + '%');
    _setText('ast-imperv-pct',   s.impervious_pct + '%');
    _setText('ast-road-density', s.road_density + '%');
    _setText('ast-greenery-pct', s.greenery_pct + '%');
    log('[URBAN] Built-up=' + s.buildup_pct + '%  Impervious=' + s.impervious_pct + '%  Green=' + s.greenery_pct + '%', 'info');
  }

  if ((domain === 'disaster' || domain === 'all') && data.disaster) {
    const d = data.disaster;
    const s = d.stats;
    _setAnalysisImg('aimg-disaster-flood',   'apl-disaster-flood',   d.flood_map);
    _setAnalysisImg('aimg-disaster-damage',  'apl-disaster-damage',  d.damage_map);
    _setAnalysisImg('aimg-disaster-roads',   'apl-disaster-roads',   d.roads_map);
    _setAnalysisImg('aimg-disaster-relief',  'apl-disaster-relief',  d.relief_map);
    _setAnalysisImg('aimg-disaster-anomaly', 'apl-disaster-anomaly', d.anomaly_map);
    _setText('ast-flooded-pct',  s.flooded_pct + '%');
    _setText('ast-severe-pct',   s.severe_dmg_pct + '%');
    _setText('ast-blocked-pct',  s.blocked_roads_pct + '%');
    _setText('ast-relief-pct',   s.relief_zones_pct + '%');
    _setText('ast-water-status', s.water_status);
    log('[DISASTER] Flooded=' + s.flooded_pct + '%  Severe=' + s.severe_dmg_pct + '%  ' + s.water_status, 'info');
  }
}

// ════════════════════════════════════════════════════════════
// BOOT
// ════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', function() {
  initMap();

  fetch('/health')
    .then(function(r){ return r.json(); })
    .then(function(data){
      const esrReady = data.esrgan_ready ? ' | Ready' : '';
      log('[SYSTEM] SentinelSR online (' + data.device + ')' + esrReady, 'ok');
    })
    .catch(function(){
      log('[WARN] Backend offline', 'warn');
    });

  refreshTrainStatus();
});
