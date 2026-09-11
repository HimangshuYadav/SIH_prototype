/* ============================================================
   SentinelSR — Frontend Application Logic (SIH Production Edition)
   ============================================================ */

const API = '';
let map, drawnItems, srLayer, lrLayer;
let currentTileId       = null;
let currentLrUrl        = null;
let currentSrUrl        = null;
let currentCirLrUrl     = null;
let currentCirSrUrl     = null;
let currentNdviLrUrl    = null;
let currentNdviSrUrl    = null;
let currentNdwiUrl      = null;
let currentUncertaintyUrl = null;
let currentMetrics      = null;
let currentComposite    = 'rgb';
let currentOverlayOpacity = 0.95;
let drawHandler         = null;
let isDrawing           = false;

const PRESETS = [
  { label: '🏙️ Hyderabad (Urban)',   bbox: [78.3692, 17.3850, 78.4800, 17.4800] },
  { label: '🌾 Punjab (Agriculture)', bbox: [75.7500, 30.8500, 75.8800, 30.9600] },
  { label: '🌊 Mumbai (Coastal)',     bbox: [72.8000, 18.9200, 72.9300, 19.0300] },
  { label: '🏛️ Delhi (Built-up)',     bbox: [77.1500, 28.5800, 77.2700, 28.6800] },
  { label: '🏔️ Shimla (Terrain)',     bbox: [77.1200, 31.0600, 77.2200, 31.1400] },
];

// ══════════════════════════════════════════════════════════════
// MAP INITIALIZATION
// ══════════════════════════════════════════════════════════════
function initMap() {
  map = L.map('map', {
    center: [20.5937, 78.9629],
    zoom: 5,
    zoomControl: true,
  });

  // Base Layer 1: EOX Sentinel-2 Cloudless (Authentic 10m Sentinel-2 Global Basemap)
  const s2Layer = L.tileLayer(
    'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg',
    {
      attribution: '© Sentinel-2 cloudless by EOX IT Services GmbH (Contains modified Copernicus Sentinel data)',
      maxZoom: 18,
    }
  );

  // Base Layer 2: Clean Dark Carto Map (for region picking without visual clutter)
  const darkLayer = L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    {
      attribution: '© CartoDB / OpenStreetMap',
      maxZoom: 19,
      subdomains: 'abcd',
    }
  );

  // Base Layer 3: High-Res Aerial (Commercial Reference)
  const esriLayer = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {
      attribution: '© Esri World Imagery (High-Res Aerial Reference)',
      maxZoom: 19,
    }
  );

  // Set Sentinel-2 10m as the default layer on map
  s2Layer.addTo(map);

  const baseLayers = {
    '🛰️ Sentinel-2 (10m Native)': s2Layer,
    '🗺️ Dark Canvas': darkLayer,
    '📸 High-Res Aerial (Ref)': esriLayer,
  };

  L.control.layers(baseLayers, null, { position: 'topright' }).addTo(map);

  drawnItems = new L.FeatureGroup().addTo(map);

  map.on(L.Draw.Event.CREATED, function (e) {
    drawnItems.clearLayers();
    drawnItems.addLayer(e.layer);
    const b = e.layer.getBounds();
    document.getElementById('bbox-west').value  = b.getWest().toFixed(5);
    document.getElementById('bbox-south').value = b.getSouth().toFixed(5);
    document.getElementById('bbox-east').value  = b.getEast().toFixed(5);
    document.getElementById('bbox-north').value = b.getNorth().toFixed(5);
    setDrawMode(false);
    hideHint();
    log('✏️ Bounding box selected. Click "Fetch Sentinel-2 Tile".', 'ok');
  });

  buildPresets();
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
        color: '#3b82f6', weight: 2.5,
        fillColor: '#3b82f6', fillOpacity: 0.15,
        dashArray: '6 4',
      },
    });
    drawHandler.enable();
    btn.classList.add('drawing');
    btn.innerHTML = '⏹ Cancel Drawing';
    showHint('🖱️ Click and drag across any area of interest');
    log('Draw mode active — drag a rectangle on map', 'info');
  } else {
    if (drawHandler) {
      try { drawHandler.disable(); } catch(e){}
      drawHandler = null;
    }
    btn.classList.remove('drawing');
    btn.innerHTML = '✏️ Draw Rectangle';
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

  drawnItems.clearLayers();
  const rect = L.rectangle([[s, w], [n, e]], {
    color: '#3b82f6', weight: 2.5,
    fillColor: '#3b82f6', fillOpacity: 0.15,
    dashArray: '6 4',
  });
  drawnItems.addLayer(rect);
  map.fitBounds([[s, w], [n, e]], { padding: [50, 50] });
  hideHint();
  log('📍 Preset selected: ' + p.label + ' — Click Fetch Tile', 'ok');
}

// ══════════════════════════════════════════════════════════════
// PANEL SWITCHING
// ══════════════════════════════════════════════════════════════
function showPanel(name) {
  const panels = ['map', 'compare', 'metrics', 'apps', 'info'];
  panels.forEach(function(p) {
    const el  = document.getElementById('panel-' + p);
    const lnk = document.getElementById('nav-' + p);
    if (el)  el.style.display = (p === name) ? (p === 'map' ? 'flex' : 'flex') : 'none';
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
    log('⚠️ Please define a region first by drawing or clicking a preset.', 'warn');
    return;
  }

  setStatus('active', 'Fetching STAC…');
  showSpinner('Querying Microsoft Planetary Computer STAC…', 'Accessing Sentinel-2 Level-2A BOA Reflectance Catalogue');
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
    log('✅ Sentinel-2 scene retrieved: ' + data.tile_id + ' (' + data.shape[1] + '×' + data.shape[2] + ' px)', 'ok');

    document.getElementById('btn-sr').disabled = false;
    document.getElementById('result-float').style.display = 'flex';
    setStatus('done', 'Scene Ready');
  } catch (e) {
    log('❌ ' + e.message, 'error');
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
  log('⏹️ Stopping process…', 'warn');
  const cancelBtn = document.getElementById('btn-cancel-task');
  if (cancelBtn) {
    cancelBtn.disabled = true;
    cancelBtn.textContent = 'Stopping…';
  }
  try {
    await fetch(API + '/api/cancel', { method: 'POST' });
    log('Cancellation signal sent to backend.', 'info');
  } catch (e) {
    log('Cancel error: ' + e.message, 'warn');
  }
}

// ══════════════════════════════════════════════════════════════
// RUN SUPER-RESOLUTION & ACCURACY PIPELINE
// ══════════════════════════════════════════════════════════════
async function runSR() {
  if (!currentTileId) return;
  const steps = parseInt(document.getElementById('steps-slider').value);
  const uncertainty = document.getElementById('chk-uncertainty').checked;
  const modelChoice = document.querySelector('input[name="model-choice"]:checked')?.value || 'ldsr';

  const modelLabel = modelChoice === 'esrgan' ? 'ESRGAN (Ours)' :
                     modelChoice === 'both'   ? 'ESRGAN + LDSR-S2 (Both)' : 'LDSR-S2 (SOTA)';

  const cancelBtn = document.getElementById('btn-cancel-task');
  if (cancelBtn) {
    cancelBtn.disabled = false;
    cancelBtn.textContent = '⏹️ Stop / Cancel Process';
  }

  setStatus('active', 'Running SR & Assessment…');
  showSpinner(
    'Running ' + modelLabel + '…',
    modelChoice === 'ldsr'
      ? 'Stochastic ensemble diffusion (' + steps + ' steps) + Wald protocol'
      : modelChoice === 'esrgan'
      ? 'ESRGAN RRDB inference + sliding-window tiling'
      : 'Running BOTH models — this will take longer'
  );
  disableButtons(true);

  try {
    const res = await fetch(API + '/api/sr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tile_id: currentTileId,
        sampling_steps: steps,
        compute_uncertainty: uncertainty,
        model_choice: modelChoice
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Super-Resolution inference failed');
    }

    const data = await res.json();

    if (modelChoice === 'both' && data.model_choice === 'both') {
      // Dual-model response: handle comparison
      const ldsr = data.ldsr;
      const esr  = data.esrgan;

      currentSrUrl     = ldsr.sr_preview;   // default SR view = LDSR
      currentLrUrl     = ldsr.lr_preview;
      currentCirLrUrl  = ldsr.lr_cir;
      currentCirSrUrl  = ldsr.sr_cir;
      currentNdviLrUrl = ldsr.lr_ndvi;
      currentNdviSrUrl = ldsr.sr_ndvi;
      currentNdwiUrl   = ldsr.sr_ndwi;
      currentUncertaintyUrl = ldsr.uncertainty_map;
      currentMetrics   = ldsr.metrics;

      log('⚔️ Both models complete! Comparison ready.', 'ok');
      log('  ESRGAN  → PSNR=' + esr.metrics.psnr + 'dB  SSIM=' + esr.metrics.ssim, 'info');
      log('  LDSR-S2 → PSNR=' + ldsr.metrics.psnr + 'dB  SSIM=' + ldsr.metrics.ssim, 'ok');

      setView('sr');
      populateCompare();
      populateMetrics(ldsr.metrics);
      populateApplications(ldsr);
      showComparisonResults(esr, ldsr);
      showPanel('model-cmp');
      populateDownloads(data.tile_id, ldsr.downloads || esr.downloads);
    } else {
      // Single model response
      currentSrUrl          = data.sr_preview;
      currentLrUrl          = data.lr_preview;
      currentCirLrUrl       = data.lr_cir;
      currentCirSrUrl       = data.sr_cir;
      currentNdviLrUrl      = data.lr_ndvi;
      currentNdviSrUrl      = data.sr_ndvi;
      currentNdwiUrl        = data.sr_ndwi;
      currentUncertaintyUrl = data.uncertainty_map;
      currentMetrics        = data.metrics;

      log('✨ SR complete: ' + (data.model || modelLabel), 'ok');
      log('   ' + data.lr_shape[1] + '×' + data.lr_shape[2] + ' → ' + data.sr_shape[1] + '×' + data.sr_shape[2], 'info');
      log('   PSNR=' + data.metrics.psnr + 'dB  SSIM=' + data.metrics.ssim + '  SAM=' + data.metrics.sam_deg + '°', 'ok');

      setView('sr');
      populateCompare();
      populateMetrics(data.metrics);
      populateApplications(data);
      populateDownloads(data.tile_id, data.downloads);
    }

    document.getElementById('btn-validate').disabled = false;
    setStatus('done', 'SR & Assessment Complete');
  } catch (e) {
    if (e.message && (e.message.toLowerCase().includes('cancel') || e.message.includes('499'))) {
      log('⏹️ Process was stopped by user.', 'warn');
      setStatus('idle', 'Process Stopped');
    } else {
      log('❌ ' + e.message, 'error');
      setStatus('error', 'Execution Error');
    }
  } finally {
    hideSpinner();
    disableButtons(false);
    document.getElementById('btn-sr').disabled = !currentTileId;
  }
}

// ══════════════════════════════════════════════════════════════
// MAP OVERLAY & OPACITY
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
// COMPARE PANEL POPULATION
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
  const pct = 100 - val;
  document.getElementById('compare-sr').style.clipPath  = 'inset(0 ' + pct + '% 0 0)';
  document.getElementById('compare-divider').style.left = val + '%';
}

// ══════════════════════════════════════════════════════════════
// VALIDATION & METRICS PANEL POPULATION
// ══════════════════════════════════════════════════════════════
function populateMetrics(m) {
  if (!m) return;

  document.getElementById('val-psnr').textContent      = m.psnr;
  document.getElementById('val-ssim').textContent      = m.ssim;
  const fidText = m.color_fidelity_pct ? ` (${m.color_fidelity_pct}% fidelity)` : '';
  document.getElementById('val-sam').textContent       = m.sam_deg + '°' + fidText;
  document.getElementById('val-ergas').textContent     = m.ergas;
  document.getElementById('val-sharpness').textContent = m.sharpness_gain;
  document.getElementById('val-ndvi-mae').textContent  = m.ndvi_mae;

  // Band details table
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
          <td><strong style="color: var(--green);">${b.correlation}</strong></td>
          <td>${b.bias >= 0 ? '+' : ''}${b.bias}</td>
          <td>${b.mae}</td>
        </tr>
      `;
    }).join('');
  }
}

// ══════════════════════════════════════════════════════════════
// APPLICATIONS & GIS POPULATION
// ══════════════════════════════════════════════════════════════
function populateApplications(data) {
  // NDVI Agriculture
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

  // NDWI Disaster
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

  // Uncertainty Quantification
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
  showHint('✏️ Draw a rectangle or pick a preset city');
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

// ════════════════════════════════════════════════════════════
// MODEL SELECTOR
// ════════════════════════════════════════════════════════════
function onModelChange(radio) {
  const v = radio.value;
  // Update steps slider: ESRGAN needs no steps (not diffusion)
  const stepsRow = document.getElementById('steps-slider')?.closest('.slider-label');
  const uncRow   = document.getElementById('chk-uncertainty')?.closest('.checkbox-label');
  if (stepsRow) stepsRow.style.opacity = (v === 'esrgan') ? '0.4' : '1';
  if (uncRow)   uncRow.style.opacity   = (v === 'esrgan') ? '0.4' : '1';

  const hint = {
    ldsr:   '🌊 LDSR-S2: high-quality generative diffusion (ESA SOTA)',
    esrgan: '⚡ ESRGAN: our trained CNN generator (faster, comparable quality)',
    both:   '⚔️ Both: run both models and compare side-by-side'
  };
  log(hint[v] || '', 'info');
}

// ════════════════════════════════════════════════════════════
// VALIDATE (Wald Protocol true HR validation)
// ════════════════════════════════════════════════════════════
async function runValidate() {
  if (!currentTileId) return;
  setStatus('active', 'Running HR Validation…');
  showSpinner('Wald\'s Protocol Validation', 'Comparing SR output vs original HR Sentinel-2 pixels…');
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

    // Show validation card in model-cmp panel
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
      { label: 'Bicubic Baseline', val: v.psnr_bicubic_baseline + ' dB', good: false },
      { label: 'SR Improvement', val: '+' + v.psnr_improvement_over_bicubic + ' dB', good: v.psnr_improvement_over_bicubic > 0 },
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

    log('🔬 Validation: PSNR=' + v.psnr_vs_hr + 'dB vs HR | Bicubic=' + v.psnr_bicubic_baseline + 'dB | Improvement=+' + v.psnr_improvement_over_bicubic + 'dB', 'ok');
    setStatus('done', 'Validation Complete');
  } catch (e) {
    log('❌ Validation: ' + e.message, 'error');
    setStatus('error', 'Validation Error');
  } finally {
    hideSpinner();
    disableButtons(false);
    document.getElementById('btn-sr').disabled = !currentTileId;
  }
}

// ════════════════════════════════════════════════════════════
// MODEL COMPARISON RESULTS DISPLAY
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

  function metricsHTML(m, model) {
    return `
      <div class="cmp-metric-row">
        <span>📊 PSNR</span><strong>${m.psnr} dB</strong>
      </div>
      <div class="cmp-metric-row">
        <span>📊 SSIM</span><strong>${m.ssim}</strong>
      </div>
      <div class="cmp-metric-row">
        <span>🌟 SAM</span><strong>${m.sam_deg}°</strong>
      </div>
      <div class="cmp-metric-row">
        <span>🔥 ERGAS</span><strong>${m.ergas}</strong>
      </div>
      <div class="cmp-metric-row">
        <span>🌿 NDVI MAE</span><strong>${m.ndvi_mae}</strong>
      </div>
      <div class="cmp-metric-row">
        <span>⚡ Sharpness</span><strong>${m.sharpness_gain}x</strong>
      </div>
    `;
  }

  const esrMetrics  = document.getElementById('cmp-metrics-esr');
  const ldsrMetrics = document.getElementById('cmp-metrics-ldsr');
  if (esrMetrics  && esrResult.metrics)  esrMetrics.innerHTML  = metricsHTML(esrResult.metrics, 'ESRGAN');
  if (ldsrMetrics && ldsrResult.metrics) ldsrMetrics.innerHTML = metricsHTML(ldsrResult.metrics, 'LDSR-S2');
}

// ════════════════════════════════════════════════════════════
// TRAINING STATUS PANEL
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
      not_started: '❌ Not Started',
      starting:    '⏳ Starting…',
      training:    '🏋️ Training',
      complete:    '✅ Complete'
    };
    if (statusEl) statusEl.textContent = statusMap[data.status] || data.status;
    if (epochsEl) epochsEl.textContent = (data.epochs_done ?? '—') + (data.status === 'complete' ? '' : (' / ' + (data.total_epochs || '?')));
    if (psnrEl)   psnrEl.textContent   = data.best_psnr ? data.best_psnr + ' dB' : '—';
    if (ssimEl)   ssimEl.textContent   = data.weights?.best_ssim ?? '—';

    // Draw PSNR chart
    if (data.history && data.history.length > 0) {
      drawTrainingChart(data.history);
    }

    // Update model-opt-esrgan availability
    const esrOpt = document.getElementById('model-opt-esrgan');
    if (esrOpt) {
      esrOpt.style.opacity = (data.status === 'complete' || data.status === 'training') ? '1' : '0.5';
    }
  } catch(e) {
    const statusEl = document.getElementById('ts-status');
    if (statusEl) statusEl.textContent = '❓ Backend offline';
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

  // Background
  ctx.fillStyle = 'rgba(255,255,255,0.03)';
  ctx.fillRect(0, 0, W, H);

  // PSNR line
  ctx.beginPath();
  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth   = 2;
  psnrs.forEach(function(p, i) {
    const x = (i / (n - 1)) * W;
    const y = H - ((p - minP) / (maxP - minP)) * H;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Label
  ctx.fillStyle = '#86efac';
  ctx.font      = '11px Inter, sans-serif';
  ctx.fillText('PSNR: ' + psnrs[psnrs.length-1].toFixed(2) + ' dB', 8, 14);
  ctx.fillText('Epoch ' + history[history.length-1].epoch, W - 60, 14);
}

// ════════════════════════════════════════════════════════════
// APPLICATION BOOT
// ════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', function() {
  initMap();

  fetch('/health')
    .then(function(r){ return r.json(); })
    .then(function(data){
      const esrReady = data.esrgan_ready ? ' | ESRGAN ✓' : ' | ESRGAN training...';
      log('SentinelSR v' + data.version + ' online (' + data.device + ')' + esrReady, 'ok');
    })
    .catch(function(){
      log('⚠️ Backend offline — start backend.py', 'warn');
    });

  // Auto-load training status
  refreshTrainStatus();
});
