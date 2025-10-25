const API_RUN_MAIN = "http://localhost:8000/run-main";

// Elements
const mainUploadArea = document.getElementById('mainUploadArea');
const fileUploader = document.getElementById('fileUploader');
const uploadTypeText = document.getElementById('uploadTypeText');
const imagePreview = document.getElementById('imagePreview'); // no longer shown on page
const popupImagePreview = document.getElementById('popupImagePreview'); // legacy, no longer used
const imagePreviewPopup = document.getElementById('imagePreviewPopup');
const sidePopupImage = document.getElementById('sidePopupImage');
const sidePreviewClose = document.getElementById('sidePreviewClose');
const predictButton = document.getElementById('predictButton');

const statusRow = document.getElementById('statusRow');
const statusText = document.getElementById('statusText');
const resultsCard = document.getElementById('resultsCard');
const resultsList = document.getElementById('resultsList');
const testClassSelect = document.getElementById('testClassSelect');


// Emoji toggle -> popup modals
const toggleElephantBtn = document.getElementById('toggleElephantBtn');
const elephantModal = document.getElementById('elephantModal');
const eleModalClose = document.getElementById('eleModalClose');
const elephantToggleImage = document.getElementById('elephantToggleImage');
const DEMO_ELE_IMG = 'https://png.pngtree.com/png-vector/20250717/ourlarge/pngtree-realistic-male-african-elephant-staring-at-camera-against-white-background-png-image_16703761.png';

const toggleHorseBtn = document.getElementById('toggleHorseBtn');
const horseModal = document.getElementById('horseModal');
const horseModalClose = document.getElementById('horseModalClose');
const horseToggleImage = document.getElementById('horseToggleImage');
const DEMO_HORSE_IMG = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAPCed-LtF456eO3YYex3iy_4aMV5HzCkjvw&s';

function openElephantModal() {
  elephantToggleImage.src = DEMO_ELE_IMG;
  elephantModal.style.display = 'flex';
  elephantModal.setAttribute('aria-hidden', 'false');
}
function closeElephantModal() {
  elephantModal.style.display = 'none';
  elephantModal.setAttribute('aria-hidden', 'true');
  elephantToggleImage.src = '';
}

function openHorseModal() {
  horseToggleImage.src = DEMO_HORSE_IMG;
  horseModal.style.display = 'flex';
  horseModal.setAttribute('aria-hidden', 'false');
}
function closeHorseModal() {
  horseModal.style.display = 'none';
  horseModal.setAttribute('aria-hidden', 'true');
  horseToggleImage.src = '';
}

toggleElephantBtn?.addEventListener('click', () => {
  const isHidden = elephantModal.style.display === 'none' || elephantModal.style.display === '';
  if (isHidden) openElephantModal(); else closeElephantModal();
});

toggleHorseBtn?.addEventListener('click', () => {
  const isHidden = horseModal.style.display === 'none' || horseModal.style.display === '';
  if (isHidden) openHorseModal(); else closeHorseModal();
});

eleModalClose?.addEventListener('click', closeElephantModal);
horseModalClose?.addEventListener('click', closeHorseModal);

elephantModal?.addEventListener('click', (e) => { if (e.target === elephantModal) closeElephantModal(); });
horseModal?.addEventListener('click', (e) => { if (e.target === horseModal) closeHorseModal(); });

let overlayCanvas, overlayCtx;

function getActivePreviewImg() {
  return sidePopupImage || imagePreview; // draw on the side popup image
}

function ensureCanvas() {
  const targetImg = getActivePreviewImg();
  if (!targetImg) return;
  const parent = targetImg.parentElement;
  if (!overlayCanvas) {
    overlayCanvas = document.createElement('canvas');
    overlayCanvas.style.position = 'absolute';
    overlayCanvas.style.left = '0';
    overlayCanvas.style.top = '0';
    overlayCanvas.style.pointerEvents = 'none';
    overlayCanvas.id = 'overlay';
    if (parent) {
      parent.style.position = 'relative';
      parent.appendChild(overlayCanvas);
    }
  }
  // Size canvas to the currently visible image in popup
  overlayCanvas.width = targetImg.clientWidth || targetImg.naturalWidth || 0;
  overlayCanvas.height = targetImg.clientHeight || targetImg.naturalHeight || 0;
  overlayCtx = overlayCanvas.getContext('2d');
}

function clearCanvas() {
  if (overlayCtx && overlayCanvas) overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// Helper to format timestamps as YYYY-MM-DD HH:MM:SS.hh (hundredths)
// Format timestamps as YYYY-MM-DD HH:MM:SS.mmm (3-digit milliseconds)
function formatDateTimeHundredths(ms) {
  const d = new Date(ms);
  const pad = (n, l = 2) => String(n).padStart(l, '0');
  const pad3 = (n) => String(n).padStart(3, '0'); // for milliseconds
  const YYYY = d.getFullYear();
  const MM = pad(d.getMonth() + 1);
  const DD = pad(d.getDate());
  const hh = pad(d.getHours());
  const mm = pad(d.getMinutes());
  const ss = pad(d.getSeconds());
  const mmm = pad3(d.getMilliseconds());
  return `${YYYY}-${MM}-${DD} ${hh}:${mm}:${ss}.${mmm}`;
}


// Format time as HH:MM:SS.mmm (3-digit milliseconds)
function formatTimeHundredths(ms) {
  const d = new Date(ms);
  const pad = (n, l = 2) => String(n).padStart(l, '0');
  const pad3 = (n) => String(n).padStart(3, '0');
  const hh = pad(d.getHours());
  const mm = pad(d.getMinutes());
  const ss = pad(d.getSeconds());
  const mmm = pad3(d.getMilliseconds());
  return `${hh}:${mm}:${ss}.${mmm}`;
}


// Pick the single largest detection by bbox area
function pickLargestDetection(dets = []) {
  let best = null;
  let bestArea = -1;
  for (const d of dets) {
    if (!d || !Array.isArray(d.bbox) || d.bbox.length < 4) continue;
    const [x1, y1, x2, y2] = d.bbox.map(Number);
    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);
    const area = w * h;
    if (area > bestArea) {
      bestArea = area;
      best = d;
    }
  }
  return best;
}

function drawDetections(dets, naturalW, naturalH) {
  const targetImg = getActivePreviewImg();
  if (!targetImg) return;
  ensureCanvas();
  clearCanvas();
  if (!naturalW || !naturalH) return;
  const scaleX = (targetImg.clientWidth || naturalW) / naturalW;
  const scaleY = (targetImg.clientHeight || naturalH) / naturalH;

  const d = pickLargestDetection(dets);
  if (!d) return;

  const [x1, y1, x2, y2] = d.bbox;
  const rx1 = Math.round(x1 * scaleX);
  const ry1 = Math.round(y1 * scaleY);
  const rx2 = Math.round(x2 * scaleX);
  const ry2 = Math.round(y2 * scaleY);
  const w = rx2 - rx1;
  const h = ry2 - ry1;

  overlayCtx.lineWidth = 3;
  overlayCtx.strokeStyle = 'rgba(0, 255, 0, 0.85)';
  overlayCtx.strokeRect(rx1, ry1, w, h);

  const label = d.overlay_text || `${d.identity} (${(d.confidence*100).toFixed(3)}%)`;
  overlayCtx.font = '14px Arial';
  const textW = overlayCtx.measureText(label).width + 8;
  const textH = 18;
  overlayCtx.fillStyle = 'rgba(0,0,0,0.6)';
  overlayCtx.fillRect(rx1, Math.max(0, ry1 - textH), textW, textH);
  overlayCtx.fillStyle = 'white';
  overlayCtx.fillText(label, rx1 + 4, ry1 - 4);
}

// Handle file selection (images only) — do not preview on page
fileUploader.addEventListener('change', (event) => {
  const files = Array.from(event.target.files || []).filter(f => f.type.startsWith('image/'));
  if (uploadTypeText) {
    const n = files.length;
    uploadTypeText.textContent = n ? `${n} image(s) selected` : 'Select your images';
  }
  // Hide the in-page preview completely
  if (imagePreview) {
    imagePreview.src = '#';
    imagePreview.style.display = 'none';
  }
  clearCanvas();
});

// Helper to show/hide loading overlay and lockout UI
function setLoading(on, msg = 'Predicting...') {
  const overlay = document.getElementById('loadingOverlay');
  const msgEl   = document.getElementById('loadingMessage');

  if (overlay && msgEl) {
    msgEl.textContent = msg;
    overlay.style.display = on ? 'flex' : 'none';
    overlay.setAttribute('aria-hidden', on ? 'false' : 'true');
  }

  // prevent background scrolling/clicks
  document.body.style.overflow = on ? 'hidden' : '';

  // keep your existing button lock
  predictButton.disabled = on;

  // fail-safe so it never stays stuck
  if (on) {
    clearTimeout(setLoading._fuse);
    setLoading._fuse = setTimeout(() => {
      if (overlay) overlay.style.display = 'none';
      predictButton.disabled = false;
      document.body.style.overflow = '';
      console.warn('[Loading] Auto-dismissed after 20s failsafe.');
    }, 20000);
  } else {
    clearTimeout(setLoading._fuse);
  }
}

async function postImage(url, file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(url, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function extractVideoFrame(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.src = URL.createObjectURL(file);
    video.onloadeddata = () => {
      const t = Math.min(Math.max(video.duration / 2, 0), video.duration || 0);
      video.currentTime = t;
    };
    video.onseeked = () => {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      canvas.toBlob((blob) => {
        if (!blob) return reject(new Error('Failed to extract frame'));
        resolve(new File([blob], 'frame.jpg', { type: 'image/jpeg' }));
      }, 'image/jpeg', 0.9);
    };
    video.onerror = () => reject(new Error('Cannot load video'));
  });
}

// Read a File as a data URL
function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// --- helper to clear results ---
function hideList() {
  resultsCard.style.display = 'none';
  resultsList.innerHTML = '';
}

// --- wait for DOM before binding ---
document.addEventListener('DOMContentLoaded', () => {
  if (!predictButton) {
    console.error('[UI] #predictButton not found.');
    return;
  }
  predictButton.addEventListener('click', onPredictClick);

  console.log('[UI] Predict handler attached');
});

// Global state for batch results
let batchResults = []; // [{ fileName, previewSrc, detections, tStart, tEnd, process_time_sec, process_time_ms }]
let overallStartMs = 0;
let overallEndMs = 0;
let currentPage = 0; // 0..batchResults.length (last is summary)

function renderPopupPage(pageIndex) {
  console.log('[Batch] Render page', pageIndex, 'of', batchResults.length, '+ summary');
  const popupEl = document.getElementById('resultsPopup');
  const list = document.getElementById('popupResultsList');
  const pager = document.getElementById('popupPager');
  list.innerHTML = '';
  pager.innerHTML = '';

  const totalPages = batchResults.length + 1; // last for summary

  // Prev button
  const prevBtn = document.createElement('button');
  prevBtn.textContent = 'Prev';
  prevBtn.className = 'pager-btn';
  prevBtn.style.margin = '6px 6px 6px 0';
  prevBtn.onclick = () => {
    currentPage = (currentPage - 1 + totalPages) % totalPages;
    renderPopupPage(currentPage);
  };
  pager.appendChild(prevBtn);

  // Build numbered buttons
  for (let i = 0; i < totalPages; i++) {
    const btn = document.createElement('button');
    btn.className = 'pager-btn';
    btn.textContent = (i < batchResults.length) ? `${i + 1}` : 'Summary';
    btn.style.margin = '6px 4px';
    btn.style.padding = '6px 10px';
    btn.style.borderRadius = '6px';
    btn.style.border = '1px solid #80A4ED';
    btn.style.background = (i === pageIndex) ? '#80A4ED' : '#fff';
    btn.style.color = '#1D1A1B';
    btn.addEventListener('click', () => {
      currentPage = i;
      renderPopupPage(currentPage);
    });
    pager.appendChild(btn);
  }

  // Next button
  const nextBtn = document.createElement('button');
  nextBtn.textContent = 'Next';
  nextBtn.className = 'pager-btn';
  nextBtn.style.margin = '6px 0 6px 6px';
  nextBtn.onclick = () => {
    currentPage = (currentPage + 1) % totalPages;
    renderPopupPage(currentPage);
  };
  pager.appendChild(nextBtn);

  // Page indicator
  const info = document.createElement('span');
  info.textContent = ` Page ${pageIndex + 1} / ${totalPages} `;
  info.style.marginLeft = '8px';
  info.style.fontSize = '12px';
  pager.appendChild(info);

  if (pageIndex < batchResults.length) {
    const item = batchResults[pageIndex];
    // Per-file details
    const h = document.createElement('li');
    h.innerHTML = `<strong>File:</strong> ${item.fileName}`;
    list.appendChild(h);

    const dets = item.detections || [];
    const d = pickLargestDetection(dets);
    if (d) {
      const conf = Number(d?.similarity ?? 0);
      const name = (d?.identity && d.identity !== 'Unknown') ? d.identity : 'Unknown Elephant';
const testClass = item.test_class ?? (testClassSelect?.value || '1');

const li = document.createElement('li');
li.innerHTML = `
  <strong>${name}</strong><br>
  Confidence: ${(conf * 100).toFixed(3)}%<br>
  Test Class: ${testClass}
`;
list.appendChild(li);
    } else {
      const li = document.createElement('li');
      li.textContent = 'No elephant detected.';
      list.appendChild(li);
    }

    // Start–End (for reference)
    const rangeEl = document.createElement('li');
    rangeEl.innerHTML = `<em>Start: ${formatDateTimeHundredths(item.tStart)} — End: ${formatDateTimeHundredths(item.tEnd)}</em>`;
    list.appendChild(rangeEl);

    // Model-only per-image time (preferred over client diff)
    const procSec = (item.process_time_sec != null)
      ? Number(item.process_time_sec)
      : ((item.tEnd - item.tStart) / 1000);
    const timeEl = document.createElement('li');
    timeEl.innerHTML = `<em>⏱ Time elapsed: ${procSec.toFixed(3)} s</em>`;
    timeEl.style.textAlign = 'right';
    list.appendChild(timeEl);

    // Show one preview image in side popup and draw overlay
    if (item.previewSrc) {
      const imgTmp = new Image();
      imgTmp.onload = () => {
        openSidePreview(item.previewSrc);
        requestAnimationFrame(() => {
          ensureCanvas();
          drawDetections(item.detections || [], imgTmp.naturalWidth, imgTmp.naturalHeight);
        });
      };
      imgTmp.src = item.previewSrc;
    } else {
      closeSidePreview();
    }
  } else {
    // Summary page content (main popup). Hide side preview on summary.
    // Summary page content (main popup). Hide side preview on summary.
// Summary page content (main popup). Hide side preview on summary.
const totalFiles = batchResults.length;

// === Compute total model time and per-batch cumulative time ===
let cumulativeTime = 0;
for (const r of batchResults) {
  cumulativeTime += Number(r.process_time_sec) || 0;
}
const totalModelSec = cumulativeTime;

// Use model time for “start” and “end” timestamps — simulate sequential model run
const modelStart = batchResults[0]?.tStart || Date.now();
const modelEnd = modelStart + totalModelSec * 1000;

// Display these synthetic timestamps to match model time
const startLi = document.createElement('li');
startLi.innerHTML = `<strong>Overall Start (model):</strong> ${formatDateTimeHundredths(modelStart)}`;
list.appendChild(startLi);

const endLi = document.createElement('li');
endLi.innerHTML = `<strong>Overall End (model):</strong> ${formatDateTimeHundredths(modelEnd)}`;
list.appendChild(endLi);

// Model-only total duration
const durLi = document.createElement('li');
durLi.innerHTML = `<em>⏱ Time elapsed: ${totalModelSec.toFixed(3)} s for ${totalFiles} image(s)</em>`;
durLi.style.textAlign = 'right';
list.appendChild(durLi);

// === Grid of elephant pictures ===
const gridWrapper = document.createElement('div');
gridWrapper.className = 'summary-grid';

let count = 0;
for (const [idx, it] of batchResults.entries()) {
  const d = pickLargestDetection(it.detections || []);
  if (!d) continue;
  count++;
  const name = d.identity && d.identity !== 'Unknown' ? d.identity : 'Unknown';
  const conf = Number(d.confidence ?? 0) * 100;

  const itemDiv = document.createElement('div');
  const thumb = document.createElement('div');
  thumb.className = 'thumb';
  const img = document.createElement('img');
  img.src = it.previewSrc;
  img.alt = `${it.fileName} → ${name}`;
  thumb.appendChild(img);

  const cap = document.createElement('div');
  cap.className = 'caption';
  cap.innerHTML = `#${idx + 1} ${name} (${conf.toFixed(3)}%)`;

  itemDiv.appendChild(thumb);
  itemDiv.appendChild(cap);
  gridWrapper.appendChild(itemDiv);
}

if (!count) {
  const none = document.createElement('li');
  none.textContent = 'No elephants predicted in the selected images.';
  list.appendChild(none);
} else {
  list.appendChild(gridWrapper);
}

// Hide side preview on summary
closeSidePreview();


  }

  // Ensure popup visible
  popupEl.style.display = 'flex';
  popupEl.setAttribute('aria-hidden', 'false');
}


function showBatchResults(results, startMs, endMs) {
  console.log('[Batch] Results pages:', results.length, '+ 1 summary');
  batchResults = results;
  overallStartMs = startMs;
  overallEndMs = endMs;
  currentPage = 0;

  const closeBtn = document.getElementById('popupCloseBtn');
  closeBtn.onclick = () => {
    const popupEl = document.getElementById('resultsPopup');
    popupEl.style.display = 'none';
    popupEl.setAttribute('aria-hidden', 'true');
  };

  const popupEl = document.getElementById('resultsPopup');
  popupEl.addEventListener('click', e => {
    if (e.target === popupEl) {
      popupEl.style.display = 'none';
      popupEl.setAttribute('aria-hidden', 'true');
    }
  });

  renderPopupPage(currentPage);
}

// --- main click handler ---
async function onPredictClick() {
  console.log('[UI] Predict clicked');
  hideList();
  clearCanvas();

  const allFiles = Array.from(fileUploader.files || []);
  const files = allFiles.filter(f => f.type.startsWith('image/'));
  console.log('[Batch] Selected images:', files.length);
  if (!files.length) {
    showInfo('Please select image(s) before Predicting!', 'No image selected');
    return;
  }
  if (files.length !== allFiles.length) {
    showInfo('Only images are supported. Non-image files were ignored.');
  }

  try {
    setLoading(true, 'Predicting...');
    const selectedTestClass = testClassSelect?.value || '1';

    const results = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const previewSrc = await fileToDataURL(file);

      setLoading(true, `Predicting ${i + 1}/${files.length}...`);
      const tStart = Date.now();
      const json = await postImage(API_RUN_MAIN, file);
      const tEnd = Date.now();

      results.push({
  fileName: file.name,
  previewSrc,
  detections: json.detections || [],
  process_time_sec: json.process_time_sec,
  process_time_ms: json.process_time_ms,
  tStart,
  tEnd,
  test_class: selectedTestClass,   // <— store once per batch run
});

    }

    // Overall times reflect first and last image (for reference only)
    const overallStart = results[0]?.tStart;
    const overallEnd = results[results.length - 1]?.tEnd;

    setLoading(false);

    // If none of the results have detections
    const anyDetections = results.some(r => (r.detections || []).length);
    if (!anyDetections) {
      const noResultsPopup = document.getElementById('noResultsPopup');
      const noResultsCloseBtn = document.getElementById('noResultsCloseBtn');

      noResultsPopup.style.display = 'flex';
      noResultsPopup.setAttribute('aria-hidden', 'false');
      noResultsCloseBtn.onclick = () => {
        noResultsPopup.style.display = 'none';
        noResultsPopup.setAttribute('aria-hidden', 'true');
      };
      noResultsPopup.addEventListener('click', (e) => {
        if (e.target === noResultsPopup) {
          noResultsPopup.style.display = 'none';
          noResultsPopup.setAttribute('aria-hidden', 'true');
        }
      });
      return;
    }
    console.log('[DEBUG] process_time_sec list:', results.map(r => r.process_time_sec));
console.log('[DEBUG] model-sum =', results.reduce((a, r) => a + (Number(r.process_time_sec) || 0), 0).toFixed(3),
  's'
);
    // Show paginated popup and sync overlay with selected page
    showBatchResults(results, overallStart, overallEnd);

  } catch (err) {
    console.error(err);
    showInfo('An error occurred during prediction');
  } finally {
    setLoading(false);
  }
}

function showInfo(text, title = 'Heads up') {
  const popup = document.getElementById('infoPopup');
  const titleEl = document.getElementById('infoTitle');
  const textEl = document.getElementById('infoText');
  const closeBtn = document.getElementById('infoCloseBtn');
  if (!popup) return alert(text || '');
  if (titleEl) titleEl.textContent = title;
  if (textEl) textEl.textContent = text;
  popup.style.display = 'flex';
  popup.setAttribute('aria-hidden', 'false');
  const onClose = () => {
    popup.style.display = 'none';
    popup.setAttribute('aria-hidden', 'true');
    if (closeBtn) closeBtn.removeEventListener('click', onClose);
  };
  if (closeBtn) closeBtn.addEventListener('click', onClose);
  popup.addEventListener('click', (e) => { if (e.target === popup) onClose(); });
}

// ---- side preview helpers ----
// ---- side preview helpers ----
function openSidePreview(src) {
  const panel = document.getElementById('imagePreviewPopup'); // side panel container
  const img   = document.getElementById('sidePopupImage');    // <img> inside the panel
  const close = document.getElementById('sidePreviewClose');  // close button (optional)

  if (!panel || !img) return; // guard if DOM not present
  img.src = src || '';
  panel.style.display = 'block';
  panel.setAttribute('aria-hidden', 'false');

  if (close) {
    // ensure we don't stack multiple listeners
    close.onclick = closeSidePreview;
  }
  // click outside to close
  panel.addEventListener('click', (e) => {
    if (e.target === panel) closeSidePreview();
  }, { once: true });
}

function closeSidePreview() {
  const panel = document.getElementById('imagePreviewPopup');
  const img   = document.getElementById('sidePopupImage');
  if (!panel) return;
  panel.style.display = 'none';
  panel.setAttribute('aria-hidden', 'true');
  if (img) img.src = '';
  // if you draw overlays, clear them too
  if (typeof clearCanvas === 'function') clearCanvas();
}

