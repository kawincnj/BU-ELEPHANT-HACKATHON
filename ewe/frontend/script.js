const API_RUN_MAIN = "http://localhost:8000/run-main";

// Elements
const mainUploadArea = document.getElementById('mainUploadArea');
const fileUploader = document.getElementById('fileUploader');
const uploadTypeText = document.getElementById('uploadTypeText');
const imagePreview = document.getElementById('imagePreview');
const predictButton = document.getElementById('predictButton');

const statusRow = document.getElementById('statusRow');
const statusText = document.getElementById('statusText');
const resultsCard = document.getElementById('resultsCard');
const resultsList = document.getElementById('resultsList');

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

function ensureCanvas() {
  if (!overlayCanvas) {
    overlayCanvas = document.createElement('canvas');
    overlayCanvas.style.position = 'absolute';
    overlayCanvas.style.left = '0';
    overlayCanvas.style.top = '0';
    overlayCanvas.style.pointerEvents = 'none';
    overlayCanvas.id = 'overlay';
    imagePreview.parentElement.style.position = 'relative';
    imagePreview.parentElement.appendChild(overlayCanvas);
  }
  overlayCanvas.width = imagePreview.clientWidth;
  overlayCanvas.height = imagePreview.clientHeight;
  overlayCtx = overlayCanvas.getContext('2d');
}

function clearCanvas() {
  if (overlayCtx && overlayCanvas) overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// Helper to format timestamps as YYYY-MM-DD HH:MM:SS.hh (hundredths)
function formatDateTimeHundredths(ms) {
  const d = new Date(ms);
  const pad = (n, l = 2) => String(n).padStart(l, '0');
  const YYYY = d.getFullYear();
  const MM = pad(d.getMonth() + 1);
  const DD = pad(d.getDate());
  const hh = pad(d.getHours());
  const mm = pad(d.getMinutes());
  const ss = pad(d.getSeconds());
  const hundredths = pad(Math.floor(d.getMilliseconds() / 10));
  return `${YYYY}-${MM}-${DD} ${hh}:${mm}:${ss}.${hundredths}`;
}

// Helper to format timestamps as HH:MM:SS.hh (hundredths)
function formatTimeHundredths(ms) {
  const d = new Date(ms);
  const pad = (n, l = 2) => String(n).padStart(l, '0');
  const hh = pad(d.getHours());
  const mm = pad(d.getMinutes());
  const ss = pad(d.getSeconds());
  const hundredths = pad(Math.floor(d.getMilliseconds() / 10));
  return `${hh}:${mm}:${ss}.${hundredths}`;
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
  ensureCanvas();
  clearCanvas();
  const scaleX = imagePreview.clientWidth / naturalW;
  const scaleY = imagePreview.clientHeight / naturalH;

  const d = pickLargestDetection(dets);
  if (!d) return;

  const [x1, y1, x2, y2] = d.bbox;
  const rx1 = Math.round(x1 * scaleX);
  const ry1 = Math.round(y1 * scaleY);
  const rx2 = Math.round(x2 * scaleX);
  const ry2 = Math.round(y2 * scaleY);
  const w = rx2 - rx1;
  const h = ry2 - ry1;

  // Box
  overlayCtx.lineWidth = 3;
  overlayCtx.strokeStyle = 'rgba(0, 255, 0, 0.85)';
  overlayCtx.strokeRect(rx1, ry1, w, h);

  // Label
  const label = d.overlay_text || `${d.identity} (${(d.confidence*100).toFixed(1)}%)`;
  overlayCtx.font = '14px Arial';
  const textW = overlayCtx.measureText(label).width + 8;
  const textH = 18;
  overlayCtx.fillStyle = 'rgba(0,0,0,0.6)';
  overlayCtx.fillRect(rx1, Math.max(0, ry1 - textH), textW, textH);
  overlayCtx.fillStyle = 'white';
  overlayCtx.fillText(label, rx1 + 4, ry1 - 4);
}

// Handle file selection (image or video)
fileUploader.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const isImage = file.type.startsWith('image/');

  if (isImage) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.onload = () => {
        imagePreview.style.display = 'block';
        ensureCanvas();
        clearCanvas();
      };
    };
    reader.readAsDataURL(file);
  } else {
    // Hide preview for non-image files (e.g., video)
    imagePreview.src = '#';
    imagePreview.style.display = 'none';
    clearCanvas();
  }
});

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


function showPopupResults(payload, startMs, endMs) {
  const dets = Array.isArray(payload) ? payload : (payload?.detections || []);
  const processTimeSec =
    Array.isArray(payload)
      ? null
      : (payload.process_time_sec ?? (payload.process_time_ms ? (payload.process_time_ms / 1000) : null));

  const popup = document.getElementById('resultsPopup');
  const list = document.getElementById('popupResultsList');
  const closeBtn = document.getElementById('popupCloseBtn');

  list.innerHTML = '';

  const d = pickLargestDetection(dets);
  if (d) {
    const conf = Number(d?.confidence ?? 0);
    const sim  = Number(d?.similarity ?? d?.sim ?? 0);
    const name = (d?.identity && d.identity !== 'Unknown') ? d.identity : 'Unknown Elephant';

    const li = document.createElement('li');
    li.innerHTML = `
      <strong>${name}</strong><br>
      Confidence: ${(conf * 100).toFixed(1)}%<br>
      Similarity: ${sim.toFixed(2)}
    `;
    list.appendChild(li);
  }

  // Add timestamps above the processed-in line if available
  if (Number.isFinite(startMs) && Number.isFinite(endMs)) {
    const rangeEl = document.createElement('li');
    rangeEl.innerHTML = `<em>Start: ${formatDateTimeHundredths(startMs)} <br> End: ${formatDateTimeHundredths(endMs)}</em>`;
    list.appendChild(rangeEl);
  }

  if (processTimeSec !== null) {
    const timeEl = document.createElement('li');
    timeEl.innerHTML = `<em>⏱ Processed in ${processTimeSec.toFixed(2)} s</em>`;
    timeEl.style.textAlign = 'right';
    list.appendChild(timeEl);
  }

  popup.style.display = 'flex';
  popup.setAttribute('aria-hidden', 'false');

  closeBtn.onclick = () => {
    popup.style.display = 'none';
    popup.setAttribute('aria-hidden', 'true');
  };

  popup.addEventListener('click', e => {
    if (e.target === popup) {
      popup.style.display = 'none';
      popup.setAttribute('aria-hidden', 'true');
    }
  });
}

function showInfo(message, title = 'Heads up') {
  const overlay = document.getElementById('infoPopup');
  const t = document.getElementById('infoTitle');
  const txt = document.getElementById('infoText');
  const btn = document.getElementById('infoCloseBtn');

  t.textContent = title;
  txt.textContent = message;

  overlay.style.display = 'flex';
  overlay.setAttribute('aria-hidden', 'false');

  const close = () => {
    overlay.style.display = 'none';
    overlay.setAttribute('aria-hidden', 'true');
    btn.removeEventListener('click', close);
    overlay.removeEventListener('click', outside);
  };
  const outside = (e) => { if (e.target === overlay) close(); };

  btn.addEventListener('click', close);
  overlay.addEventListener('click', outside);
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

// --- main click handler ---
async function onPredictClick() {
  console.log('[UI] Predict clicked');
  hideList();
  clearCanvas();

  const file = fileUploader.files[0];
  if (!file) {
    showInfo('Please select a file before Predicting!', 'No file selected');
    return;
  }

  const isImage = file.type.startsWith('image/');
  const isVideo = file.type.startsWith('video/');
  if (!isImage && !isVideo) {
    showInfo('File format not supported.');
    return;
  }

  try {
    setLoading(true, 'Predicting...');
    let toSend = file;

    if (isVideo) {
      setLoading(true, 'Pulling frame from video...');
      toSend = await extractVideoFrame(file);
      await new Promise(resolve => {
        const reader = new FileReader();
        reader.onload = e => {
          imagePreview.src = e.target.result;
          imagePreview.onload = () => {
            imagePreview.style.display = 'block';
            ensureCanvas();
            clearCanvas();
            resolve();
          };
        };
        reader.readAsDataURL(toSend);
      });
      setLoading(true, 'Predicting...');
    } else if (isImage && (!imagePreview.src || imagePreview.src === '#')) {
      await new Promise(resolve => {
        const reader = new FileReader();
        reader.onload = e => {
          imagePreview.src = e.target.result;
          imagePreview.onload = () => {
            imagePreview.style.display = 'block';
            ensureCanvas();
            clearCanvas();
            resolve();
          };
        };
        reader.readAsDataURL(file);
      });
    }

    const tStart = Date.now();
    const json = await postImage(API_RUN_MAIN, toSend);
    const tEnd = Date.now();
    setLoading(false);

    const dets = json.detections || [];
    const noResultsMessage = document.getElementById('noResultsMessage');
    const noResultsPopup = document.getElementById('noResultsPopup');
    const noResultsCloseBtn = document.getElementById('noResultsCloseBtn');

    if (!dets.length) {
      setLoading(false);

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

    // Only show largest detection in popup, with timestamps
    showPopupResults(json, tStart, tEnd);

    // Drawing: only draw the largest detection
    const imgTmp = new Image();
    imgTmp.onload = () => {
      drawDetections(dets, imgTmp.naturalWidth, imgTmp.naturalHeight);
    };
    imgTmp.src = imagePreview.src;

  } catch (err) {
    console.error(err);
    showInfo('An error occurred during prediction');
  } finally {
    setLoading(false);
  }
}

