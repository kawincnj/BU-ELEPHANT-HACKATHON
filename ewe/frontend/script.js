const API_RUN_MAIN = "http://localhost:8000/run-main";

// Elements
const openModalBtn = document.getElementById('openModalBtn');
const uploadModal = document.getElementById('uploadModal');
const closeModal = document.getElementById('closeModal');

const mainUploadArea = document.getElementById('mainUploadArea');
const fileUploader = document.getElementById('fileUploader');
const uploadTypeText = document.getElementById('uploadTypeText');
const imagePreview = document.getElementById('imagePreview');
const predictButton = document.getElementById('predictButton');

const statusRow = document.getElementById('statusRow');
const statusText = document.getElementById('statusText');
const resultsCard = document.getElementById('resultsCard');
const resultsList = document.getElementById('resultsList');

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

function drawDetections(dets, naturalW, naturalH) {
  ensureCanvas();
  clearCanvas();
  const scaleX = imagePreview.clientWidth / naturalW;
  const scaleY = imagePreview.clientHeight / naturalH;

  dets.forEach(d => {
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
  });
}

// Modal controls
function openModal() {
  uploadModal.style.display = 'flex';
  uploadModal.setAttribute('aria-hidden', 'false');
}
function closeModalFn() {
  uploadModal.style.display = 'none';
  uploadModal.setAttribute('aria-hidden', 'true');
}
openModalBtn?.addEventListener('click', openModal);
closeModal?.addEventListener('click', closeModalFn);
uploadModal?.addEventListener('click', (e) => {
  if (e.target === uploadModal) closeModalFn();
});

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
  statusRow.style.display = on ? 'block' : 'none';
  statusText.textContent = msg;
  predictButton.disabled = on;
}

function showList(dets) {
  resultsList.innerHTML = '';
  dets.forEach(d => {
    const li = document.createElement('li');
    li.textContent = d.overlay_text || `${d.identity} (${(d.confidence*100).toFixed(1)}%) | Sim: ${d.similarity.toFixed(2)}`;
    resultsList.appendChild(li);
  });
  resultsCard.style.display = 'block';
}

function hideList() {
  resultsCard.style.display = 'none';
  resultsList.innerHTML = '';
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

predictButton.addEventListener('click', async () => {
  hideList();
  clearCanvas();

  const file = fileUploader.files[0];
  if (!file) {
    alert('Please select a file before Predicting!');
    return;
  }

  const isImage = file.type.startsWith('image/');
  const isVideo = file.type.startsWith('video/');
  if (!isImage && !isVideo) {
    alert('File format not supported.');
    return;
  }

  try {
    setLoading(true, 'Predicting...');

    let toSend = file;
    if (isVideo) {
      setLoading(true, 'Pulling frame from video...');
      toSend = await extractVideoFrame(file);

      // Show the extracted frame as preview
      await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
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
      // Ensure preview exists for images even if user skipped viewing
      await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
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

    const json = await postImage(API_RUN_MAIN, toSend);
    const dets = json.detections || [];
    setLoading(false);

    if (!dets.length) {
      alert('No results found');
      return;
    }

    showList(dets);

    const imgTmp = new Image();
    imgTmp.onload = () => {
      drawDetections(dets, imgTmp.naturalWidth, imgTmp.naturalHeight);
    };
    imgTmp.src = imagePreview.src;

    // Close modal after starting prediction (optional)
    closeModalFn();

  } catch (err) {
    console.error(err);
    setLoading(false);
    alert('An error occurred during prediction');
  }
});
