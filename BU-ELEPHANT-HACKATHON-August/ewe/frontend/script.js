const API_RUN_MAIN = "http://localhost:8000/run-main";

const selectImageButton = document.getElementById('selectImage');
const selectVideoButton = document.getElementById('selectVideo');
const mainUploadArea = document.getElementById('mainUploadArea');
const fileUploader = document.getElementById('fileUploader');
const uploadTypeText = document.getElementById('uploadTypeText');
const imagePreview = document.getElementById('imagePreview');
const predictButton = document.getElementById('predictButton');

const statusRow = document.getElementById('statusRow');
const statusText = document.getElementById('statusText');
const resultsCard = document.getElementById('resultsCard');
const resultsList = document.getElementById('resultsList');

let currentFileType = null;
let overlayCanvas, overlayCtx;

// --- เพิ่มตัวแปรและฟังก์ชันจับเวลา ---
let timerStart = null;
let timerInterval = null;
let lastElapsed = null;

function startTimer() {
	// เริ่มนับเวลา (ความละเอียด ms)
	if (timerInterval) {
		clearInterval(timerInterval);
		timerInterval = null;
	}
	timerStart = performance.now();
	// อัพเดต lastElapsed ระหว่างทำงาน (แต่ไม่แสดงทุกครั้ง เพื่อไม่รบกวน UI)
	timerInterval = setInterval(() => {
		lastElapsed = (performance.now() - timerStart) / 1000;
	}, 100);
}

function stopTimer() {
	if (!timerStart) return null;
	lastElapsed = (performance.now() - timerStart) / 1000;
	if (timerInterval) {
		clearInterval(timerInterval);
		timerInterval = null;
	}
	timerStart = null;
	return lastElapsed;
}
// --- จบการเพิ่มตัวจับเวลา ---

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

function setFileType(type) {
  currentFileType = type;
  mainUploadArea.style.display = 'block';
  imagePreview.style.display = 'none';
  fileUploader.value = '';

  if (type === 'image') {
    selectImageButton.classList.add('active');
    selectVideoButton.classList.remove('active');
    uploadTypeText.textContent = 'เลือกไฟล์รูปภาพ (Image)';
    fileUploader.setAttribute('accept', 'image/*');
  } else {
    selectImageButton.classList.remove('active');
    selectVideoButton.classList.add('active');
    uploadTypeText.textContent = 'เลือกไฟล์วิดีโอ (Video)';
    fileUploader.setAttribute('accept', 'video/*');
  }
}

selectImageButton.addEventListener('click', () => setFileType('image'));
selectVideoButton.addEventListener('click', () => setFileType('video'));

fileUploader.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file && currentFileType === 'image') {
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
    imagePreview.src = '#';
    imagePreview.style.display = 'none';
    clearCanvas();
  }
});

function setLoading(on, msg = 'กำลังประมวลผล…') {
  statusRow.style.display = on ? 'block' : 'none';
  statusText.textContent = msg;
  predictButton.disabled = on;
  if (!on) {
    // หยุดตัวจับเวลาเมื่อลบสถานะ loading (จะเก็บค่าไว้ที่ lastElapsed)
    stopTimer();
  }
}

function showList(dets) {
  resultsList.innerHTML = '';
  dets.forEach(d => {
    // สร้างกล่องผลลัพธ์แต่ละชิ้น
    const li = document.createElement('li');
    li.className = 'result-item';

    const title = document.createElement('div');
    title.className = 'ri-title';
    title.textContent = d.overlay_text || d.identity || 'Unknown';

    const fields = document.createElement('div');
    fields.className = 'ri-fields';

    const conf = document.createElement('div');
    conf.className = 'ri-field';
    conf.innerHTML = `<span class="ri-label">ความถูกต้อง:</span> <span class="ri-val">${(d.confidence*100).toFixed(1)}%</span>`;

    const sim = document.createElement('div');
    sim.className = 'ri-field';
    sim.innerHTML = `<span class="ri-label">Sim:</span> <span class="ri-val">${d.similarity !== undefined ? d.similarity.toFixed(2) : 'N/A'}</span>`;

    const time = document.createElement('div');
    time.className = 'ri-time';
    time.innerHTML = `<span class="ri-label">เวลา:</span> <span class="ri-val">${lastElapsed ? lastElapsed.toFixed(2) : '0.00'} วินาที</span>`;

    fields.appendChild(conf);
    fields.appendChild(sim);
    fields.appendChild(time);

    li.appendChild(title);
    li.appendChild(fields);
    resultsList.appendChild(li);
  });

  resultsCard.style.display = 'block';
}

function hideList() {
  resultsCard.style.display = 'none';
  resultsList.innerHTML = '';
  // ลบ element แสดงเวลาเมื่อซ่อนผลลัพธ์
  const timeEl = document.getElementById('elapsedTime');
  if (timeEl) timeEl.remove();
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

  if (!currentFileType) {
    alert('กรุณาเลือกประเภทไฟล์ก่อน (Image หรือ Video)!');
    return;
  }
  const file = fileUploader.files[0];
  if (!file) {
    alert('กรุณาเลือกไฟล์ก่อนกด Predict!');
    return;
  }

  try {
    // เริ่มจับเวลาเมื่อผู้ใช้กด Predict
    startTimer();
    setLoading(true, 'กำลังประมวลผล…');

    let toSend = file;
    if (currentFileType === 'video') {
      setLoading(true, 'กำลังดึงเฟรมจากวิดีโอ…');
      toSend = await extractVideoFrame(file);
      setLoading(true, 'กำลังประมวลผล…');
    }

    const json = await postImage(API_RUN_MAIN, toSend);
    const dets = json.detections || [];
    setLoading(false);

    if (!dets.length) {
      alert('ไม่พบผลลัพธ์');
      return;
    }

    showList(dets);

    const imgTmp = new Image();
    imgTmp.onload = () => {
      drawDetections(dets, imgTmp.naturalWidth, imgTmp.naturalHeight);
    };
    imgTmp.src = imagePreview.src;

  } catch (err) {
    console.error(err);
    setLoading(false);
    alert('เกิดข้อผิดพลาดระหว่างการทำนาย');
  }
});
