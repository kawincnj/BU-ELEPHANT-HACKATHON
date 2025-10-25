const API_RUN_MAIN = "http://localhost:8000/run-main";

const selectImageButton = document.getElementById("selectImage");
const selectVideoButton = document.getElementById("selectVideo");
const mainUploadArea = document.getElementById("mainUploadArea");
const fileUploader = document.getElementById("fileUploader");
const uploadTypeText = document.getElementById("uploadTypeText");
const imagePreview = document.getElementById("imagePreview");
const predictButton = document.getElementById("predictButton");

const statusRow = document.getElementById("statusRow");
const statusText = document.getElementById("statusText");
const resultsCard = document.getElementById("resultsCard");
const resultsList = document.getElementById("resultsList");
const currentDateTimeElement = document.getElementById("currentDateTime");

let currentFileType = null;
let overlayCanvas, overlayCtx;

// --- Variables and Time Tracking ---
let timerStart = null;
let timerInterval = null;
let lastElapsed = null;
let startTimeDisplay = null;
let endTimeDisplay = null;

function startTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  timerStart = performance.now();
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
// Helper function to format Date object into d/m/y HH:MM:SS.SSS
function formatDateTime(date) {
  const day = String(date.getDate()).padStart(2, "0");
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const year = date.getFullYear();

  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");
  const milliseconds = String(date.getMilliseconds()).padStart(3, "0");

  return `${day}/${month}/${year} ${hours}:${minutes}:${seconds}.${milliseconds}`;
}

function updateDateTime() {
  const now = new Date();
  if (currentDateTimeElement) {
    currentDateTimeElement.textContent = `Current Time: ${formatDateTime(now)}`;
  }
}

// Start updating time every second
updateDateTime();
setInterval(updateDateTime, 1000);

// --- Canvas and Drawing Functions ---
function ensureCanvas() {
  if (!overlayCanvas) {
    overlayCanvas = document.createElement("canvas");
    overlayCanvas.style.position = "absolute";
    overlayCanvas.style.left = "0";
    overlayCanvas.style.top = "0";
    overlayCanvas.style.pointerEvents = "none";
    overlayCanvas.id = "overlay";
    imagePreview.parentElement.style.position = "relative";
    imagePreview.parentElement.appendChild(overlayCanvas);
  }
  overlayCanvas.width = imagePreview.clientWidth;
  overlayCanvas.height = imagePreview.clientHeight;
  overlayCtx = overlayCanvas.getContext("2d");
}

function clearCanvas() {
  if (overlayCtx && overlayCanvas)
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function drawDetections(dets, naturalW, naturalH) {
  ensureCanvas();
  clearCanvas();
  const scaleX = imagePreview.clientWidth / naturalW;
  const scaleY = imagePreview.clientHeight / naturalH;

  dets.forEach((d) => {
    const [x1, y1, x2, y2] = d.bbox;
    const rx1 = Math.round(x1 * scaleX);
    const ry1 = Math.round(y1 * scaleY);
    const rx2 = Math.round(x2 * scaleX);
    const ry2 = Math.round(y2 * scaleY);
    const w = rx2 - rx1;
    const h = ry2 - ry1;

    // Box
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeStyle = "rgba(0, 255, 0, 0.85)";
    overlayCtx.strokeRect(rx1, ry1, w, h);

    // Label (3 decimal places)
    const label =
      d.overlay_text || `${d.identity} (${(d.confidence * 100).toFixed(3)}%)`;
    overlayCtx.font = "14px Arial";
    const textW = overlayCtx.measureText(label).width + 8;
    const textH = 18;
    overlayCtx.fillStyle = "rgba(0,0,0,0.6)";
    overlayCtx.fillRect(rx1, Math.max(0, ry1 - textH), textW, textH);
    overlayCtx.fillStyle = "white";
    overlayCtx.fillText(label, rx1 + 4, ry1 - 4);
  });
}

// --- File Selection and UI Management ---
function setFileType(type) {
  currentFileType = type;
  mainUploadArea.style.display = "block";
  imagePreview.style.display = "none";
  fileUploader.value = "";
  hideList();

  if (type === "image") {
    selectImageButton.classList.add("active");
    selectVideoButton.classList.remove("active");
    uploadTypeText.textContent = "เลือกไฟล์รูปภาพ (Image)";
    fileUploader.setAttribute("accept", "image/*");
  } else {
    selectImageButton.classList.remove("active");
    selectVideoButton.classList.add("active");
    uploadTypeText.textContent = "เลือกไฟล์วิดีโอ (Video)";
    fileUploader.setAttribute("accept", "video/*");
  }
}

selectImageButton.addEventListener("click", () => setFileType("image"));
selectVideoButton.addEventListener("click", () => setFileType("video"));

fileUploader.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file && currentFileType === "image") {
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.onload = () => {
        imagePreview.style.display = "block";
        ensureCanvas();
        clearCanvas();
      };
    };
    reader.readAsDataURL(file);
  } else {
    imagePreview.src = "#";
    imagePreview.style.display = "none";
    clearCanvas();
  }
  hideList();
});

function setLoading(on, msg = "กำลังประมวลผล…") {
  statusRow.style.display = on ? "block" : "none";
  statusText.textContent = msg;
  predictButton.disabled = on;
  if (!on) {
    stopTimer();
  }
}

// --- Result Display Logic (Show the requested format) ---
function showList(dets) {
  resultsList.innerHTML = "";

  // Create a helper function for a detail row
  const createDetailRow = (label, value, className) => {
    const row = document.createElement("div");
    row.className = `detail-row ${className || ""}`;
    row.innerHTML = `<span class="detail-label">${label}</span> <span class="detail-value">${value}</span>`;
    return row;
  };

  // 1. Loop through detections and show detailed results in the requested format
  dets.forEach((d, index) => {
    // Get the specific name (e.g., "Elephant: IKO")
    const identityName = d.identity || "Unknown Elephant";

    // Predict Class value is hardcoded to "elephant"
    const predictedClassValue = "elephant";

    const li = document.createElement("li");
    li.className = "result-item detailed-result";

    // Title: Detection Number and Identity
    const title = document.createElement("div");
    title.className = "ri-title";
    title.textContent = `Detection ${index + 1}: ${identityName}`;
    li.appendChild(title);

    // Main detail table container
    const fieldsContainer = document.createElement("div");
    fieldsContainer.className = "ri-details-table";

    // Row 1: Test Class / Identity
    fieldsContainer.appendChild(
      createDetailRow("Predicted Class:", identityName)
    );

    // Row 2: Predict Class (HARDCODED TO 'elephant')
    // fieldsContainer.appendChild(createDetailRow('Predict Class:', predictedClassValue));

    // COMMENTED OUT: Row 3: Confidence (3 decimal places) - Removed as requested
    /*
    fieldsContainer.appendChild(createDetailRow('Confidence:', `${(d.confidence*100).toFixed(3)}%`, 'confidence-row'));
    */

    // Row 3 (was 4): Sim Confidence (Similarity) - Now displayed as a percentage (x100)
    if (d.similarity !== undefined) {
      fieldsContainer.appendChild(
        createDetailRow(
          "Confidence:",
          `${(d.similarity * 100).toFixed(3)}%`,
          "similarity-row"
        )
      );
    }

    // --- Time details are added only to the FIRST result item for clarity ---
    if (index === 0) {
      // Row 4 (was 5): Start Time (with milliseconds)
      fieldsContainer.appendChild(
        createDetailRow("Start Time:", startTimeDisplay, "time-row")
      );

      // Row 5 (was 6): End Time (with milliseconds)
      fieldsContainer.appendChild(
        createDetailRow("End Time:", endTimeDisplay, "time-row")
      );

      // Row 6 (was 7): Time Elapsed (3 decimal places)
      fieldsContainer.appendChild(
        createDetailRow(
          "Time using:",
          `${lastElapsed ? lastElapsed.toFixed(3) : "0.000"} second`,
          "elapsed-time-row"
        )
      );
    }

    li.appendChild(fieldsContainer);
    resultsList.appendChild(li);
  });

  resultsCard.style.display = "block";
}

function hideList() {
  resultsCard.style.display = "none";
  resultsList.innerHTML = "";
  startTimeDisplay = null;
  endTimeDisplay = null;
}

// --- API and Video Handling ---
async function postImage(url, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function extractVideoFrame(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.src = URL.createObjectURL(file);
    video.onloadeddata = () => {
      const t = Math.min(Math.max(video.duration / 2, 0), video.duration || 0);
      video.currentTime = t;
    };
    video.onseeked = () => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);
      canvas.toBlob(
        (blob) => {
          if (!blob) return reject(new Error("Failed to extract frame"));
          resolve(new File([blob], "frame.jpg", { type: "image/jpeg" }));
        },
        "image/jpeg",
        0.9
      );
    };
    video.onerror = () => reject(new Error("Cannot load video"));
  });
}

// --- Predict Button Handler ---
predictButton.addEventListener("click", async () => {
  hideList();
  clearCanvas();

  if (!currentFileType) {
    alert("กรุณาเลือกประเภทไฟล์ก่อน (Image หรือ Video)!");
    return;
  }
  const file = fileUploader.files[0];
  if (!file) {
    alert("กรุณาเลือกไฟล์ก่อนกด Predict!");
    return;
  }

  try {
    // CAPTURE START TIME (now with milliseconds)
    startTimeDisplay = formatDateTime(new Date());
    startTimer();
    setLoading(true, "กำลังประมวลผล…");

    let toSend = file;
    if (currentFileType === "video") {
      setLoading(true, "กำลังดึงเฟรมจากวิดีโอ…");
      toSend = await extractVideoFrame(file);
      setLoading(true, "กำลังประมวลผล…");
    }

    const json = await postImage(API_RUN_MAIN, toSend);

    // CAPTURE END TIME (now with milliseconds)
    endTimeDisplay = formatDateTime(new Date());

    const dets = json.detections || [];
    setLoading(false);

    if (!dets.length) {
      alert("ไม่พบผลลัพธ์");
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
    // Ensure end time is captured even on error
    endTimeDisplay = formatDateTime(new Date());
    setLoading(false);
    alert("เกิดข้อผิดพลาดระหว่างการทำนาย");
  }
});
