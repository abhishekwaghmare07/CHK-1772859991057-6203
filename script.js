const imageInput = document.getElementById("imageFile") || document.getElementById("imageInput");
const videoInput = document.getElementById("videoFile") || document.getElementById("videoInput");

const imageDetectBtn = document.getElementById("imageDetectBtn");
const videoDetectBtn = document.getElementById("videoDetectBtn");
const goImageBtn = document.getElementById("goImageBtn");
const goVideoBtn = document.getElementById("goVideoBtn");
const goWorkflowBtn = document.getElementById("goWorkflowBtn");
const goArchitectureBtn = document.getElementById("goArchitectureBtn");
const startDetectionBtn = document.getElementById("startDetectionBtn");
const backHomeBtn = document.getElementById("backHomeBtn");

const imageLoading = document.getElementById("imageLoading");
const videoLoading = document.getElementById("videoLoading");
const imageFileName = document.getElementById("imageFileName");
const videoFileName = document.getElementById("videoFileName");

const imageResult = document.getElementById("imageResult");
const videoResult = document.getElementById("videoResult");
const imagePrediction = document.getElementById("imagePrediction");
const videoPrediction = document.getElementById("videoPrediction");
const imageConfidenceText = document.getElementById("imageConfidenceText");
const videoConfidenceText = document.getElementById("videoConfidenceText");
const imageProgress = document.getElementById("imageProgress");
const videoProgress = document.getElementById("videoProgress");

const API_BASE = "http://127.0.0.1:5000";
const IMAGE_API = `${API_BASE}/predict/image`;
const VIDEO_API = `${API_BASE}/predict/video`;

function goToImagePage() {
  window.location.href = "image.html";
}

function goToVideoPage() {
  window.location.href = "video.html";
}

function goToWorkflowPage() {
  window.location.href = "model_workflow.html";
}

function goToArchitecturePage() {
  window.location.href = "model_architecture.html";
}

function goToDetectionOptionsPage() {
  window.location.href = "detection_options.html";
}

function goToHomePage() {
  window.location.href = "index.html";
}

function previewImage(event) {
  const file = event.target.files[0];
  const preview = document.getElementById("imagePreview");

  if (!preview) {
    return;
  }

  if (file) {
    const reader = new FileReader();

    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.style.display = "block";
    };

    reader.readAsDataURL(file);
  } else {
    preview.style.display = "none";
    preview.removeAttribute("src");
  }
}

function previewVideo(event) {
  const file = event.target.files[0];
  const preview = document.getElementById("videoPreview");

  if (!preview) {
    return;
  }

  if (file) {
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.style.display = "block";
  } else {
    preview.style.display = "none";
    preview.removeAttribute("src");
  }
}

function setOutputState(resultBox, predictionEl, confidenceEl, progressEl, prediction, confidence) {
  const normalized = String(prediction || "Unknown").toUpperCase();
  const confidenceValue = Number.isFinite(confidence) ? confidence : 0;
  const confidencePct = Math.max(0, Math.min(100, confidenceValue * 100));

  resultBox.classList.remove("result-neutral", "result-real", "result-fake");
  if (normalized === "REAL") {
    resultBox.classList.add("result-real");
  } else if (normalized === "FAKE") {
    resultBox.classList.add("result-fake");
  } else {
    resultBox.classList.add("result-neutral");
  }

  predictionEl.textContent = normalized;
  confidenceEl.textContent = `Confidence: ${confidencePct.toFixed(2)}%`;
  progressEl.style.width = `${confidencePct.toFixed(2)}%`;
}

async function sendForPrediction({ file, endpoint, loadingEl, buttonEl, resultBox, predictionEl, confidenceEl, progressEl, typeLabel }) {
  if (!file) {
    predictionEl.textContent = "NO FILE";
    confidenceEl.textContent = `Confidence: --`;
    progressEl.style.width = "0%";
    resultBox.classList.remove("result-real", "result-fake");
    resultBox.classList.add("result-neutral");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  loadingEl.classList.remove("hidden");
  buttonEl.disabled = true;

  try {
    console.log("Connecting to backend:", endpoint);

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      predictionEl.textContent = "ERROR";
      confidenceEl.textContent = data.error ? `Confidence: ${data.error}` : "Confidence: Request failed";
      progressEl.style.width = "0%";
      resultBox.classList.remove("result-real");
      resultBox.classList.add("result-fake");
      return;
    }

    setOutputState(
      resultBox,
      predictionEl,
      confidenceEl,
      progressEl,
      data.prediction,
      Number(data.confidence)
    );
  } catch (error) {
    console.error("Backend connection failed:", error);
    predictionEl.textContent = `${typeLabel.toUpperCase()} API ERROR`;
    confidenceEl.textContent = "Confidence: Backend not reachable. Start Flask server first.";
    progressEl.style.width = "0%";
    resultBox.classList.remove("result-real");
    resultBox.classList.add("result-fake");
  } finally {
    loadingEl.classList.add("hidden");
    buttonEl.disabled = false;
  }
}

function detectImage() {
  const file = imageInput?.files?.[0];
  sendForPrediction({
    file,
    endpoint: IMAGE_API,
    loadingEl: imageLoading,
    buttonEl: imageDetectBtn,
    resultBox: imageResult,
    predictionEl: imagePrediction,
    confidenceEl: imageConfidenceText,
    progressEl: imageProgress,
    typeLabel: "image",
  });
}

function detectVideo() {
  const file = videoInput?.files?.[0];
  sendForPrediction({
    file,
    endpoint: VIDEO_API,
    loadingEl: videoLoading,
    buttonEl: videoDetectBtn,
    resultBox: videoResult,
    predictionEl: videoPrediction,
    confidenceEl: videoConfidenceText,
    progressEl: videoProgress,
    typeLabel: "video",
  });
}

if (goImageBtn) {
  goImageBtn.addEventListener("click", goToImagePage);
}

if (goVideoBtn) {
  goVideoBtn.addEventListener("click", goToVideoPage);
}

if (goWorkflowBtn) {
  goWorkflowBtn.addEventListener("click", goToWorkflowPage);
}

if (goArchitectureBtn) {
  goArchitectureBtn.addEventListener("click", goToArchitecturePage);
}

if (startDetectionBtn) {
  startDetectionBtn.addEventListener("click", goToDetectionOptionsPage);
}

if (backHomeBtn) {
  backHomeBtn.addEventListener("click", goToHomePage);
}

if (imageInput && imageFileName) {
  imageInput.addEventListener("change", (event) => {
    imageFileName.textContent = event.target.files[0]?.name ?? "No file selected";
  });
}

if (videoInput && videoFileName) {
  videoInput.addEventListener("change", (event) => {
    videoFileName.textContent = event.target.files[0]?.name ?? "No file selected";
  });
}

if (imageDetectBtn) {
  imageDetectBtn.addEventListener("click", detectImage);
}

if (videoDetectBtn) {
  videoDetectBtn.addEventListener("click", detectVideo);
}
