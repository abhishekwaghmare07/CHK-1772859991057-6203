const API_URL = "http://127.0.0.1:5000/predict";
const VIDEO_API_URL = "http://127.0.0.1:5000/predict/video";

const imageInput = document.getElementById("imageInput");
const predictBtn = document.getElementById("predictBtn");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const videoInput = document.getElementById("videoInput");
const predictVideoBtn = document.getElementById("predictVideoBtn");
const videoLoading = document.getElementById("videoLoading");
const videoResult = document.getElementById("videoResult");

async function sendPrediction({ file, endpoint, loadingEl, resultEl, buttonEl, typeLabel }) {
  if (!file) {
    resultEl.textContent = `Please choose a ${typeLabel} first.`;
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  buttonEl.disabled = true;
  loadingEl.classList.remove("hidden");
  resultEl.textContent = `Sending ${typeLabel} to backend...`;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      resultEl.textContent = `Error: ${data.error ?? "Prediction failed"}`;
      return;
    }

    const confidencePct = (Number(data.confidence || 0) * 100).toFixed(2);
    resultEl.textContent = [
      `Prediction: ${data.prediction}`,
      `Confidence: ${confidencePct}%`,
      `Raw Score: ${Number(data.raw_score || 0).toFixed(6)}`,
    ].join("\n");
  } catch (error) {
    resultEl.textContent = "Could not connect to backend. Make sure Flask server is running.";
  } finally {
    loadingEl.classList.add("hidden");
    buttonEl.disabled = false;
  }
}

predictBtn.addEventListener("click", async () => {
  const file = imageInput.files?.[0];
  await sendPrediction({
    file,
    endpoint: API_URL,
    loadingEl: loading,
    resultEl: result,
    buttonEl: predictBtn,
    typeLabel: "image",
  });
});

predictVideoBtn.addEventListener("click", async () => {
  const file = videoInput.files?.[0];
  await sendPrediction({
    file,
    endpoint: VIDEO_API_URL,
    loadingEl: videoLoading,
    resultEl: videoResult,
    buttonEl: predictVideoBtn,
    typeLabel: "video",
  });
});
