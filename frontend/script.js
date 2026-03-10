const API_URL = "http://127.0.0.1:5000/predict";

const imageInput = document.getElementById("imageInput");
const predictBtn = document.getElementById("predictBtn");
const loading = document.getElementById("loading");
const result = document.getElementById("result");

predictBtn.addEventListener("click", async () => {
  const file = imageInput.files?.[0];
  if (!file) {
    result.textContent = "Please choose an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  predictBtn.disabled = true;
  loading.classList.remove("hidden");
  result.textContent = "Sending image to backend...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      result.textContent = `Error: ${data.error ?? "Prediction failed"}`;
      return;
    }

    const confidencePct = (Number(data.confidence || 0) * 100).toFixed(2);
    result.textContent = [
      `Prediction: ${data.prediction}`,
      `Confidence: ${confidencePct}%`,
      `Raw Score: ${Number(data.raw_score || 0).toFixed(6)}`,
    ].join("\n");
  } catch (error) {
    result.textContent = "Could not connect to backend. Make sure Flask server is running.";
  } finally {
    loading.classList.add("hidden");
    predictBtn.disabled = false;
  }
});
