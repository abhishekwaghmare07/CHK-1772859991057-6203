# AI Deepfake Detection Web Application

## Project Overview
This project is a Flask-based web application for deepfake detection. Users upload images or videos from the browser, the backend runs model inference using trained TensorFlow models, and the UI shows predicted class with confidence.

## System Architecture
1. Frontend (HTML/CSS/JavaScript)
2. Backend API (Flask)
3. Trained model files (`backend/deepfake_detector_model.h5` and `backend/deepfake_video_detector_model.h5`)

### Request Flow
1. User selects an image or video in the web UI.
2. Frontend sends the file to `POST /predict` (image) or `POST /predict/video` (video).
3. Backend preprocesses media and runs inference.
4. API returns JSON prediction result.
5. Frontend renders prediction and confidence.

## Technologies Used
- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- OpenCV
- HTML5, CSS3, JavaScript (Fetch API)

## Project Structure
```text
deepfake-project/
├── backend/
│   ├── app.py
│   ├── deepfake_detector_model.h5
│   └── deepfake_video_detector_model.h5
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
└── README.md
```

## How To Run Locally
1. Open a terminal in the project root.
2. Create and activate virtual environment.
3. Install dependencies.
4. Start backend.
5. Start frontend static server.
6. Open browser and test.

### Commands (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend/app.py
```

In a second terminal:
```powershell
cd frontend
python -m http.server 8080 --bind 127.0.0.1
```

Open:
- Frontend: http://127.0.0.1:8080
- Backend health: http://127.0.0.1:5000/health

## API Endpoint
### POST /predict
- Content-Type: multipart/form-data
- Field name: `file`
- Label mapping used by backend: `Fake=0`, `Real=1`
- Decision threshold: `score >= 0.6 => Real`, else `Fake`

### POST /predict/video
- Content-Type: multipart/form-data
- Field name: `file`
- Label mapping used by backend: `Fake=0`, `Real=1`
- Decision threshold: `score >= 0.5 => Real`, else `Fake`

### Example Response
```json
{
  "prediction": "Real",
  "confidence": 0.98,
  "raw_score": 0.02
}
```

## Screenshots
Add your screenshots to `assets/screenshots/` and update links below:
- Home UI: `assets/screenshots/home.png`
- Prediction result: `assets/screenshots/prediction.png`

## Example Prediction Output
```text
Prediction: Fake
Confidence: 93.41%
Raw Score: 0.934100
```

## Notes
- This setup supports image and video detection.
- Ensure both model files are present before starting the backend.
