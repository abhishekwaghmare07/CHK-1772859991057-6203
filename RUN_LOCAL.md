# Run Locally

## Step 1: Start backend server

From `deepfake_detection/` run:

```powershell
python backend/app.py
```

## Step 2: Start frontend server

From `deepfake_detection/frontend/` run one of these:

```powershell
python -m http.server 5500
```

Or use the helper script that prints the URL clearly:

```powershell
python start_frontend_server.py
```

## Step 3: Open the frontend in browser

```text
http://localhost:5500
```

## API endpoints used by frontend

- `http://localhost:5000/predict/image`
- `http://localhost:5000/predict/video`
