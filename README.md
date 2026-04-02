# 🔤 Real-Time OCR

> A blazing-fast, browser-based OCR system powered by custom trained **YOLOv8** (text detection) + **CRNN/PaddleOCR** (text recognition), running entirely on-device via the **MNN inference engine** — no cloud, no API keys.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MNN-on--device-orange" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📷 **Live Webcam OCR** | Point your camera at any text — bounding boxes and labels appear in real time |
| 🖼️ **Image Upload OCR** | Upload any `.jpg` / `.png` and get instant text extraction |
| 🧠 **On-Device AI** | YOLOv8 + CRNN run locally via MNN — zero cloud calls |
| ⚡ **WebSocket Streaming** | Binary JPEG frames over WebSocket for ultra-low latency |
| 🎯 **IoU Object Tracking** | Bounding boxes are tracked across frames for stable, flicker-free results |
| 🔒 **Privacy-First** | Everything stays on your machine |

---

## 🏗️ Architecture

```
Browser (index.html)
    │
    │  WebSocket — binary JPEG frames @ configurable FPS
    ▼
FastAPI Server (app.py)
    │
    ├── YOLOv8 (best.mnn)       → detects text regions → bounding boxes
    ├── CRNN / PaddleOCR        → reads text inside each bounding box
    │   (rec_crnn.mnn)
    └── IoU Tracker             → matches boxes across frames for stability
    │
    │  JSON response  [{x1, y1, x2, y2, text}, ...]
    ▼
Browser → draws bounding boxes + labels on <canvas> overlay
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/bhaskar345/Real-Time-OCR.git
cd Real-Time-OCR
```

### 2. Create & activate a virtual environment

```bash
python -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**

| Package | Purpose |
|---|---|
| `fastapi` | Web framework & API routing |
| `uvicorn[standard]` | ASGI server with WebSocket support |
| `opencv-python` | Frame capture & image preprocessing |
| `numpy` | Tensor / array operations |
| `MNN` | Lightweight on-device neural network inference |
| `starlette<0.37` | Required version pin for FastAPI compatibility |

---

## 🚀 Running the App

```bash
python app.py
```

Open your browser and navigate to:

```
http://localhost:8000
```

The server binds to `0.0.0.0:8000`, so it's accessible from any device on your local network (e.g. `http://192.168.x.x:8000` from your phone).

---

## 🖥️ Usage

### 📷 Webcam Mode (Real-Time)

1. Click **"📷 Enable Camera"**
2. Grant camera permission when the browser prompts
3. Point your camera at any printed or on-screen text
4. Green bounding boxes with recognized text overlay appear live on the canvas
5. All detections are logged in the **"🧠 Detected Text"** feed below
6. Click **"🚫 Disable Camera"** to stop

### 🖼️ Image Upload Mode

1. Click **"Choose File"** under *Upload Image for OCR*
2. Select any image (`.jpg`, `.png`, etc.)
3. The image is POSTed to `/upload_frame`, processed server-side, and results are drawn immediately on the canvas

> Selecting a file automatically stops the webcam if it was running.

### 📋 Detection Feed

The **"🧠 Detected Text"** panel logs every recognized text string with a timestamp, updating in real time. New detections are prepended to the top.

---

## ⚙️ Configuration & Tuning

### 🎛️ Adjusting FPS (Capture Rate)

Open **`templates/index.html`** and find **line 218**:

```js
setTimeout(loop, 250); // 4 FPS
```

Change `250` (milliseconds per frame) to control the capture rate:

| `setTimeout` value | FPS | Recommended For |
|---|---|---|
| `1000` | 1 | Minimal CPU / battery saving |
| `500` | 2 | Low-power devices |
| `250` | **4** | **Default — balanced** |
| `100` | 10 | High responsiveness |
| `50` | 20 | Maximum smoothness (CPU-heavy) |

---

### 🔁 OCR Interval — `app.py`

OCR doesn't run on every single frame — it runs every `OCR_INTERVAL` frames to save CPU:

```python
OCR_INTERVAL = 2  # in app.py — default is FPS / 2
```

**The rule of thumb:**

```
OCR_INTERVAL = FPS / 2
```

| FPS (index.html) | Recommended OCR_INTERVAL (app.py) | OCR runs at |
|---|---|---|
| 2 | 1 | every frame |
| 4 | **2** | **every 2nd frame (default)** |
| 10 | 5 | every 5th frame |
| 20 | 10 | every 10th frame |

This keeps OCR running at roughly **2 recognitions/second** regardless of FPS, balancing accuracy with CPU load.

---

### 🔢 Max Concurrent OCR Crops — `app.py`

```python
MAX_OCR = 3  # maximum text regions OCR'd per frame
```

- **Increase** for dense-text scenes (menus, documents, signs)
- **Decrease** for better performance on low-end hardware

---

## 🛠️ Troubleshooting

**Camera not working?**
- Ensure your browser has camera permission for `localhost`
- Chrome: `chrome://settings/content/camera`
- Firefox: click the 🔒 lock icon in the address bar

**Slow inference / high CPU?**
- Lower the FPS in `templates/index.html` line 218 (e.g. `500` for 2 FPS)
- Reduce `MAX_OCR` in `app.py` (e.g. `MAX_OCR = 1`)
- MNN uses CPU by default; GPU backends can be enabled via MNN session config

**No text detected?**
- Ensure good lighting — text should be well-lit and in focus
- Text regions smaller than **40×20 pixels** are filtered out (too small to read reliably)
- Try the image upload mode to test with a static image first

**WebSocket keeps disconnecting?**
- The client auto-reconnects with exponential backoff (1s → 5s max)
- Check the server terminal for Python tracebacks
- Make sure `app.py` is still running

**`ModuleNotFoundError: No module named 'MNN'`?**
- Activate your virtual environment first: `env\Scripts\activate` (Windows) or `source env/bin/activate`
- Then re-run: `pip install -r requirements.txt`

---