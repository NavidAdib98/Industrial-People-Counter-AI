# Industrial People Counter AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-purple.svg)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-ByteTrack-green.svg)](https://github.com/roboflow/supervision)

> **Real-time people counting inside a polygon ROI using YOLO detection, ByteTrack tracking, and multithreaded frame processing**

---

## Overview

Industrial People Counter AI is a computer vision system that counts people inside a defined polygon region of interest (ROI) in real time. It uses YOLO11 for detection, ByteTrack for tracking across frames, and a custom multithreaded pipeline to keep the display live even when inference is slow.

---

## Architecture

The system uses a **producer-consumer pattern** with 3 threads:

```
┌─────────────────┐     deque     ┌──────────────────┐     annotated     ┌──────────────┐
│  Reader Thread   │───frames────▶│ Processor Thread  │─────frame───────▶│ Main Display  │
│ (cv2.VideoCapture)│              │ (YOLO + ByteTrack)│                  │  (cv2.imshow) │
└─────────────────┘              └──────────────────┘                  └──────────────┘
```

- **Reader Thread** — reads frames at the video's native FPS, pushes them into a shared `deque` (max size 2)
- **Processor Thread** — pops the *latest* frame, runs YOLO detection → ByteTrack tracking → polygon check → event logging → drawing
- **Main Thread** — displays the annotated frame and handles keyboard input

---

## Features

- **Multithreaded processing** — separates frame capture from inference so the video never stalls
- **Pop + clear deque** — always processes the latest frame, discards backlog (low-latency display)
- **YOLO11n detection** — fast person detection with configurable confidence threshold
- **ByteTrack tracking** — Kalman filter + IoU matching for persistent track IDs across frames
- **Polygon ROI** — define any region via GeoJSON with normalized coordinates
- **3 anti-jitter techniques** — offset, hysteresis, and temporal filtering for stable boundary counting
- **Enter/Exit event logging** — CSV + JSON output with timestamps and occupancy
- **Annotated video output** — bounding boxes, track IDs, inside/outside status, and occupancy panel
- **Configuration via .env** — no code changes needed for different videos, models, or polygons

---

## How It Works

### 1. RealtimeProcessor (`src/core/realtime_processor.py`)

Two daemon threads with thread-safe locks:

| Component | Type | What it does |
|---|---|---|
| `_reader_loop` | `threading.Thread` | Opens `cv2.VideoCapture`, reads frames at video FPS, pushes `FrameData` into deque |
| `_processor_loop` | `threading.Thread` | Pops latest frame from deque, calls the processing callback (YOLO + tracking) |
| `frame_lock` | `threading.Lock()` | Protects deque push/pop/clear from race conditions |
| `latest_frame_lock` | `threading.Lock()` | Protects the shared latest frame for the display thread |

The processor uses **pop + clear** — it always grabs the most recent frame and discards older ones. This prevents latency from accumulating when the detector is slower than the video FPS.

### 2. PersonTracker (`src/core/tracker.py`)

Per-frame pipeline:

```
Frame → YOLO (class 0 only) → sv.Detections → ByteTrack.update_with_detections()
                                                      ↓
                                            For each tracked person:
                                              1. Compute reference point (center_x, y2 - offset)
                                              2. Check inside/outside with hysteresis
                                              3. Log ENTER/EXIT event if near boundary
                                                      ↓
                                            Return: detections + counts + metadata
```

**3 anti-jitter techniques:**

| Technique | What | Why |
|---|---|---|
| **Offset** | Reference point = `(center_x, y2 - 15px)` instead of raw bounding box center | Bounding boxes often extend below feet (shadows, partial detections); lifting the point avoids foot-boundary ambiguity |
| **Hysteresis** | Two asymmetric thresholds: need **15px inside** to switch to "inside", **25px outside** to switch to "outside" | Creates a dead zone around the boundary so small movements don't flip the count |
| **Temporal filtering** | Requires **3 consecutive frames** in the new state before confirming a change | Prevents single-frame glitches (motion blur, occlusion) from toggling status |

Events are only logged when the person is **within 40px of the polygon boundary** — far-away state changes update internal state silently.

### 3. EventLogger (`src/core/event_logger.py`)

Logs ENTER/EXIT events with:
- Track ID, frame number, video time (seconds), confidence
- Current occupancy after the event
- Saves to **CSV** and **JSON** files in `outputs/`

### 4. Visualizer (`src/visualization/visualizer.py`)

Draws on the frame:
- **Polygon** — semi-transparent fill with outline
- **Bounding boxes** — green (inside), red (outside), yellow (near boundary)
- **Track ID + status label** on each box
- **Info panel** — FPS, inside/outside/total counts
- **Legend** — color code reference

---

## Tracker Comparison

The code currently uses **Supervision's ByteTrack** (`sv.ByteTrack()`). Supervision also supports other trackers via the same `update_with_detections()` interface:

| Tracker | Association | Re-ID | Appearance | Inference cost | Camera motion | Occlusion handling | Key params |
|---|---|---|---|---|---|---|---|
| **ByteTrack** (used) | IoU only | No | None | **Very low** | ❌ Fails | Good — uses low-score detections too | `track_thresh`, `match_thresh`, `track_buffer` |
| **DeepSORT** | IoU + Re-ID cos-dist | Yes (CNN) | 128-d feature | High (+5-15ms) | ❌ Fails | Moderate — Re-ID helps after occlusion | `max_dist`, `nn_budget`, `max_age` |
| **BoT-SORT** | IoU + Re-ID + GMC | Yes (CNN) | Feature + Kalman + GMC | **Highest** (+GMC + Re-ID) | ✅ Handles via Global Motion Comp | Best — GMC + Re-ID + IoU | `gmc_method`, `track_high_thresh`, `new_track_thresh` |
| **OC-SORT** | IoU + OC | No | None | Low (no Re-ID) | ❌ Fails | Very good — virtual trajectory after occlusion | `det_thresh`, `max_age`, `min_hits` |
| **SORT** | IoU only | No | None | **Lowest** | ❌ Fails | Poor — loses track after occlusion | `max_age`, `min_hits`, `iou_threshold` |

**Key takeaways:**
- ByteTrack = best **speed/accuracy balance** for static-camera industrial scenes
- Switch to BoT-SORT if you have a **moving camera** (handles camera motion via GMC)
- Re-ID adds 5-50ms per frame — not worth it for real-time people counting unless occlusions are severe

---

## Input Files

| File | Location | Required | Auto-created |
|---|---|---|---|
| Video | `videos/your_video.mp4` | Yes | No |
| Polygon GeoJSON | `polygons/zone1.geojson` | Yes | No |
| .env | `.env` | Yes | Copy from `.env.example` |
| YOLO model | `models/yolo11n.pt` | No | Yes (auto-download) |

### Polygon Format (GeoJSON)

Coordinates are **normalized** (0.0 = left/top edge, 1.0 = right/bottom edge):

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {
      "name": "Main Zone",
      "color_inside": [0, 255, 0],
      "color_outside": [255, 0, 0],
      "alpha": 0.3
    },
    "geometry": {
      "type": "Polygon",
      "coordinates": [[
        [0.2, 0.1],
        [0.8, 0.1],
        [0.9, 0.9],
        [0.1, 0.9],
        [0.2, 0.1]
      ]]
    }
  }]
}
```

Pre-built examples: `polygons/zone1.geojson`, `polygons/main_zone.geojson`, `polygons/default_polygon.geojson`

### Output Files

```
outputs/
├── tracked_video.mp4      # Annotated video with bounding boxes and counts
├── events_YYYYMMDD_HHMMSS.csv   # Event log (CSV)
└── events_YYYYMMDD_HHMMSS.json  # Event log (JSON)
```

---

## Installation

### Prerequisites
- Python 3.10+
- `pip`
- OpenCV system libraries (`libgl1-mesa-glx`, `libglib2.0-0` on Linux)

### Step-by-step

```bash
# 1. Clone
git clone https://github.com/NavidAdib98/Industrial-People-Counter-AI.git
cd Industrial-People-Counter-AI

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/MacOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt   # ultralytics, supervision, python-dotenv

# 4. Configure environment
cp .env.example .env
# Edit .env with your video path, polygon file, and device

# 5. Run
python src/main.py
```

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `videos/Hike_Vision.mp4` | Path to video file (or `0` for webcam) |
| `POLYGON_FILE` | `polygons/zone1.geojson` | Path to GeoJSON polygon |
| `MODEL_PATH` | `models/yolo11n.pt` | YOLO model file |
| `CONF_THRESHOLD` | `0.4` | Detection confidence threshold |
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |
| `RESIZE_VIDEO` | `false` | Resize frames before processing |
| `SAVE_OUTPUT` | `true` | Save annotated output video |
| `SHOW_PREVIEW` | `true` | Show live OpenCV window |

---

## Usage

```bash
# Run with .env settings
python src/main.py

# With GPU
# Set DEVICE=cuda in .env
```

Press **`q`** or **`Esc`** to stop. The live window shows:

```
Read: 30 FPS | Target: 30 FPS | Process: 15 FPS | Queue: 0 | Inside: 5 | Outside: 3 | Total: 8
```

After completion, the annotated video and event logs are saved to `outputs/`.

---

## Project Structure

```
Industrial-People-Counter-AI/
├── .env                          # Configuration
├── .env.example                  # Example config
├── Dockerfile                    # Docker build (experimental)
├── requirements.txt              # Python dependencies
├── README.md
│
├── src/
│   ├── main.py                   # Entry point (RealtimeTracker)
│   │
│   ├── core/
│   │   ├── realtime_processor.py # Multithreaded frame reader + processor
│   │   ├── tracker.py            # YOLO detection + ByteTrack + polygon counting
│   │   └── event_logger.py       # ENTER/EXIT event logging (CSV/JSON)
│   │
│   ├── visualization/
│   │   └── visualizer.py         # OpenCV drawing (boxes, polygon, info panel)
│   │
│   └── utils/
│       ├── settings.py           # .env loader
│       └── polygon_loader.py     # GeoJSON parser
│
├── models/                       # YOLO model files
├── videos/                       # Input videos
├── polygons/                     # GeoJSON polygon files
└── outputs/                      # Annotated video + event logs
```

---

## Docker

A `Dockerfile` is included but **not tested for production use**. It installs OpenCV system dependencies and runs the application:

```bash
docker build -t people-counter .
docker run --rm \
  -v "$(pwd)"/videos:/app/videos \
  -v "$(pwd)"/polygons:/app/polygons \
  -v "$(pwd)"/outputs:/app/outputs \
  -v "$(pwd)"/.env:/app/.env \
  people-counter
```

GPU passthrough and production readiness are **not yet validated**.

---

## License

MIT