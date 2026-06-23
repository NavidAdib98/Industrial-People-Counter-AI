# Industrial Occupancy Tracker

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-purple.svg)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-ByteTrack-green.svg)](https://github.com/roboflow/supervision)

> **Real-time people counting system for industrial environments with advanced occlusion handling and boundary hysteresis**

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Input & Output](#-input--output)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Tracking & Occlusion Handling](#-tracking--occlusion-handling)
- [Project Structure](#-project-structure)
- [Performance Metrics](#-performance-metrics)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Overview

Industrial Occupancy Tracker is a computer vision system designed to count people in industrial environments using YOLOv8 detection and ByteTrack tracking. The system handles challenging scenarios including:

- 🚶 **Occlusion** - People partially hidden behind objects or each other
- 🎯 **Boundary tracking** - Accurate counting when people cross ROI boundaries
- 🔄 **Enter/Exit detection** - Tracks when people enter or leave the monitored area
- 📊 **Real-time occupancy** - Current count in the region of interest

## ✨ Features

- ✅ **Real-time person detection** using YOLO11n
- ✅ **Robust tracking** with ByteTrack algorithm
- ✅ **Polygon-based ROI** for flexible area definition
- ✅ **Advanced hysteresis** to prevent boundary jitter
- ✅ **Enter/Exit event logging** in CSV and JSON formats
- ✅ **Annotated video output** with tracking IDs and occupancy
- ✅ **Performance monitoring** with FPS and latency metrics
- ✅ **Configuration via .env** for easy deployment

## 🧠 How It Works

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Input (File/Stream)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Frame Preprocessing & ROI Extraction           │
│                        (realtime)                           |
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Person Detection (Ultralytics)                 │
│         Detects all people in each frame with bbox          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         Tracking (Supervision)                              │
│    Tracks individuals across frames, handles occlusions     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│      Counting Logic & Occupancy Calculation                │
│   - Count unique tracks within polygon                     │
│   - Handle Enter/Exit events                               │
│   - Maintain current occupancy                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Output Generation                                │
│   - Annotated video with counts and IDs                    │
│   - CSV/JSON log with timestamps, track IDs, events       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Input & Output

### Input Files

Place your files in these directories:

```
📁 industrial-occupancy-tracker/
├── 📁 videos/
│   └── input_video.mp4          # Your video file (required)
├── 📁 polygons/
│   └── zone1.json         # ROI polygon definition (required)
├── 📁 models/
│   └── yolo11n.pt              # YOLO model (optional - auto-downloads)
└── 📄 .env                      # Configuration file (required)
```

---

#### 1. Video File (`videos/input_video.mp4`)

- **Required:** Yes
- **Format:** MP4, AVI, MOV, or any OpenCV-supported format
- **Naming:** You can change the filename in `.env` (VIDEO_PATH)
- **Example:** `videos/factory_footage.mp4`

```env
VIDEO_PATH=./videos/factory_footage.mp4
```

---

#### 2. Polygon Configuration (`polygons/roi_polygon.json`)

- **Required:** Yes
- **Format:** JSON with normalized coordinates (0-1)
- **Points:** Define the region of interest (minimum 3 points)

**Example:**
```json
{
  "name": "Main Production Area",
  "points": [
    [0.2, 0.1],
    [0.8, 0.1],
    [0.9, 0.9],
    [0.1, 0.9]
  ],
  "color_inside": [0, 255, 0],
  "color_outside": [0, 0, 255],
  "alpha": 0.3
}
```

**Field explanations:**
- `name`: Display name for the area
- `points`: Array of [x, y] coordinates (normalized 0-1)
  - x: 0 = left edge, 1 = right edge
  - y: 0 = top edge, 1 = bottom edge
- `color_inside`: BGR color for inside zone (optional)
- `color_outside`: BGR color for outside zone (optional)
- `alpha`: Transparency level 0-1 (optional)

```env
POLYGON_FILE=./polygons/roi_polygon.json
```

---

#### 3. YOLO Model (`models/yolo11n.pt`)

- **Required:** No (auto-downloads if missing)
- **Default:** yolo11n.pt (nano model - fastest)
- **Alternatives:** yolo11s.pt, yolo11m.pt, yolo11l.pt (larger = more accurate but slower)

**If you want to download manually:**
```bash
wget -P models/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
```

```env
MODEL_PATH=./models/yolo11n.pt
```

---

#### 4. Environment Configuration (`.env`)

- **Required:** Yes
- **Create from:** `.env.example`

```bash
# Create your .env file
cp .env.example .env
```

**Minimal configuration:**
```env
VIDEO_PATH=./videos/input_video.mp4
POLYGON_FILE=./polygons/roi_polygon.json
CONF_THRESHOLD=0.45
DEVICE=cpu
```

---

### Summary of Required Files

| File | Location | Required | Auto-Created |
|------|----------|----------|--------------|
| Video | `videos/input_video.mp4` | ✅ Yes | ❌ No |
| Polygon JSON | `polygons/roi_polygon.json` | ✅ Yes | ❌ No |
| .env | `.env` | ✅ Yes | ⚠️ Copy from .env.example |
| YOLO Model | `models/yolo11n.pt` | ❌ No | ✅ Yes (auto-download) |

---

### Quick Setup Commands

```bash
# 1. Create directories
mkdir -p models videos polygons outputs

# 2. Place your video
cp /path/to/your/video.mp4 videos/input_video.mp4

# 3. Create polygon config
cat > polygons/roi_polygon.json << 'EOF'
{
  "name": "My Area",
  "points": [
    [0.2, 0.1],
    [0.8, 0.1],
    [0.9, 0.9],
    [0.1, 0.9]
  ]
}
EOF

# 4. Create .env from example
cp .env.example .env

# 5. Edit .env with your paths
nano .env
```

---

### Output Files

All outputs are saved in the `outputs/` directory:

```
outputs/
├── annotated_video.mp4    # Video with bounding boxes, IDs, and occupancy
├── events.csv            # Event log in CSV format
└── events.json           # Event log in JSON format
```

#### CSV Format
```csv
timestamp,frame_id,track_id,event,occupancy,confidence
2024-01-15 10:30:15.123,45,T-001,ENTER,12,0.87
2024-01-15 10:30:16.456,60,T-003,EXIT,11,0.92
2024-01-15 10:30:17.789,75,T-001,INSIDE,12,0.88
```

#### JSON Format
```json
{
  "events": [
    {
      "timestamp": "2024-01-15T10:30:15.123",
      "frame_id": 45,
      "track_id": "T-001",
      "event": "ENTER",
      "occupancy": 12,
      "confidence": 0.87
    }
  ],
  "summary": {
    "total_frames": 1500,
    "total_entries": 23,
    "total_exits": 21,
    "max_occupancy": 15,
    "avg_occupancy": 10.5
  }
}
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/NavidAdib98/Industrial-People-Counter-AI.git
cd industrial-occupancy-tracker
```

### Step 2: Create Virtual Environment
```bash
# On Linux/MacOS
python -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use any text editor
```

### Step 5: Prepare Input Files
```bash
# Create necessary directories
mkdir -p models videos polygons outputs

# Place your files:
# - models/yolo11n.pt (optional - will auto-download)
# - videos/input_video.mp4 (your video)
# - polygons/roi_polygon.json (your polygon definition)
```

---

## ⚙️ Configuration

### Environment Variables (.env)

Create a `.env` file with the following configuration:

```env
# ============================================
# PATHS (relative to project root)
# ============================================
MODEL_PATH=./models/yolo11n.pt
VIDEO_PATH=./videos/input_video.mp4
POLYGON_FILE=./polygons/roi_polygon.json
OUTPUT_VIDEO_PATH=./outputs/annotated_video.mp4
EVENT_LOG_CSV_PATH=./outputs/events.csv
EVENT_LOG_JSON_PATH=./outputs/events.json

# ============================================
# MODEL CONFIGURATION
# ============================================
CONF_THRESHOLD=0.45          # Detection confidence threshold (0-1)
DEVICE=cpu                   # cpu, cuda, or mps

# ============================================
# TRACKING CONFIGURATION
# ============================================
TRACK_HISTORY_BUFFER_SIZE=30    # Number of frames to keep in history
TRACK_ACTIVATION_THRESHOLD=3    # Frames before activating a new track
TRACK_LOST_THRESHOLD=30         # Frames before considering track lost
MINIMUM_TRACK_LENGTH=5          # Minimum frames for valid track

# ============================================
# HYSTERESIS CONFIGURATION (Boundary handling)
# ============================================
HYSTERESIS_INSIDE_THRESHOLD=15   # Pixels inside to be "inside"
HYSTERESIS_OUTSIDE_THRESHOLD=25  # Pixels outside to be "outside"
MIN_FRAMES_BETWEEN_CHANGES=10    # Min frames between status changes
MIN_CONSECUTIVE_FRAMES=3         # Consecutive frames required

# ============================================
# BOUNDARY CONFIGURATION
# ============================================
BOUNDARY_PROXIMITY_THRESHOLD=40  # Pixels from boundary for event logging
CENTER_OFFSET=15                 # Pixels above bottom of bbox

# ============================================
# EVENT LOGGING
# ============================================
ENABLE_EVENT_LOGGING=true

# ============================================
# PERFORMANCE
# ============================================
MAX_FPS=30
BATCH_SIZE=1
USE_OPTIMIZED_INFERENCE=true

# ============================================
# VISUALIZATION
# ============================================
SHOW_LABELS=true
SHOW_TRACK_IDS=true
SHOW_BOUNDING_BOXES=true
SHOW_OCCUPANCY_TEXT=true
```

---

## 🎬 Usage

### Basic Usage

```bash
# Run with default settings from .env
python src/main.py
```

### Run with GPU Acceleration
```bash
# Set DEVICE=cuda in .env
# Or run with:
CUDA_VISIBLE_DEVICES=0 python src/main.py
```

### Expected Output
During runtime, you'll see:
```
==================================================
Initializing Person Tracker
==================================================
Loading model: ./models/yolo11n.pt
Tracker ready
   Model: yolo11n.pt
   Tracker: ByteTrack (Supervision)
   Confidence: 0.45
   Device: cpu
   Boundary Proximity: 40px
   Hysteresis: Inside=15px, Outside=25px
   Center Offset: 15px above bottom
   Polygon: Main Production Area (4 points)
==================================================

==================================================
Starting People Tracker Application
==================================================
Processing video: ./videos/input_video.mp4
Frame 100 | Inside: 12 | Outside: 3 | Total: 15 | FPS: 28.5
Frame 200 | Inside: 14 | Outside: 2 | Total: 16 | FPS: 29.1
...
==================================================
Processing Complete!
  Total frames: 1500
  Average FPS: 28.7
  Max occupancy: 18
  Output video: ./outputs/annotated_video.mp4
  Events saved to: ./outputs/events.csv
==================================================
```

---

## 🎯 Tracking & Occlusion Handling

### Detection Pipeline

1. **YOLO Detection**
   - Processes each frame with YOLO
   - Returns bounding boxes for all detected people
   - Confidence threshold filters weak detections

2. **ByteTrack Tracking**
   - Associates detections across frames
   - Uses Kalman filters for motion prediction
   - Maintains track IDs for each person
   - Handles short-term occlusions

### Boundary Hysteresis Strategy

The system uses a sophisticated hysteresis mechanism to prevent jitter when people are on the ROI boundary:

```python
# Key Concepts:
# 1. OFFSET: Uses a point slightly above the bottom of the bbox
#    (avoids foot-boundary issues)
# 
# 2. HYSTERESIS: Different thresholds for entering vs exiting
#    - Inside → Outside: Must go 25px outside
#    - Outside → Inside: Must go 15px inside
# 
# 3. TEMPORAL FILTERING: Requires consecutive frames to change status
#    - Prevents rapid toggling
#    - Requires 3 consecutive frames for confirmation
# 
# 4. EVENT LOGGING: Only logs events when near boundary
#    - Reduces noise from far-away objects
#    - Focuses on important transitions
```

### How It Prevents Boundary Jitter

```
Traditional Approach (No Hysteresis):
┌─────────────────────────────┐
│  Person crosses boundary     │
│  → Status changes every frame │
│  → Rapid toggling (jitter)   │
└─────────────────────────────┘

Our Approach (With Hysteresis):
┌──────────────────────────────────────────┐
│  Person moving IN:                       │
│  → Must be 15px inside to switch         │
│  → Requires 3 consecutive frames         │
│  → Smooth transition                     │
│                                          │
│  Person moving OUT:                      │
│  → Must be 25px outside to switch        │
│  → Requires 3 consecutive frames         │
│  → Smooth transition                     │
└──────────────────────────────────────────┘
```

### Visual Representation

```
Boundary Zone Visualization:

         INSIDE ZONE
    ┌─────────────────────┐
    │   -15px threshold   │ ← Hysteresis Inside (15px)
    │   ┌───────────┐    │
    │   │  BOUNDARY │    │ ← Polygon Boundary
    │   └───────────┘    │
    │   +25px threshold   │ ← Hysteresis Outside (25px)
    └─────────────────────┘
         OUTSIDE ZONE

- Blue: Inside zone (requires +15px to switch)
- Red: Outside zone (requires -25px to switch)
- Green: Boundary area (event logging zone, 40px)
```

### Offset Strategy

```python
# We use a point slightly above the bottom of the bounding box
# Why? People's feet are on the ground, but bounding boxes often
# extend below the actual foot position due to shadows or artifacts

# Without offset:
bbox_bottom = y2  # May include shadow/artifact

# With offset:
reference_point = (center_x, y2 - 15)  # 15px above bottom
# More accurate for boundary crossing detection
```

---

## 📁 Project Structure

```
industrial-occupancy-tracker/
│
├── .env                           # Environment configuration
├── .env.example                   # Example configuration
├── .dockerignore                  # Docker ignore file
├── .gitignore                     # Git ignore file
├── Dockerfile                     # Docker build file
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── main.py                    # Entry point
│   │
│   ├── core/                      # Core modules
│   │   ├── __init__.py
│   │   ├── event_logger.py        # Event logging to CSV/JSON
│   │   ├── realtime_processor.py  # realtime frame capture
│   │   └── tracker.py      
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── polygon_loader.py      # Polygon configuration loader
│       └── settings.py      
│
├── models/                        # Model files (volume)
│   └── yolo11n.pt                 # YOLO model (auto-download)
│
├── videos/                        # Input videos (volume)
│   └── input_video.mp4            # Your video file
│
├── polygons/                      # Polygon configurations (volume)
│   └── roi_polygon.json           # ROI polygon definition
│
└── outputs/                       # Output files (volume)
    ├── annotated_video.mp4        # Annotated video
    ├── events.csv                 # Event log (CSV)
    └── events.json                # Event log (JSON)
```

---

## 🐳 Docker (In Progress)

> ⚠️ **Note:** Docker support is currently under development and not recommended for production use.

The Docker setup is being prepared and will be available in future releases. Current Dockerfile is present but not yet production-ready.

**Planned Docker features:**
- GPU acceleration support
- Volume mounting for inputs/outputs
- Easy deployment
- Scalable architecture

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Detection model
- [Supervision](https://github.com/roboflow/supervision) - Tracking and utilities
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Tracking algorithm

---

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Supervision Documentation](https://supervision.roboflow.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)

---

**Made with ❤️ for industrial safety and efficiency**