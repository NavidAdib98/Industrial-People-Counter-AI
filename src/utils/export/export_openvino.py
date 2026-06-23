"""
Export YOLO model to OpenVINO for CPU optimization
OpenVINO is Intel's optimized inference engine
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

# Create models directory
Path("models").mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("📦 Exporting Model to OpenVINO")
print("=" * 50)
print()

# Load and export model to OpenVINO
model = YOLO("models\yolo11n.pt")
model.export(
    format="openvino",
    half=False,       # FP32 (better for CPU)
    device="cpu",
    imgsz=640
)
