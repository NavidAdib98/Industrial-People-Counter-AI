"""
Export YOLO model to ONNX with INT8 quantization for CPU optimization
Run this script once to create an optimized model
"""

from ultralytics import YOLO

# Load and export model with INT8 quantization
model = YOLO("models\yolo26n.pt")
model.export(format="onnx", int8=True, data="coco8.yaml", device="cpu", imgsz=640, simplify=True)

print("\n✅ Model exported to: yolo11n_int8.onnx")
print("   Update your .env file:")
print("   MODEL_NAME=yolo11n_int8.onnx")