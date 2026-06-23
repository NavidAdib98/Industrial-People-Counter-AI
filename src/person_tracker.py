"""
Person Tracker Class using Ultralytics YOLO
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO


class PersonTracker:
    """
    Person tracker using YOLO with built-in tracking
    """
    
    def __init__(self, settings):
        """
        Initialize tracker with settings
        
        Args:
            settings: Settings object
        """
        print("=" * 50)
        print("👤 Initializing Person Tracker")
        print("=" * 50)
        
        self.settings = settings
        
        # Load model
        model_path = settings.MODEL_PATH
        print(f"📦 Model: {model_path}")
        
        # Check if model exists
        if not model_path.exists():
            print(f"⚠️  Model not found at: {model_path}")
            print("   Downloading default model...")
            # Download the model to the models directory
            self.model = YOLO("yolo11n.pt")
            # Save it to the models directory
            self.model.save(str(model_path))
            print(f"✅ Model downloaded and saved to: {model_path}")
        else:
            # Load the model (works for both .pt and .onnx)
            self.model = YOLO(str(model_path))
        
        # For ONNX models, don't call .to(device)
        # The device is handled during inference
        if not str(model_path).endswith('.onnx'):
            self.model.to(settings.DEVICE)
        
        # Settings
        self.conf_threshold = settings.CONF_THRESHOLD
        self.tracker_type = settings.TRACKER_TYPE
        self.device = settings.DEVICE
        
        # Performance tracking
        self.frame_count = 0
        self.fps_list = []
        self.last_time = time.time()
        
        # Track history for paths
        self.track_history = {}
        
        print(f"✅ Tracker ready")
        print(f"   Model: {model_path.name}")
        print(f"   Model Type: {'ONNX' if str(model_path).endswith('.onnx') else 'PyTorch'}")
        print(f"   Tracker: {settings.TRACKER_TYPE}")
        print(f"   Confidence: {settings.CONF_THRESHOLD}")
        print(f"   Device: {settings.DEVICE}")
        print("=" * 50)
        print()
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input image
            
        Returns:
            annotated_frame: Frame with visualizations
            detections: List of detections
        """
        self.frame_count += 1
        
        # Prepare tracking arguments
        track_kwargs = {
            'persist': True,
            'conf': self.conf_threshold,
            'iou': 0.5,
            'classes': [0],  # Only people
            'verbose': False
        }
        
        # Add device for PyTorch models, but not for ONNX
        # For ONNX, the device is handled automatically
        if not str(self.settings.MODEL_PATH).endswith('.onnx'):
            track_kwargs['device'] = self.device
        
        # Add tracker if specified
        if self.tracker_type:
            track_kwargs['tracker'] = self.tracker_type
        
        # Run YOLO tracking
        results = self.model.track(frame, **track_kwargs)
        
        result = results[0]
        annotated_frame = frame.copy()
        detections = []
        
        # Process detections
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'track_id': int(track_id),
                    'confidence': float(conf)
                })
                
                # Draw bounding box
                color = self._get_color(track_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID label
                label = f"ID:{track_id}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_h - 8),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
                
                # Draw track path
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(center)
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                points = self.track_history[track_id]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame, points[i-1], points[i], color, 2)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.last_time = current_time
        self.fps_list.append(fps)
        
        avg_fps = sum(self.fps_list[-30:]) / min(len(self.fps_list), 30)
        
        # Draw stats
        cv2.putText(
            annotated_frame,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            annotated_frame,
            f"People: {len(detections)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Show model type
        model_type = "ONNX" if str(self.settings.MODEL_PATH).endswith('.onnx') else "PyTorch"
        cv2.putText(
            annotated_frame,
            f"Model: {model_type}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        return annotated_frame, detections
    
    def _get_color(self, track_id):
        """Generate consistent color for track ID"""
        np.random.seed(int(track_id) * 10 + 1)
        return tuple(map(int, np.random.randint(50, 255, 3)))
    
    def get_stats(self):
        """Get performance statistics"""
        if self.fps_list:
            return {
                'total_frames': self.frame_count,
                'avg_fps': sum(self.fps_list) / len(self.fps_list),
                'max_fps': max(self.fps_list),
                'min_fps': min(self.fps_list)
            }
        return None