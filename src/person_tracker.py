"""
Person Tracker Class using Ultralytics YOLO
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from polygon_loader import PolygonLoader


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
            self.model = YOLO("yolo11n.pt")
            self.model.save(str(model_path))
            print(f"✅ Model downloaded and saved to: {model_path}")
        else:
            self.model = YOLO(str(model_path))
        
        # Load polygon
        self.polygon_loader = None
        if settings.POLYGON_FILE:
            self.polygon_loader = PolygonLoader(settings.POLYGON_FILE)
            if not self.polygon_loader.is_loaded():
                self.polygon_loader = None
        
        # Get colors
        if self.polygon_loader:
            self.color_inside, self.color_outside = self.polygon_loader.get_colors_bgr()
            self.polygon_alpha = self.polygon_loader.alpha
            self.polygon_name = self.polygon_loader.name
        else:
            # Default colors if no polygon
            self.color_inside = (0, 255, 0)   # Green
            self.color_outside = (0, 0, 255)  # Red
            self.polygon_alpha = 0.3
            self.polygon_name = None
        
        # Settings
        self.conf_threshold = settings.CONF_THRESHOLD
        self.tracker_type = settings.TRACKER_TYPE
        self.device = settings.DEVICE
        
        # Performance tracking
        self.frame_count = 0
        self.fps_list = []
        self.last_time = time.time()
        
        # Track history
        self.track_history = {}
        
        # Counts
        self.people_inside = 0
        self.people_outside = 0
        
        # Determine model type
        model_path_str = str(model_path)
        if model_path_str.endswith('.pt'):
            model_type = "PyTorch"
        elif model_path_str.endswith('.onnx'):
            model_type = "ONNX"
        elif 'openvino' in model_path_str.lower() or model_path.is_dir():
            model_type = "OpenVINO"
        else:
            model_type = "Unknown"
        
        print(f"✅ Tracker ready")
        print(f"   Model: {model_path.name}")
        print(f"   Model Type: {model_type}")
        print(f"   Tracker: {settings.TRACKER_TYPE}")
        print(f"   Confidence: {settings.CONF_THRESHOLD}")
        print(f"   Device: {settings.DEVICE}")
        
        if self.polygon_loader:
            print(f"   Polygon: {self.polygon_name} ({len(self.polygon_loader.points)} points)")
        else:
            print(f"   Polygon: None (no ROI)")
        
        print("=" * 50)
        print()
    
    def _point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        if polygon is None or len(polygon) < 3:
            return False
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def _draw_polygon(self, frame, polygon):
        """Draw polygon on frame"""
        if polygon is None or len(polygon) < 3:
            return frame
        
        # Create overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], self.color_inside)
        cv2.addWeighted(overlay, self.polygon_alpha, frame, 1 - self.polygon_alpha, 0, frame)
        
        # Draw outline
        cv2.polylines(frame, [polygon], True, self.color_inside, 2)
        
        # Draw name
        if self.polygon_name:
            moments = cv2.moments(polygon)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                cv2.putText(frame, self.polygon_name, (cx - 40, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_inside, 2)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        
        height, width = frame.shape[:2]
        
        # Get polygon points
        polygon = None
        if self.polygon_loader:
            polygon = self.polygon_loader.get_pixel_points(width, height)
        
        # Prepare tracking
        track_kwargs = {
            'persist': True,
            'conf': self.conf_threshold,
            'iou': 0.5,
            'classes': [0],
            'verbose': False
        }
        
        if self.tracker_type:
            track_kwargs['tracker'] = self.tracker_type
        
        # Run tracking
        results = self.model.track(frame, **track_kwargs)
        result = results[0]
        
        annotated_frame = frame.copy()
        detections = []
        
        # Reset counts
        self.people_inside = 0
        self.people_outside = 0
        
        # Draw polygon
        if polygon is not None:
            annotated_frame = self._draw_polygon(annotated_frame, polygon)
        
        # Process detections
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Center point (bottom center)
                center_x = (x1 + x2) // 2
                bottom_y = y2
                center_point = (center_x, bottom_y)
                
                # Check if inside
                is_inside = self._point_in_polygon(center_point, polygon) if polygon is not None else True
                
                # Choose color
                color = self.color_inside if is_inside else self.color_outside
                
                # Count
                if is_inside:
                    self.people_inside += 1
                else:
                    self.people_outside += 1
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'track_id': int(track_id),
                    'is_inside': is_inside,
                    'center': center_point
                })
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"ID:{track_id}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                cv2.rectangle(annotated_frame, (x1, y1 - label_h - 8), (x1 + label_w, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw center dot
                cv2.circle(annotated_frame, center_point, 4, color, -1)
                
                # Track path
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(center_point)
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                points = self.track_history[track_id]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame, points[i-1], points[i], color, 2)
        
        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.last_time = current_time
        self.fps_list.append(fps)
        avg_fps = sum(self.fps_list[-30:]) / min(len(self.fps_list), 30)
        
        # Draw stats
        y = 30
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30
        
        total = self.people_inside + self.people_outside
        cv2.putText(annotated_frame, f"Total: {total}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
        
        cv2.putText(annotated_frame, f"Inside: {self.people_inside}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_inside, 2)
        y += 30
        
        cv2.putText(annotated_frame, f"Outside: {self.people_outside}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_outside, 2)
        y += 30
        
        # Show polygon name
        if self.polygon_name:
            cv2.putText(annotated_frame, f"Zone: {self.polygon_name}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_inside, 1)
        
        return annotated_frame, detections
    
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
    
    def get_counts(self):
        """Get people counts"""
        return {
            'inside': self.people_inside,
            'outside': self.people_outside,
            'total': self.people_inside + self.people_outside
        }