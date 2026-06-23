"""
Person Tracker using Ultralytics + Supervision
Pure tracking logic with Supervision's ByteTrack
"""

import time
import numpy as np
from ultralytics import YOLO
import supervision as sv
from polygon_loader import PolygonLoader


class PersonTracker:
    """
    Person tracker using YOLO detection + Supervision ByteTrack
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
        
        if not model_path.exists():
            print(f"⚠️  Model not found at: {model_path}")
            print("   Downloading default model...")
            self.model = YOLO("yolo11n.pt")
            self.model.save(str(model_path))
            print(f"✅ Model downloaded and saved to: {model_path}")
        else:
            self.model = YOLO(str(model_path))
        
        # Initialize Supervision ByteTrack
        self.tracker = sv.ByteTrack()
        
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
            self.color_inside = (0, 255, 0)
            self.color_outside = (0, 0, 255)
            self.polygon_alpha = 0.3
            self.polygon_name = None
        
        # Settings
        self.conf_threshold = settings.CONF_THRESHOLD
        self.device = settings.DEVICE
        
        # Performance tracking
        self.frame_count = 0
        self.fps_list = []
        self.last_time = time.time()
        
        # Counts
        self.people_inside = 0
        self.people_outside = 0
        
        print(f"✅ Tracker ready")
        print(f"   Model: {model_path.name}")
        print(f"   Tracker: ByteTrack (Supervision)")
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
        import cv2
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def process_frame(self, frame):
        """
        Process a single frame - returns detections and counts
        
        Args:
            frame: Input image
            
        Returns:
            dict: {
                'detections': sv.Detections,
                'people_inside': int,
                'people_outside': int,
                'polygon_points': pixel polygon points or None,
                'polygon_name': str or None,
                'color_inside': tuple,
                'color_outside': tuple,
                'polygon_alpha': float,
                'fps': float,
                'frame_count': int
            }
        """
        self.frame_count += 1
        
        height, width = frame.shape[:2]
        
        # Get polygon points
        polygon = None
        if self.polygon_loader:
            polygon = self.polygon_loader.get_pixel_points(width, height)
        
        # Run YOLO detection
        results = self.model(frame, conf=self.conf_threshold, classes=[0], verbose=False)[0]
        
        # Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Update tracker
        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
        
        # Reset counts
        self.people_inside = 0
        self.people_outside = 0
        
        # Add inside/outside info to detections
        if len(detections) > 0 and detections.tracker_id is not None:
            is_inside_list = []
            center_list = []
            
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                center_x = (x1 + x2) // 2
                bottom_y = y2
                center_point = (int(center_x), int(bottom_y))
                
                is_inside = self._point_in_polygon(center_point, polygon) if polygon is not None else True
                
                is_inside_list.append(is_inside)
                center_list.append(center_point)
                
                if is_inside:
                    self.people_inside += 1
                else:
                    self.people_outside += 1
            
            detections.is_inside = is_inside_list
            detections.centers = center_list
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.last_time = current_time
        self.fps_list.append(fps)
        avg_fps = sum(self.fps_list[-30:]) / min(len(self.fps_list), 30)
        
        return {
            'detections': detections,
            'people_inside': self.people_inside,
            'people_outside': self.people_outside,
            'total': self.people_inside + self.people_outside,
            'polygon_points': polygon,
            'polygon_name': self.polygon_name,
            'color_inside': self.color_inside,
            'color_outside': self.color_outside,
            'polygon_alpha': self.polygon_alpha,
            'fps': avg_fps,
            'frame_count': self.frame_count
        }
    
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