"""
Person Tracker Class - Pure tracking logic only
"""

import time
import numpy as np
from ultralytics import YOLO
from polygon_loader import PolygonLoader


class PersonTracker:
    """
    Person tracker using YOLO with built-in tracking
    Only handles detection, tracking, and counting logic
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
        
        # Get colors (for tracking logic only)
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
        
        print(f"✅ Tracker ready")
        print(f"   Model: {model_path.name}")
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
        import cv2
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def process_frame(self, frame):
        """
        Process a single frame - returns detections and counts only
        
        Args:
            frame: Input image
            
        Returns:
            dict: {
                'detections': List of detection dicts,
                'people_inside': int,
                'people_outside': int,
                'polygon_points': pixel polygon points or None,
                'polygon_name': str or None,
                'color_inside': tuple,
                'color_outside': tuple,
                'polygon_alpha': float,
                'fps': float
            }
        """
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
        
        detections = []
        
        # Reset counts
        self.people_inside = 0
        self.people_outside = 0
        
        # Process detections
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Center point (bottom center for feet position)
                center_x = (x1 + x2) // 2
                bottom_y = y2
                center_point = (center_x, bottom_y)
                
                # Check if inside
                is_inside = self._point_in_polygon(center_point, polygon) if polygon is not None else True
                
                # Count
                if is_inside:
                    self.people_inside += 1
                else:
                    self.people_outside += 1
                
                # Store detection
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'track_id': int(track_id),
                    'is_inside': is_inside,
                    'center': center_point
                })
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(center_point)
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.last_time = current_time
        self.fps_list.append(fps)
        avg_fps = sum(self.fps_list[-30:]) / min(len(self.fps_list), 30)
        
        # Return all data
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
            'track_history': self.track_history,
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