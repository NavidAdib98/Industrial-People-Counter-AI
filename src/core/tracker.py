"""
Person Tracker using Ultralytics + Supervision
Pure tracking logic with Supervision's ByteTrack
"""

import time
import numpy as np
from ultralytics import YOLO
import supervision as sv
from utils.polygon_loader import PolygonLoader


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
        
        # ==================== HYSTERESIS SETTINGS ====================
        # Hysteresis thresholds for inside/outside detection
        # Prevents jitter when people are on the boundary
        self.hysteresis_inside_threshold = 15   # pixels inside to be considered "inside"
        self.hysteresis_outside_threshold = 25  # pixels outside to be considered "outside"
        
        # Track previous status for each track ID (for hysteresis)
        # Format: {track_id: {'is_inside': bool, 'last_status_change': frame_count}}
        self.track_status_history = {}
        
        # Minimum frames between status changes (to prevent rapid toggling)
        self.min_frames_between_changes = 10
        
        # ==================== OFFSET SETTINGS ====================
        # Offset from bottom of bounding box (in pixels)
        # Using a point slightly above the bottom helps avoid foot-boundary issues
        self.center_offset = 15  # pixels above the bottom of the box
        
        print(f"✅ Tracker ready")
        print(f"   Model: {model_path.name}")
        print(f"   Tracker: ByteTrack (Supervision)")
        print(f"   Confidence: {settings.CONF_THRESHOLD}")
        print(f"   Device: {settings.DEVICE}")
        print(f"   Hysteresis: Inside={self.hysteresis_inside_threshold}px, Outside={self.hysteresis_outside_threshold}px")
        print(f"   Center Offset: {self.center_offset}px above bottom")
        
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
    
    def _get_point_with_offset(self, x1, y1, x2, y2):
        """
        Get the reference point for inside/outside check with offset
        
        Instead of using the bottom center, we use a point slightly above
        the bottom to avoid foot-boundary issues.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            tuple: (center_x, reference_y) - reference point with offset
        """
        center_x = (x1 + x2) // 2
        bottom_y = y2
        
        # Apply offset: move the reference point UP from the bottom
        # This helps when people's feet are on the boundary but their body is inside
        ref_y = bottom_y - self.center_offset
        
        return int(center_x), int(ref_y)
    
    def _distance_to_polygon_boundary(self, point, polygon):
        """
        Calculate the distance from a point to the polygon boundary
        
        Returns:
            float: Positive if inside, negative if outside
        """
        import cv2
        if polygon is None or len(polygon) < 3:
            return float('inf')
        
        # cv2.pointPolygonTest returns:
        #   Positive: distance from point to polygon boundary (inside)
        #   Negative: distance from point to polygon boundary (outside)
        #   0: point is on boundary
        return cv2.pointPolygonTest(polygon, point, True)
    
    def _is_inside_with_hysteresis(self, track_id, point, polygon, current_frame):
        """
        Determine if a point is inside the polygon using hysteresis
        
        Args:
            track_id: The track ID
            point: The reference point (x, y)
            polygon: The polygon points
            current_frame: Current frame number
            
        Returns:
            bool: True if inside, False if outside
        """
        if polygon is None or len(polygon) < 3:
            return True
        
        # Get distance to polygon boundary
        distance = self._distance_to_polygon_boundary(point, polygon)
        
        # Get previous status for this track
        prev_status = self.track_status_history.get(track_id, {
            'is_inside': True,
            'last_status_change': 0,
            'frames_inside': 0,
            'frames_outside': 0
        })
        
        # Calculate frames since last change
        frames_since_change = current_frame - prev_status.get('last_status_change', 0)
        
        # If we've changed status recently, prevent rapid toggling
        if frames_since_change < self.min_frames_between_changes:
            return prev_status.get('is_inside', True)
        
        # Determine new status based on hysteresis
        if prev_status.get('is_inside', True):
            # Currently inside - need to go outside_threshold to switch
            if distance > -self.hysteresis_outside_threshold:
                # Still inside (or just slightly outside)
                new_status = True
            else:
                # Far enough outside - switch to outside
                new_status = False
        else:
            # Currently outside - need to go inside_threshold to switch
            if distance < self.hysteresis_inside_threshold:
                # Still outside (or just slightly inside)
                new_status = False
            else:
                # Far enough inside - switch to inside
                new_status = True
        
        # Update history if status changed
        if new_status != prev_status.get('is_inside', True):
            prev_status['last_status_change'] = current_frame
        
        prev_status['is_inside'] = new_status
        
        # Track frames inside/outside (for additional stability)
        if new_status:
            prev_status['frames_inside'] = prev_status.get('frames_inside', 0) + 1
            prev_status['frames_outside'] = 0
        else:
            prev_status['frames_outside'] = prev_status.get('frames_outside', 0) + 1
            prev_status['frames_inside'] = 0
        
        # Additional stability: require at least 3 consecutive frames to change
        # This prevents flickering due to detection noise
        if frames_since_change >= self.min_frames_between_changes:
            # Check if we have enough consecutive frames
            if new_status:
                if prev_status.get('frames_inside', 0) < 3:
                    new_status = prev_status.get('is_inside', True)
            else:
                if prev_status.get('frames_outside', 0) < 3:
                    new_status = prev_status.get('is_inside', True)
        
        # Update history
        self.track_status_history[track_id] = prev_status
        
        return new_status
    
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
                track_id = int(detections.tracker_id[i])
                
                # Get reference point with OFFSET (above bottom)
                center_x, ref_y = self._get_point_with_offset(x1, y1, x2, y2)
                ref_point = (center_x, ref_y)
                
                # Check if inside using HYSTERESIS
                is_inside = self._is_inside_with_hysteresis(
                    track_id, ref_point, polygon, self.frame_count
                ) if polygon is not None else True
                
                # Store the reference point (bottom center for display)
                bottom_center = (int(center_x), int(y2))
                
                is_inside_list.append(is_inside)
                center_list.append(bottom_center)
                
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
    
    def reset_track_status(self):
        """Reset the track status history (useful when changing video or polygon)"""
        self.track_status_history = {}
        print("🔄 Track status history reset")