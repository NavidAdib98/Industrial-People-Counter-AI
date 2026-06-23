"""
Person Tracker using Ultralytics + Supervision
Pure tracking logic with Supervision's ByteTrack
"""

import time
import logging
import numpy as np
from ultralytics import YOLO
import supervision as sv
from utils.polygon_loader import PolygonLoader
from core.event_logger import EventLogger


# Configure logger for this module
logger = logging.getLogger(__name__)


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
        logger.info("=" * 50)
        logger.info("Initializing Person Tracker")
        logger.info("=" * 50)
        
        self.settings = settings
        
        # Load model
        model_path = settings.MODEL_PATH
        logger.info(f"Loading model: {model_path}")
        
        if not model_path.exists():
            logger.warning(f"Model not found at: {model_path}")
            logger.info("Downloading default model...")
            self.model = YOLO("yolo11n.pt")
            self.model.save(str(model_path))
            logger.info(f"Model downloaded and saved to: {model_path}")
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
        
        # ==================== EVENT LOGGER ====================
        self.event_logger = EventLogger()
        
        # ==================== BOUNDARY PROXIMITY SETTINGS ====================
        self.boundary_proximity_threshold = 40  # pixels
        
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
        self.hysteresis_inside_threshold = 15   # pixels inside to be considered "inside"
        self.hysteresis_outside_threshold = 25  # pixels outside to be considered "outside"
        self.track_status_history = {}
        self.min_frames_between_changes = 10
        
        # ==================== OFFSET SETTINGS ====================
        self.center_offset = 15  # pixels above the bottom of the box
        
        logger.info("Tracker ready")
        logger.info(f"   Model: {model_path.name}")
        logger.info(f"   Tracker: ByteTrack (Supervision)")
        logger.info(f"   Confidence: {settings.CONF_THRESHOLD}")
        logger.info(f"   Device: {settings.DEVICE}")
        logger.info(f"   Boundary Proximity: {self.boundary_proximity_threshold}px")
        logger.info(f"   Hysteresis: Inside={self.hysteresis_inside_threshold}px, Outside={self.hysteresis_outside_threshold}px")
        logger.info(f"   Center Offset: {self.center_offset}px above bottom")
        
        if self.polygon_loader:
            logger.info(f"   Polygon: {self.polygon_name} ({len(self.polygon_loader.points)} points)")
        else:
            logger.info("   Polygon: None (no ROI)")
        
        logger.info("=" * 50)
    
    def _point_in_polygon(self, point, polygon):
        """
        Check if point is inside polygon
        
        Args:
            point: (x, y) tuple
            polygon: List of polygon points (pixel coordinates)
            
        Returns:
            bool: True if inside, False otherwise
        """
        if polygon is None or len(polygon) < 3:
            return False
        import cv2
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def _get_point_with_offset(self, x1, y1, x2, y2):
        """
        Get the reference point for inside/outside check with offset
        
        Uses a point slightly above the bottom of the bounding box
        to avoid foot-boundary issues.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            tuple: (center_x, reference_y)
        """
        center_x = (x1 + x2) // 2
        bottom_y = y2
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
        return cv2.pointPolygonTest(polygon, point, True)
    
    def _is_near_boundary(self, point, polygon):
        """
        Check if a point is near the polygon boundary
        
        Args:
            point: Reference point (x, y)
            polygon: List of polygon points
            
        Returns:
            bool: True if near boundary
        """
        if polygon is None or len(polygon) < 3:
            return False
        
        distance = self._distance_to_polygon_boundary(point, polygon)
        return abs(distance) < self.boundary_proximity_threshold
    
    def _is_inside_with_hysteresis(self, track_id, point, polygon, current_frame):
        """
        Determine if a point is inside the polygon using hysteresis
        
        Hysteresis prevents jitter when people are on the boundary by using
        different thresholds for entering and exiting.
        
        Args:
            track_id: The track ID
            point: Reference point (x, y)
            polygon: List of polygon points
            current_frame: Current frame number
            
        Returns:
            bool: True if inside, False if outside
        """
        if polygon is None or len(polygon) < 3:
            return True
        
        distance = self._distance_to_polygon_boundary(point, polygon)
        
        # Get previous status for this track
        prev_status = self.track_status_history.get(track_id, {
            'is_inside': True,
            'last_status_change': 0,
            'frames_inside': 0,
            'frames_outside': 0
        })
        
        frames_since_change = current_frame - prev_status.get('last_status_change', 0)
        
        # Prevent rapid toggling if status changed recently
        if frames_since_change < self.min_frames_between_changes:
            return prev_status.get('is_inside', True)
        
        # Determine new status based on hysteresis
        if prev_status.get('is_inside', True):
            # Currently inside - need to go outside_threshold to switch
            if distance > -self.hysteresis_outside_threshold:
                new_status = True
            else:
                new_status = False
        else:
            # Currently outside - need to go inside_threshold to switch
            if distance < self.hysteresis_inside_threshold:
                new_status = False
            else:
                new_status = True
        
        # Update history if status changed
        if new_status != prev_status.get('is_inside', True):
            prev_status['last_status_change'] = current_frame
        
        prev_status['is_inside'] = new_status
        
        # Track consecutive frames in current state
        if new_status:
            prev_status['frames_inside'] = prev_status.get('frames_inside', 0) + 1
            prev_status['frames_outside'] = 0
        else:
            prev_status['frames_outside'] = prev_status.get('frames_outside', 0) + 1
            prev_status['frames_inside'] = 0
        
        # Require 3 consecutive frames to confirm status change
        if frames_since_change >= self.min_frames_between_changes:
            if new_status:
                if prev_status.get('frames_inside', 0) < 3:
                    new_status = prev_status.get('is_inside', True)
            else:
                if prev_status.get('frames_outside', 0) < 3:
                    new_status = prev_status.get('is_inside', True)
        
        self.track_status_history[track_id] = prev_status
        return new_status
    
    def process_frame(self, frame, video_frame_number=None, video_time=None, timestamp=None):
        """
        Process a single frame - returns detections and counts
        
        Args:
            frame: Input image (numpy array)
            video_frame_number: Actual video frame number (from VideoCapture)
            video_time: Video time in seconds
            timestamp: System timestamp when frame was read
            
        Returns:
            dict: {
                'detections': sv.Detections object with tracker IDs,
                'people_inside': int,
                'people_outside': int,
                'total': int,
                'polygon_points': pixel polygon points or None,
                'polygon_name': str or None,
                'color_inside': tuple (BGR),
                'color_outside': tuple (BGR),
                'polygon_alpha': float,
                'fps': float,
                'frame_count': int,
                'video_frame_number': int
            }
        """
        current_frame = video_frame_number if video_frame_number is not None else self.frame_count
        self.frame_count += 1
        
        height, width = frame.shape[:2]
        
        # Get polygon points in pixel coordinates
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
            near_boundary_list = []
            
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                track_id = int(detections.tracker_id[i])
                confidence = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                
                # Get reference point with OFFSET (above bottom)
                center_x, ref_y = self._get_point_with_offset(x1, y1, x2, y2)
                ref_point = (center_x, ref_y)
                
                # Check if inside using HYSTERESIS
                is_inside = self._is_inside_with_hysteresis(
                    track_id, ref_point, polygon, current_frame
                ) if polygon is not None else True
                
                # Check if near boundary
                is_near_boundary = self._is_near_boundary(ref_point, polygon) if polygon is not None else False
                
                # Log state change event ONLY if near boundary
                if is_near_boundary:
                    self.event_logger.check_and_log_state_change(
                        track_id, is_inside, is_near_boundary, current_frame,
                        confidence, video_time, timestamp
                    )
                else:
                    # If not near boundary, just update state without logging
                    current_state = 'inside' if is_inside else 'outside'
                    prev_state = self.event_logger.track_last_state.get(track_id, None)
                    if prev_state is None:
                        self.event_logger.track_last_state[track_id] = current_state
                        if is_inside:
                            self.event_logger.current_occupancy += 1
                    else:
                        self.event_logger.track_last_state[track_id] = current_state
                
                # Store the reference point (bottom center for display)
                bottom_center = (int(center_x), int(y2))
                
                is_inside_list.append(is_inside)
                center_list.append(bottom_center)
                near_boundary_list.append(is_near_boundary)
                
                if is_inside:
                    self.people_inside += 1
                else:
                    self.people_outside += 1
            
            detections.is_inside = is_inside_list
            detections.centers = center_list
            detections.is_near_boundary = near_boundary_list
        
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
            'frame_count': self.frame_count,
            'video_frame_number': current_frame
        }
    
    def get_stats(self):
        """
        Get performance statistics
        
        Returns:
            dict or None: Statistics if frames processed
        """
        if self.fps_list:
            return {
                'total_frames': self.frame_count,
                'avg_fps': sum(self.fps_list) / len(self.fps_list),
                'max_fps': max(self.fps_list),
                'min_fps': min(self.fps_list)
            }
        return None
    
    def get_counts(self):
        """
        Get current people counts
        
        Returns:
            dict: {'inside': int, 'outside': int, 'total': int}
        """
        return {
            'inside': self.people_inside,
            'outside': self.people_outside,
            'total': self.people_inside + self.people_outside
        }
    
    def reset_track_status(self):
        """
        Reset the track status history
        
        Useful when switching videos or re-starting processing
        """
        self.track_status_history = {}
        self.event_logger.reset()
        logger.info("Track status history reset")
    
    def save_events(self):
        """
        Save events to files (JSON and CSV)
        """
        self.event_logger.save()
        self.event_logger.print_summary()