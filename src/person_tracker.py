"""
Person Tracker Class using Ultralytics YOLO
Handles person detection and tracking with consistent IDs
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO


class PersonTracker:
    """
    Person tracker using YOLO with built-in tracking capabilities
    Handles detection, tracking, and visualization
    """
    
    def __init__(self, model_name="yolo11n.pt", tracker_type="bytetrack.yaml", 
                 conf_threshold=0.4, device="cpu"):
        """
        Initialize the person tracker
        
        Args:
            model_name: YOLO model file name
            tracker_type: Tracker config (bytetrack.yaml, botsort.yaml, etc.)
            conf_threshold: Confidence threshold for detections
            device: 'cpu' or 'cuda'
        """
        print("=" * 60)
        print("👤 Initializing Person Tracker")
        print("=" * 60)
        
        # Load model
        print(f"📦 Loading model: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)
        
        # Tracking settings
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Performance tracking
        self.frame_count = 0
        self.fps_list = []
        self.last_time = time.time()
        
        # Track history for drawing paths
        self.track_history = {}
        
        print(f"✅ Tracker initialized")
        print(f"   Tracker: {tracker_type}")
        print(f"   Confidence: {conf_threshold}")
        print(f"   Device: {device}")
        print("=" * 60)
        print()
    
    def process_frame(self, frame):
        """
        Process a single frame - detect and track people
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            annotated_frame: Frame with visualizations
            detections: List of detected people with their info
        """
        self.frame_count += 1
        
        # Run YOLO tracking
        results = self.model.track(
            frame,
            persist=True,               # Keep track IDs consistent
            tracker=self.tracker_type,  # Which tracker to use
            conf=self.conf_threshold,   # Confidence threshold
            iou=0.5,                    # IoU threshold for NMS
            classes=[0],                # Only detect people (class 0)
            verbose=False               # Don't print every frame
        )
        
        # Get the first result
        result = results[0]
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Store detection info
        detections = []
        
        # Check if we have any detections with track IDs
        if result.boxes is not None and result.boxes.id is not None:
            # Get boxes (xyxy format), track IDs, and confidences
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            # Process each detection
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                # Convert to integers
                x1, y1, x2, y2 = map(int, box)
                
                # Store detection info
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'track_id': int(track_id),
                    'confidence': float(conf)
                })
                
                # Get a color for this track ID
                color = self._get_color(track_id)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID label with background
                label = f"ID:{track_id} {conf:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_h - 8),
                    (x1 + label_w, y1),
                    color,
                    -1  # Filled
                )
                
                # Draw text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
                
                # Update track history for path visualization
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)
                
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(center)
                
                # Keep only last 30 points
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                # Draw track path
                points = self.track_history[track_id]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        cv2.line(
                            annotated_frame,
                            points[i-1],
                            points[i],
                            color,
                            2
                        )
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.last_time = current_time
        self.fps_list.append(fps)
        
        # Calculate average FPS (last 30 frames)
        avg_fps = sum(self.fps_list[-30:]) / min(len(self.fps_list), 30)
        
        # Draw FPS and count on frame
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
        
        # Show which tracker we're using
        tracker_name = self.tracker_type.split('.')[0]
        cv2.putText(
            annotated_frame,
            f"Tracker: {tracker_name}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        return annotated_frame, detections
    
    def _get_color(self, track_id):
        """
        Generate a consistent color for each track ID
        
        Args:
            track_id: The track ID
            
        Returns:
            color: BGR color tuple
        """
        np.random.seed(int(track_id) * 10 + 1)
        return tuple(map(int, np.random.randint(50, 255, 3)))
    
    def get_stats(self):
        """
        Get performance statistics
        
        Returns:
            dict: Dictionary with performance metrics
        """
        if self.fps_list:
            return {
                'total_frames': self.frame_count,
                'avg_fps': sum(self.fps_list) / len(self.fps_list),
                'max_fps': max(self.fps_list),
                'min_fps': min(self.fps_list)
            }
        return None
    
    def reset(self):
        """Reset tracker state for a new video"""
        self.frame_count = 0
        self.fps_list = []
        self.last_time = time.time()
        self.track_history = {}