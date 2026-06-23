"""
Visualizer - Handles all drawing/visualization on frames
Uses OpenCV directly for maximum compatibility
"""

import cv2
import numpy as np


class Visualizer:
    """
    Handles all drawing operations using OpenCV
    """
    
    def __init__(self):
        """Initialize visualizer"""
        pass
    
    def draw_polygon(self, frame, polygon_points, color, alpha=0.3, name=None):
        """
        Draw polygon on frame with transparency
        
        Args:
            frame: Input frame
            polygon_points: List of polygon points (pixel coordinates)
            color: BGR color tuple
            alpha: Transparency (0-1)
            name: Polygon name to display
            
        Returns:
            frame: Frame with polygon drawn
        """
        if polygon_points is None or len(polygon_points) < 3:
            return frame
        
        # Create overlay for transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon_points], color)
        
        # Blend with original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw outline
        cv2.polylines(frame, [polygon_points], True, color, 2)
        
        # Draw name in center of polygon
        if name:
            moments = cv2.moments(polygon_points)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                cv2.putText(frame, name, (cx - 40, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def draw_detections(self, frame, detections, color_inside, color_outside):
        """
        Draw bounding boxes, IDs, and center dots for detections
        
        Args:
            frame: Input frame
            detections: sv.Detections object
            color_inside: BGR color for inside polygon
            color_outside: BGR color for outside polygon
            
        Returns:
            frame: Frame with detections drawn
        """
        if len(detections) == 0:
            return frame
        
        # Check if we have inside/outside info
        has_inside_info = hasattr(detections, 'is_inside')
        
        for i in range(len(detections)):
            # Get bbox
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            
            # Get track ID
            track_id = detections.tracker_id[i] if detections.tracker_id is not None else i
            
            # Determine color
            if has_inside_info and i < len(detections.is_inside):
                is_inside = detections.is_inside[i]
                color = color_inside if is_inside else color_outside
            else:
                color = color_inside
                is_inside = True
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{track_id}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Background for label
            cv2.rectangle(frame, (x1, y1 - label_h - 8), (x1 + label_w, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center dot (bottom center for feet position)
            center_x = (x1 + x2) // 2
            bottom_y = y2
            cv2.circle(frame, (center_x, bottom_y), 4, color, -1)
        
        return frame
    
    def draw_info_panel(self, frame, fps, people_inside, people_outside):
        """
        Draw transparent info panel with statistics
        
        Args:
            frame: Input frame
            fps: Current FPS
            people_inside: Number of people inside polygon
            people_outside: Number of people outside polygon
            
        Returns:
            frame: Frame with info panel
        """
        # Panel settings
        panel_x = 10
        panel_y = 10
        panel_width = 170
        panel_height = 120
        padding = 10
        line_spacing = 25
        
        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0),  # Black
                     -1)  # Filled
        
        # Blend with original frame (60% opacity)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (100, 100, 100),  # Gray border
                     1)
        
        # Starting position for text
        text_x = panel_x + padding
        text_y = panel_y + padding + 15
        
        # All text in white
        color = (255, 255, 255)
        
        # FPS
        cv2.putText(frame, f">> FPS: {fps:.1f}", 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        text_y += line_spacing
        
        # Inside
        cv2.putText(frame, f"[+] Inside: {people_inside}", 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        text_y += line_spacing
        
        # Outside
        cv2.putText(frame, f"[-] Outside: {people_outside}", 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        text_y += line_spacing
        
        # Total
        total = people_inside + people_outside
        cv2.putText(frame, f"[*] Total: {total}", 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        
        return frame
    
    def annotate_frame(self, frame, tracker_data):
        """
        Full annotation pipeline - draws everything on frame
        
        Args:
            frame: Input frame
            tracker_data: Data from PersonTracker.process_frame()
            
        Returns:
            frame: Fully annotated frame
        """
        # Draw polygon (behind detections)
        if tracker_data['polygon_points'] is not None:
            frame = self.draw_polygon(
                frame,
                tracker_data['polygon_points'],
                tracker_data['color_inside'],
                tracker_data['polygon_alpha'],
                tracker_data['polygon_name']
            )
        
        # Draw detections
        if len(tracker_data['detections']) > 0:
            frame = self.draw_detections(
                frame,
                tracker_data['detections'],
                tracker_data['color_inside'],
                tracker_data['color_outside']
            )
        
        # Draw info panel (on top)
        frame = self.draw_info_panel(
            frame,
            tracker_data['fps'],
            tracker_data['people_inside'],
            tracker_data['people_outside']
        )
        
        return frame