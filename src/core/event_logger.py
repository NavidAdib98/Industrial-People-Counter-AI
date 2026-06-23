"""
Event Logger for tracking enter/exit events
Logs events to CSV and JSON formats
"""

import csv
import json
import time
import logging
from datetime import datetime
from pathlib import Path


# Configure logger for this module
logger = logging.getLogger(__name__)


class EventLogger:
    """
    Logs enter/exit events for people tracking
    Only logs events for objects near the polygon boundary
    """
    
    def __init__(self, output_dir="outputs"):
        """
        Initialize event logger
        
        Args:
            output_dir: Directory to save log files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Events storage
        self.events = []
        self.event_id = 0
        
        # Track last known state for each track
        self.track_last_state = {}  # track_id -> 'inside' or 'outside'
        self.track_last_near_boundary = {}  # track_id -> bool
        
        # Current occupancy
        self.current_occupancy = 0
        
        # File paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = self.output_dir / f"events_{timestamp}.json"
        self.csv_path = self.output_dir / f"events_{timestamp}.csv"
        
        logger.info(f"Event Logger initialized")
        logger.info(f"   JSON: {self.json_path}")
        logger.info(f"   CSV: {self.csv_path}")
    
    def log_event(self, track_id, event_type, frame_id, confidence=1.0, 
                  video_time=None, timestamp=None):
        """
        Log an enter/exit event
        
        Args:
            track_id: The track ID
            event_type: 'ENTER' or 'EXIT'
            frame_id: Frame number
            confidence: Detection confidence
            video_time: Video time in seconds
            timestamp: System timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update occupancy
        if event_type == 'ENTER':
            self.current_occupancy += 1
        elif event_type == 'EXIT':
            self.current_occupancy -= 1
        
        # Ensure occupancy doesn't go negative
        self.current_occupancy = max(0, self.current_occupancy)
        
        # Create event record
        event = {
            'event_id': self.event_id,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'frame_id': frame_id,
            'video_time': round(video_time, 3) if video_time is not None else None,
            'track_id': track_id,
            'event_type': event_type,
            'occupancy_after': self.current_occupancy,
            'confidence': confidence
        }
        
        # Add to events list
        self.events.append(event)
        self.event_id += 1
        
        # Update track last state
        self.track_last_state[track_id] = 'inside' if event_type == 'ENTER' else 'outside'
    
    def check_and_log_state_change(self, track_id, is_inside, is_near_boundary, 
                                   frame_id, confidence=1.0, video_time=None, timestamp=None):
        """
        Check if track state has changed and log event ONLY if near boundary
        
        Args:
            track_id: The track ID
            is_inside: Current inside status
            is_near_boundary: Whether the object is near the polygon boundary
            frame_id: Frame number
            confidence: Detection confidence
            video_time: Video time in seconds
            timestamp: System timestamp
            
        Returns:
            bool: True if event was logged
        """
        # Get previous state
        prev_state = self.track_last_state.get(track_id, None)
        current_state = 'inside' if is_inside else 'outside'
        
        # Update near-boundary status
        self.track_last_near_boundary[track_id] = is_near_boundary
        
        # If first time seeing this track - just record state
        if prev_state is None:
            self.track_last_state[track_id] = current_state
            if is_inside:
                self.current_occupancy += 1
            return False
        
        # Only log event if object is NEAR BOUNDARY
        if is_near_boundary and prev_state != current_state:
            # State changed and object is near boundary - log event
            event_type = 'ENTER' if is_inside else 'EXIT'
            self.log_event(track_id, event_type, frame_id, confidence, video_time, timestamp)
            return True
        
        # If not near boundary, just update state without logging
        self.track_last_state[track_id] = current_state
        return False
    
    def get_current_occupancy(self):
        """
        Get current occupancy count
        
        Returns:
            int: Current occupancy
        """
        return self.current_occupancy
    
    def get_events(self):
        """
        Get all events
        
        Returns:
            list: List of event dictionaries
        """
        return self.events
    
    def save(self, format='both'):
        """
        Save events to file
        
        Args:
            format: 'json', 'csv', or 'both'
        """
        if format in ['json', 'both']:
            self._save_json()
        
        if format in ['csv', 'both']:
            self._save_csv()
        
        logger.info(f"Events saved: {len(self.events)} events")
    
    def _save_json(self):
        """Save events to JSON file"""
        data = {
            'total_events': len(self.events),
            'current_occupancy': self.current_occupancy,
            'events': self.events
        }
        
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_csv(self):
        """Save events to CSV file"""
        if not self.events:
            return
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['event_id', 'timestamp', 'datetime', 'frame_id', 'video_time',
                         'track_id', 'event_type', 'occupancy_after', 'confidence']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)
    
    def reset(self):
        """
        Reset logger (for new video)
        """
        self.events = []
        self.event_id = 0
        self.track_last_state = {}
        self.track_last_near_boundary = {}
        self.current_occupancy = 0
        logger.info("Event logger reset")
    
    def print_summary(self):
        """
        Print summary of events
        """
        if not self.events:
            logger.info("No events logged")
            return
        
        enters = sum(1 for e in self.events if e['event_type'] == 'ENTER')
        exits = sum(1 for e in self.events if e['event_type'] == 'EXIT')
        
        logger.info("=" * 50)
        logger.info("EVENT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Events: {len(self.events)}")
        logger.info(f"  ENTER: {enters}")
        logger.info(f"  EXIT: {exits}")
        logger.info(f"Current Occupancy: {self.current_occupancy}")
        logger.info(f"Unique Tracks: {len(set(e['track_id'] for e in self.events))}")
        logger.info("=" * 50)