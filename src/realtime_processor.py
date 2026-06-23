"""
Real-time video processor with threaded frame reading
Separates frame capture from processing for real-time performance
"""

import cv2
import time
import threading
import numpy as np
from collections import deque


class RealtimeProcessor:
    """
    Real-time video processor with separate reader and processing threads
    """
    
    def __init__(self, video_path, max_queue_size=2):
        """
        Initialize real-time processor
        
        Args:
            video_path: Path to video file or 0 for webcam
            max_queue_size: Maximum frames to keep in queue
        """
        self.video_path = video_path
        self.max_queue_size = max_queue_size
        
        # Threading controls
        self.running = False
        self.reader_thread = None
        self.processor_thread = None
        
        # Frame queue for passing frames between threads
        self.frame_queue = deque(maxlen=max_queue_size)
        self.frame_lock = threading.Lock()
        
        # Latest frame (for fast access)
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        
        # Video info
        self.fps = 30
        self.width = 640
        self.height = 480
        self.total_frames = 0
        
        # Statistics - using separate counters and timers
        self.read_count = 0
        self.process_count = 0
        self.read_fps = 0
        self.process_fps = 0
        self.last_read_time = time.time()
        self.last_process_time = time.time()
        
        # For FPS smoothing
        self.read_fps_history = deque(maxlen=10)
        self.process_fps_history = deque(maxlen=10)
        
        print("📹 Realtime Processor initialized")
    
    def start(self):
        """
        Start the real-time processing threads
        """
        if self.running:
            print("⚠️  Already running")
            return
        
        # Open video to get info
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return
        
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"✅ Video info: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Start threads
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
        
        self.reader_thread.start()
        self.processor_thread.start()
        
        print("🚀 Threads started - Reader and Processor running")
    
    def stop(self):
        """
        Stop the real-time processing
        """
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
        print("🛑 Threads stopped")
    
    def _reader_loop(self):
        """
        Reader thread - continuously reads frames from video
        """
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print(f"❌ Reader: Cannot open video")
            self.running = False
            return
        
        print("📖 Reader thread started")
        
        # Reset FPS tracking
        self.read_count = 0
        self.last_read_time = time.time()
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                # If video ends, loop back to start (for continuous streaming)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Store frame in queue
            with self.frame_lock:
                self.frame_queue.append(frame)
            
            # Store as latest frame
            with self.latest_frame_lock:
                self.latest_frame = frame.copy()
            
            # Update read FPS
            self.read_count += 1
            current_time = time.time()
            if current_time - self.last_read_time >= 1.0:
                self.read_fps = self.read_count
                self.read_fps_history.append(self.read_fps)
                self.read_count = 0
                self.last_read_time = current_time
        
        cap.release()
        print("📖 Reader thread stopped")
    
    def _processor_loop(self):
        """
        Processor thread - waits for frames and processes them
        """
        print("⚙️  Processor thread started")
        
        # Reset FPS tracking
        self.process_count = 0
        self.last_process_time = time.time()
        
        while self.running:
            # Get the latest frame from queue
            frame = None
            with self.frame_lock:
                if len(self.frame_queue) > 0:
                    # Get the most recent frame
                    frame = self.frame_queue.pop()
                    self.frame_queue.clear()  # Clear old frames (only keep latest)
            
            if frame is None:
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
                continue
            
            # Process the frame
            if self.process_callback:
                try:
                    self.process_callback(frame)
                except Exception as e:
                    print(f"❌ Processor error: {e}")
            
            # Update process FPS
            self.process_count += 1
            current_time = time.time()
            if current_time - self.last_process_time >= 1.0:
                self.process_fps = self.process_count
                self.process_fps_history.append(self.process_fps)
                self.process_count = 0
                self.last_process_time = current_time
        
        print("⚙️  Processor thread stopped")
    
    def get_latest_frame(self):
        """
        Get the latest frame (non-blocking)
        
        Returns:
            numpy.ndarray or None: Latest frame if available
        """
        with self.latest_frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def get_stats(self):
        """
        Get processing statistics
        
        Returns:
            dict: Statistics
        """
        # Get smoothed FPS values
        read_fps = sum(self.read_fps_history) / len(self.read_fps_history) if self.read_fps_history else 0
        process_fps = sum(self.process_fps_history) / len(self.process_fps_history) if self.process_fps_history else 0
        
        return {
            'read_fps': int(read_fps) if read_fps > 0 else 0,
            'process_fps': int(process_fps) if process_fps > 0 else 0,
            'queue_size': len(self.frame_queue),
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'fps': self.fps
        }