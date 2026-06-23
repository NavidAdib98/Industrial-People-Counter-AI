"""
Real-time video processor with threaded frame reading
Separates frame capture from processing for real-time performance
"""

import cv2
import time
import threading
import numpy as np
from collections import deque


class FrameData:
    """
    Container for frame data including frame number and timestamp
    """
    def __init__(self, frame, frame_number, timestamp, video_time=None):
        self.frame = frame
        self.frame_number = frame_number  # Real video frame number (1-based)
        self.timestamp = timestamp        # System timestamp when frame was read
        self.video_time = video_time      # Video time in seconds (if available)


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
        
        # Latest frame data (with metadata)
        self.latest_frame_data = None
        self.latest_frame_lock = threading.Lock()
        
        # Video info
        self.fps = 30
        self.width = 640
        self.height = 480
        self.total_frames = 0
        self.frame_duration = 1.0 / 30
        self.video_duration = 0
        
        # FPS tracking
        self.read_count = 0
        self.process_count = 0
        self.read_fps = 0
        self.process_fps = 0
        self.last_read_time = time.time()
        self.last_process_time = time.time()
        self.read_fps_history = deque(maxlen=10)
        self.process_fps_history = deque(maxlen=10)
        
        # For FPS limiting
        self.last_frame_time = 0
        self.read_frame_number = 0
        
        # Callback for processing
        self.process_callback = None
        
        print("📹 Realtime Processor initialized")
    
    def start(self):
        """
        Start the real-time processing threads
        """
        if self.running:
            return
        
        # Get video info
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return
        
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.fps > 0 and self.total_frames > 0:
            self.video_duration = self.total_frames / self.fps
        
        cap.release()
        
        # Calculate frame duration for FPS limiting
        if self.fps > 0:
            self.frame_duration = 1.0 / self.fps
        else:
            self.frame_duration = 1.0 / 30
            self.fps = 30
        
        print(f"✅ Video: {self.width}x{self.height}, {self.fps} FPS")
        print(f"   Frame duration: {self.frame_duration * 1000:.1f}ms per frame")
        print(f"   Total frames: {self.total_frames}")
        print(f"   Video duration: {self.video_duration:.1f} seconds")
        
        # Reset frame counter
        self.read_frame_number = 0
        
        # Start threads
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
        
        self.reader_thread.start()
        self.processor_thread.start()
        print("🚀 Threads started")
    
    def stop(self):
        """Stop the real-time processing"""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
        print("🛑 Stopped")
    
    def _reader_loop(self):
        """
        Reader thread - reads frames at the video's FPS
        If reading is faster than video FPS, it waits
        If reading is slower, it just continues (can't catch up)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print("❌ Cannot open video")
            self.running = False
            return
        
        print(f"📖 Reader started (target: {self.fps} FPS, {self.frame_duration*1000:.1f}ms per frame)")
        self.read_count = 0
        self.last_read_time = time.time()
        self.last_frame_time = time.time()
        self.read_frame_number = 0
        
        while self.running:
            # Start timing this frame
            frame_start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                # If video ends, loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.read_frame_number = 0
                continue
            
            # Increment frame counter
            self.read_frame_number += 1
            
            # Get actual video frame number from OpenCV
            actual_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Calculate video time
            video_time = (self.read_frame_number - 1) / self.fps if self.fps > 0 else 0
            
            # Create FrameData object
            frame_data = FrameData(
                frame=frame,
                frame_number=actual_frame_number,
                timestamp=time.time(),
                video_time=video_time
            )
            
            # Store frame data in queue
            with self.frame_lock:
                self.frame_queue.append(frame_data)
            
            # Store as latest frame data
            with self.latest_frame_lock:
                self.latest_frame_data = frame_data
            
            # Update read FPS
            self.read_count += 1
            current_time = time.time()
            if current_time - self.last_read_time >= 1.0:
                self.read_fps = self.read_count
                self.read_fps_history.append(self.read_fps)
                self.read_count = 0
                self.last_read_time = current_time
            
            # ❗ FPS LIMITER: Wait if we read faster than video FPS
            elapsed = time.time() - frame_start_time
            sleep_time = self.frame_duration - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()
        print("📖 Reader stopped")
    
    def _processor_loop(self):
        """
        Processor thread - waits for frames and processes them
        """
        print("⚙️  Processor started")
        self.process_count = 0
        self.last_process_time = time.time()
        
        while self.running:
            # Get the latest frame data from queue
            frame_data = None
            with self.frame_lock:
                if len(self.frame_queue) > 0:
                    frame_data = self.frame_queue.pop()
                    self.frame_queue.clear()
            
            if frame_data is None:
                time.sleep(0.001)
                continue
            
            # Process the frame with full metadata
            if self.process_callback:
                try:
                    self.process_callback(frame_data)
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
        
        print("⚙️  Processor stopped")
    
    def get_latest_frame_data(self):
        """
        Get the latest frame data with metadata
        
        Returns:
            FrameData or None: Latest frame data if available
        """
        with self.latest_frame_lock:
            if self.latest_frame_data is not None:
                return FrameData(
                    frame=self.latest_frame_data.frame.copy(),
                    frame_number=self.latest_frame_data.frame_number,
                    timestamp=self.latest_frame_data.timestamp,
                    video_time=self.latest_frame_data.video_time
                )
            return None
    
    def get_latest_frame(self):
        """
        Get the latest frame only (for backward compatibility)
        
        Returns:
            numpy.ndarray or None: Latest frame if available
        """
        frame_data = self.get_latest_frame_data()
        return frame_data.frame if frame_data else None
    
    def get_stats(self):
        """
        Get processing statistics
        
        Returns:
            dict: Statistics
        """
        read_fps = sum(self.read_fps_history) / len(self.read_fps_history) if self.read_fps_history else 0
        process_fps = sum(self.process_fps_history) / len(self.process_fps_history) if self.process_fps_history else 0
        
        return {
            'read_fps': int(read_fps) if read_fps > 0 else 0,
            'process_fps': int(process_fps) if process_fps > 0 else 0,
            'target_fps': self.fps,
            'queue_size': len(self.frame_queue),
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'video_duration': self.video_duration
        }