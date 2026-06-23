"""
Main script - runs the person tracker in real-time
"""

import cv2
import time
import threading
from tracker import PersonTracker
from visualizer import Visualizer
from settings import settings
from realtime_processor import RealtimeProcessor


class RealtimeTracker:
    """
    Real-time tracker that processes frames as they arrive
    """
    
    def __init__(self):
        """Initialize real-time tracker"""
        print("=" * 50)
        print("👥 REAL-TIME INDUSTRIAL PEOPLE TRACKER")
        print("=" * 50)
        print()
        
        # Show settings
        settings.print_settings()
        
        # Initialize tracker and visualizer
        self.tracker = PersonTracker(settings)
        self.visualizer = Visualizer()
        
        # Latest processed results
        self.latest_annotated_frame = None
        self.latest_tracker_data = None
        self.result_lock = threading.Lock()
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.processed_frames_for_save = []  # Store frames for saving
        self.save_fps = 0
        
        # Setup video writer
        self.video_writer = None
        self.output_width = 0
        self.output_height = 0
        self.original_fps = 30
    
    def process_frame(self, frame):
        """
        Process a single frame - called by the processor thread
        
        Args:
            frame: Input frame
        """
        self.frame_count += 1
        
        # Resize if enabled
        if settings.RESIZE_VIDEO:
            frame = cv2.resize(frame, (settings.RESIZE_WIDTH, settings.RESIZE_HEIGHT))
            self.output_width = settings.RESIZE_WIDTH
            self.output_height = settings.RESIZE_HEIGHT
        else:
            self.output_width = frame.shape[1]
            self.output_height = frame.shape[0]
        
        # Process frame (this is the slow part)
        tracker_data = self.tracker.process_frame(frame)
        
        # Annotate frame
        annotated_frame = self.visualizer.annotate_frame(frame, tracker_data)
        
        # Store results
        with self.result_lock:
            self.latest_annotated_frame = annotated_frame
            self.latest_tracker_data = tracker_data
        
        # Save frame if enabled - store for later writing
        if settings.SAVE_OUTPUT:
            self.processed_frames_for_save.append(annotated_frame)
    
    def _calculate_save_fps(self):
        """Calculate the actual processing FPS for saving"""
        if self.frame_count > 0 and self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                return self.frame_count / elapsed
        return 5  # Default fallback
    
    def _write_saved_video(self):
        """Write the saved frames to a video file with correct FPS"""
        if not self.processed_frames_for_save:
            print("⚠️  No frames to save")
            return
        
        # Calculate actual processing FPS
        self.save_fps = self._calculate_save_fps()
        print(f"📊 Saving video at {self.save_fps:.1f} FPS (actual processing speed)")
        
        # Ensure FPS is reasonable (min 1, max 30)
        save_fps = max(1, min(30, self.save_fps))
        
        # Get dimensions from first frame
        first_frame = self.processed_frames_for_save[0]
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = settings.OUTPUT_VIDEO
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            save_fps,
            (width, height)
        )
        
        # Write all frames
        for frame in self.processed_frames_for_save:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"✅ Video saved: {output_path}")
        print(f"   Frames: {len(self.processed_frames_for_save)}")
        print(f"   FPS: {save_fps:.1f}")
        print(f"   Duration: {len(self.processed_frames_for_save) / save_fps:.1f} seconds")
    
    def run(self):
        """
        Main loop - runs the real-time processor
        """
        # Initialize processor
        processor = RealtimeProcessor(settings.VIDEO_PATH)
        processor.process_callback = self.process_frame
        
        # Start processor
        processor.start()
        
        print("🚀 Real-time tracking started... Press 'q' to quit")
        print("-" * 50)
        
        # Display counter
        last_display_time = time.time()
        
        try:
            # Main display loop
            while True:
                # Get latest results
                with self.result_lock:
                    annotated_frame = self.latest_annotated_frame
                    tracker_data = self.latest_tracker_data
                
                # Display if available
                if annotated_frame is not None and settings.SHOW_PREVIEW:
                    # Get processor stats
                    stats = processor.get_stats()
                    
                    # Update display every 0.5 seconds
                    current_time = time.time()
                    if current_time - last_display_time >= 0.5:
                        last_display_time = current_time
                        
                        # Show clear status
                        if tracker_data:
                            status = (f"Read: {stats['read_fps']} FPS | "
                                     f"Process: {stats['process_fps']} FPS | "
                                     f"Inside: {tracker_data['people_inside']} | "
                                     f"Outside: {tracker_data['people_outside']} | "
                                     f"Total: {tracker_data['total']} | "
                                     f"Queue: {stats['queue_size']}")
                            print(status, end='\r')
                    
                    cv2.imshow('Real-Time People Tracker', annotated_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Stop processor
            processor.stop()
            cv2.destroyAllWindows()
        
        # Show final statistics
        print("\n" + "-" * 50)
        print()
        print("📊 FINAL STATISTICS")
        print("=" * 50)
        
        stats = self.tracker.get_stats()
        if stats:
            print(f"Total Frames Processed: {stats['total_frames']}")
            print(f"Average Process FPS: {stats['avg_fps']:.2f}")
        
        counts = self.tracker.get_counts()
        print()
        print("📊 PEOPLE COUNTS")
        print("=" * 50)
        print(f"People Inside: {counts['inside']}")
        print(f"People Outside: {counts['outside']}")
        print(f"Total People: {counts['total']}")
        
        elapsed = time.time() - self.start_time
        print(f"\n⏱️  Total runtime: {elapsed:.1f} seconds")
        
        # Write saved video
        if settings.SAVE_OUTPUT and self.processed_frames_for_save:
            self._write_saved_video()
        
        print()
        print("🎉 Done!")


def main():
    """Main function"""
    rt_tracker = RealtimeTracker()
    rt_tracker.run()


if __name__ == "__main__":
    main()