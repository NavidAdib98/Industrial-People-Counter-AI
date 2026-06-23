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
        
        # Setup video writer (for saving output)
        self.video_writer = None
        if settings.SAVE_OUTPUT:
            # We'll initialize this when we know the video size
            pass
    
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
        
        # Process frame (this is the slow part)
        tracker_data = self.tracker.process_frame(frame)
        
        # Annotate frame
        annotated_frame = self.visualizer.annotate_frame(frame, tracker_data)
        
        # Store results
        with self.result_lock:
            self.latest_annotated_frame = annotated_frame
            self.latest_tracker_data = tracker_data
        
        # Save if enabled
        if settings.SAVE_OUTPUT and self.video_writer:
            self.video_writer.write(annotated_frame)
    
    def run(self):
        """
        Main loop - runs the real-time processor
        """
        # Initialize processor
        processor = RealtimeProcessor(settings.VIDEO_PATH)
        processor.process_callback = self.process_frame
        
        # Initialize video writer with correct dimensions
        if settings.SAVE_OUTPUT:
            width = settings.RESIZE_WIDTH if settings.RESIZE_VIDEO else processor.width
            height = settings.RESIZE_HEIGHT if settings.RESIZE_VIDEO else processor.height
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(settings.OUTPUT_VIDEO),
                fourcc,
                processor.fps,
                (width, height)
            )
            print(f"💾 Output: {settings.OUTPUT_VIDEO}")
            print()
        
        # Start processor
        processor.start()
        
        print("🚀 Real-time tracking started... Press 'q' to quit")
        print("-" * 50)
        
        # FPS display counter
        display_counter = 0
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
                        display_counter += 1
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
                
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            processor.stop()
            if self.video_writer:
                self.video_writer.release()
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
        
        if settings.SAVE_OUTPUT:
            print(f"✅ Output saved: {settings.OUTPUT_VIDEO}")
        
        print()
        print("🎉 Done!")


def main():
    """Main function"""
    # Create and run real-time tracker
    rt_tracker = RealtimeTracker()
    rt_tracker.run()


if __name__ == "__main__":
    main()