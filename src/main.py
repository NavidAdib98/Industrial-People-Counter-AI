"""
Main script for running the person tracker on a video
Simple and easy to understand
"""

import cv2
import os
from person_tracker import PersonTracker


def main():
    """
    Main function - runs the tracker on a video
    """
    print("=" * 60)
    print("👥 INDUSTRIAL PEOPLE TRACKER")
    print("=" * 60)
    print()
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # ============================================
    # CONFIGURATION - Change these as needed
    # ============================================
    
    # Video path (put your video in the 'videos' folder)
    VIDEO_PATH = "videos/Hike_Vision.mp4"
    # For webcam, use: VIDEO_PATH = 0
    
    # Model settings
    MODEL_NAME = "yolo26n.pt"         # Smallest model for CPU
    TRACKER_TYPE = "bytetrack.yaml"   # Good for occlusion handling
    CONF_THRESHOLD = 0.4              # Lower = more detections
    DEVICE = "cpu"                    # 'cpu' or 'cuda'
    
    # Output settings
    SAVE_OUTPUT = False                # Save annotated video
    OUTPUT_PATH = "outputs/tracked_video.mp4"
    
    # Display settings
    SHOW_PREVIEW = True               # Show video window
    
    # ============================================
    
    # 1. Initialize tracker
    tracker = PersonTracker(
        model_name=MODEL_NAME,
        tracker_type=TRACKER_TYPE,
        conf_threshold=CONF_THRESHOLD,
        device=DEVICE
    )
    
    # 2. Open video
    print(f"📹 Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video!")
        print(f"   Make sure the file exists: {VIDEO_PATH}")
        print()
        print("   You can:")
        print("   1. Put a video file in the 'videos' folder")
        print("   2. Change VIDEO_PATH to your video path")
        print("   3. Use webcam by setting VIDEO_PATH = 0")
        return
    
    # Get video properties for saving output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video loaded: {width}x{height}, {fps} FPS, {total_frames} frames")
    print()
    
    # 3. Setup video writer (to save output)
    video_writer = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
        print(f"💾 Output will be saved to: {OUTPUT_PATH}")
        print()
    
    # 4. Process video frame by frame
    print("🚀 Starting tracking...")
    print("   Press 'q' or ESC to quit")
    print("-" * 60)
    
    frame_count = 0
    
    while True:
        # Read frame
        success, frame = cap.read()
        
        if not success:
            break
        
        frame_count += 1
        
        # Process frame
        annotated_frame, detections = tracker.process_frame(frame)
        
        # Save frame
        if SAVE_OUTPUT and video_writer:
            video_writer.write(annotated_frame)
        
        # Show progress
        progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
        print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - {len(detections)} people", end='\r')
        
        # Display frame
        if SHOW_PREVIEW:
            cv2.imshow('People Tracker', annotated_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
    
    print()  # New line after progress
    print("-" * 60)
    
    # 5. Show statistics
    print()
    print("📊 FINAL STATISTICS")
    print("=" * 60)
    
    stats = tracker.get_stats()
    if stats:
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Max FPS: {stats['max_fps']:.2f}")
        print(f"Min FPS: {stats['min_fps']:.2f}")
    
    print()
    if SAVE_OUTPUT:
        print(f"✅ Output video saved to: {OUTPUT_PATH}")
    
    # 6. Clean up
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print()
    print("🎉 Done!")


if __name__ == "__main__":
    main()