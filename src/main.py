"""
Main script - runs the person tracker on a video
"""

import cv2
from person_tracker import PersonTracker
from settings import settings


def main():
    """Main function"""
    print("=" * 50)
    print("👥 INDUSTRIAL PEOPLE TRACKER")
    print("=" * 50)
    print()
    
    # Show settings
    settings.print_settings()
    
    # Initialize tracker
    tracker = PersonTracker(settings)
    
    # Open video
    print(f"📹 Opening: {settings.VIDEO_PATH}")
    cap = cv2.VideoCapture(str(settings.VIDEO_PATH))
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video!")
        print(f"   Check VIDEO_PATH in .env file")
        return
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    print()
    
    # Setup video writer
    video_writer = None
    if settings.SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(settings.OUTPUT_VIDEO),
            fourcc,
            fps,
            (width, height)
        )
        print(f"💾 Output: {settings.OUTPUT_VIDEO}")
        print()
    
    # Process video
    print("🚀 Tracking started... Press 'q' to quit")
    print("-" * 50)
    
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Process frame
        annotated_frame, detections = tracker.process_frame(frame)
        
        # Save frame
        if settings.SAVE_OUTPUT and video_writer:
            video_writer.write(annotated_frame)
        
        # Show progress
        progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
        print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - {len(detections)} people", end='\r')
        
        # Show preview
        if settings.SHOW_PREVIEW:
            cv2.imshow('People Tracker', annotated_frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    
    print("\n" + "-" * 50)
    
    # Show statistics
    print()
    print("📊 STATISTICS")
    print("=" * 50)
    
    stats = tracker.get_stats()
    if stats:
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Max FPS: {stats['max_fps']:.2f}")
        print(f"Min FPS: {stats['min_fps']:.2f}")
    
    print()
    if settings.SAVE_OUTPUT:
        print(f"✅ Output saved: {settings.OUTPUT_VIDEO}")
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print()
    print("🎉 Done!")


if __name__ == "__main__":
    main()