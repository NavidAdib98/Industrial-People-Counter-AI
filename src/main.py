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
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine output size
    if settings.RESIZE_VIDEO:
        output_width = settings.RESIZE_WIDTH
        output_height = settings.RESIZE_HEIGHT
        print(f"✅ Original: {original_width}x{original_height}, {original_fps} FPS")
        print(f"✅ Resizing to: {output_width}x{output_height}")
    else:
        output_width = original_width
        output_height = original_height
        print(f"✅ Video: {original_width}x{original_height}, {original_fps} FPS, {total_frames} frames")
    print()
    
    # Setup video writer
    video_writer = None
    if settings.SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(settings.OUTPUT_VIDEO),
            fourcc,
            original_fps,
            (output_width, output_height)
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
        
        # Resize if enabled
        if settings.RESIZE_VIDEO:
            frame = cv2.resize(frame, (output_width, output_height))
        
        # Process frame
        annotated_frame, detections = tracker.process_frame(frame)
        
        # Save
        if settings.SAVE_OUTPUT and video_writer:
            video_writer.write(annotated_frame)
        
        # Progress
        counts = tracker.get_counts()
        progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
        print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
              f"Inside: {counts['inside']}, Outside: {counts['outside']}, "
              f"Total: {counts['total']}", end='\r')
        
        # Preview
        if settings.SHOW_PREVIEW:
            cv2.imshow('People Tracker', annotated_frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    
    print("\n" + "-" * 50)
    
    # Statistics
    print()
    print("📊 STATISTICS")
    print("=" * 50)
    
    stats = tracker.get_stats()
    if stats:
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Max FPS: {stats['max_fps']:.2f}")
        print(f"Min FPS: {stats['min_fps']:.2f}")
    
    counts = tracker.get_counts()
    print()
    print("📊 PEOPLE COUNTS")
    print("=" * 50)
    print(f"People Inside: {counts['inside']}")
    print(f"People Outside: {counts['outside']}")
    print(f"Total People: {counts['total']}")
    
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