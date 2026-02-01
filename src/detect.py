"""
Object Detection Module
-----------------------
Uses YOLOv8 for real-time object detection on video streams.
Integrates with tracking module for persistent object IDs.
"""

import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO


# Target classes from COCO dataset
# 0: person, 1: bicycle, 2: car, 3: motorcycle
TARGET_CLASSES = [0, 1, 2, 3]
CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}

# Colors for each class (BGR format)
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for person
    1: (255, 0, 0),    # Blue for bicycle
    2: (0, 0, 255),    # Red for car
    3: (255, 255, 0),  # Cyan for motorcycle
}


def load_model(model_path: str = 'yolov8n.pt') -> YOLO:
    """Load YOLOv8 model (downloads automatically if not present)."""
    model = YOLO(model_path)
    return model


def process_video(
    video_path: str,
    output_path: str = None,
    model: YOLO = None,
    target_fps: int = 30,
    resize_width: int = 640,
    show_display: bool = True,
    enable_tracking: bool = True
):
    """
    Process video with object detection and optional tracking.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video (optional)
        model: YOLO model instance
        target_fps: Target FPS for processing
        resize_width: Width to resize frames to (maintains aspect ratio)
        show_display: Whether to show live display
        enable_tracking: Whether to enable object tracking
    """
    if model is None:
        model = load_model()
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate resize dimensions
    aspect_ratio = orig_height / orig_width
    resize_height = int(resize_width * aspect_ratio)
    
    # Frame skip for FPS control
    frame_skip = max(1, int(orig_fps / target_fps))
    
    # Video writer setup
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, target_fps, (resize_width, resize_height)
        )
    
    frame_count = 0
    fps_list = []
    
    print(f"Processing video: {video_path}")
    print(f"Original: {orig_width}x{orig_height} @ {orig_fps:.1f} FPS")
    print(f"Processing: {resize_width}x{resize_height} @ {target_fps} FPS")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for FPS control
        if frame_count % frame_skip != 0:
            continue
        
        # Start timing
        start_time = time.time()
        
        # Resize frame
        frame = cv2.resize(frame, (resize_width, resize_height))
        
        # Run detection or tracking
        if enable_tracking:
            results = model.track(
                frame,
                persist=True,
                classes=TARGET_CLASSES,
                verbose=False,
                tracker="bytetrack.yaml"
            )
        else:
            results = model(
                frame,
                classes=TARGET_CLASSES,
                verbose=False
            )
        
        # Draw results
        annotated_frame = draw_detections(frame, results[0], enable_tracking)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(current_fps)
        
        # Draw FPS overlay
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        cv2.putText(
            annotated_frame,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Write output
        if writer:
            writer.write(annotated_frame)
        
        # Display
        if show_display:
            cv2.imshow('Object Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Print stats
    if fps_list:
        print(f"\nProcessing complete!")
        print(f"Frames processed: {len(fps_list)}")
        print(f"Average FPS: {sum(fps_list) / len(fps_list):.1f}")


def draw_detections(frame, result, show_track_ids: bool = True):
    """
    Draw bounding boxes, labels, and track IDs on frame.
    
    Args:
        frame: Input frame (numpy array)
        result: YOLO result object
        show_track_ids: Whether to show tracking IDs
    
    Returns:
        Annotated frame
    """
    boxes = result.boxes
    
    if boxes is None or len(boxes) == 0:
        return frame
    
    for i, box in enumerate(boxes):
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Get class and confidence
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
        
        # Get track ID if available
        track_id = None
        if show_track_ids and box.id is not None:
            track_id = int(box.id[0])
        
        # Get color for this class
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        if track_id is not None:
            label = f"ID:{track_id} {cls_name} {conf:.2f}"
        else:
            label = f"{cls_name} {conf:.2f}"
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return frame


def main():
    parser = argparse.ArgumentParser(description='Object Detection on Video')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    parser.add_argument('--width', type=int, default=640, help='Resize width')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--no-track', action='store_true', help='Disable tracking')
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    
    process_video(
        video_path=args.video,
        output_path=args.output,
        model=model,
        target_fps=args.fps,
        resize_width=args.width,
        show_display=not args.no_display,
        enable_tracking=not args.no_track
    )


if __name__ == '__main__':
    main()
