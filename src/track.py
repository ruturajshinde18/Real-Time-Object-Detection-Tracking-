"""
Object Tracking Module
----------------------
Provides advanced tracking functionality using ByteTrack.
Manages track lifecycle and provides tracking statistics.
"""

import cv2
import time
import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


# Target classes
TARGET_CLASSES = [0, 1, 2, 3]
CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
}


class ObjectTracker:
    """
    Wrapper class for YOLOv8 + ByteTrack tracking.
    Maintains track history and provides statistics.
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model = YOLO(model_path)
        self.track_history = defaultdict(list)  # track_id -> list of positions
        self.track_classes = {}  # track_id -> class_id
        self.active_tracks = set()
        self.total_tracks = 0
    
    def update(self, frame):
        """
        Process frame and update tracks.
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            YOLO results object with tracking info
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=TARGET_CLASSES,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        # Update track history
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            
            current_ids = set()
            
            for box in boxes:
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                
                # Get center point
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Update history
                self.track_history[track_id].append((cx, cy))
                self.track_classes[track_id] = cls_id
                current_ids.add(track_id)
                
                # Limit history length
                if len(self.track_history[track_id]) > 50:
                    self.track_history[track_id].pop(0)
            
            # Track new objects
            new_tracks = current_ids - self.active_tracks
            self.total_tracks += len(new_tracks)
            self.active_tracks = current_ids
        
        return results[0]
    
    def draw_tracks(self, frame, result, draw_trails: bool = True):
        """
        Draw bounding boxes, labels, track IDs, and optional trails.
        
        Args:
            frame: Input frame
            result: YOLO result object
            draw_trails: Whether to draw movement trails
        
        Returns:
            Annotated frame
        """
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return frame
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw trail
            if draw_trails and track_id is not None:
                history = self.track_history.get(track_id, [])
                if len(history) > 1:
                    for i in range(1, len(history)):
                        pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                        pt2 = (int(history[i][0]), int(history[i][1]))
                        # Gradient thickness
                        thickness = int(1 + (i / len(history)) * 2)
                        cv2.line(frame, pt1, pt2, color, thickness)
            
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
    
    def get_stats(self):
        """Return tracking statistics."""
        return {
            'active_tracks': len(self.active_tracks),
            'total_tracks': self.total_tracks,
            'tracks_by_class': self._count_by_class()
        }
    
    def _count_by_class(self):
        """Count active tracks by class."""
        counts = defaultdict(int)
        for track_id in self.active_tracks:
            if track_id in self.track_classes:
                cls_id = self.track_classes[track_id]
                counts[CLASS_NAMES.get(cls_id, f'class_{cls_id}')] += 1
        return dict(counts)


def process_video_with_tracking(
    video_path: str,
    output_path: str = None,
    model_path: str = 'yolov8n.pt',
    resize_width: int = 640,
    show_display: bool = True,
    draw_trails: bool = True
):
    """
    Process video with full tracking capabilities.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        model_path: YOLO model path
        resize_width: Width to resize frames
        show_display: Whether to show live display
        draw_trails: Whether to draw movement trails
    """
    tracker = ObjectTracker(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    
    aspect_ratio = orig_height / orig_width
    resize_height = int(resize_width * aspect_ratio)
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, int(orig_fps), (resize_width, resize_height)
        )
    
    fps_list = []
    frame_count = 0
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {resize_width}x{resize_height}")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        frame = cv2.resize(frame, (resize_width, resize_height))
        
        result = tracker.update(frame)
        annotated = tracker.draw_tracks(frame, result, draw_trails)
        
        elapsed = time.time() - start_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(current_fps)
        
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        stats = tracker.get_stats()
        
        # Draw overlay
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Active: {stats['active_tracks']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Total: {stats['total_tracks']}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if writer:
            writer.write(annotated)
        
        if show_display:
            cv2.imshow('Object Tracking', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Final stats
    print(f"\n{'='*40}")
    print("TRACKING SUMMARY")
    print(f"{'='*40}")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {sum(fps_list) / len(fps_list):.1f}")
    print(f"Total unique objects tracked: {tracker.total_tracks}")
    print(f"Objects by class: {tracker.get_stats()['tracks_by_class']}")


def main():
    parser = argparse.ArgumentParser(description='Object Tracking on Video')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--width', type=int, default=640, help='Resize width')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--no-trails', action='store_true', help='Disable trails')
    
    args = parser.parse_args()
    
    process_video_with_tracking(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        resize_width=args.width,
        show_display=not args.no_display,
        draw_trails=not args.no_trails
    )


if __name__ == '__main__':
    main()
