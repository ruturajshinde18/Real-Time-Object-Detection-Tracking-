"""
Evaluation Module
-----------------
Computes detection and tracking metrics:
- Precision, Recall, F1-Score
- FPS (real-time performance)
- Per-class statistics
"""

import cv2
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from ultralytics import YOLO


TARGET_CLASSES = [0, 1, 2, 3]
CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def match_detections(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Match predictions to ground truths using IoU.
    
    Args:
        predictions: List of {'box': [x1,y1,x2,y2], 'class': int, 'conf': float}
        ground_truths: List of {'box': [x1,y1,x2,y2], 'class': int}
        iou_threshold: Minimum IoU for a match
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    tp = 0
    matched_gt = set()
    
    # Sort predictions by confidence (descending)
    preds_sorted = sorted(predictions, key=lambda x: x['conf'], reverse=True)
    
    for pred in preds_sorted:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if pred['class'] != gt['class']:
                continue
            
            iou = compute_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
    
    fp = len(predictions) - tp
    fn = len(ground_truths) - len(matched_gt)
    
    return tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute precision, recall, and F1-score.
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
    
    Returns:
        Dictionary with precision, recall, f1
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


class DetectionEvaluator:
    """
    Evaluates detection performance on video.
    Supports ground truth labels in YOLO format.
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model = YOLO(model_path)
        self.all_predictions = []
        self.all_ground_truths = []
        self.fps_measurements = []
        self.per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    def load_ground_truth(self, label_path: str, img_width: int, img_height: int) -> List[Dict]:
        """
        Load ground truth from YOLO format label file.
        
        Format: class_id center_x center_y width height (normalized)
        
        Args:
            label_path: Path to label file
            img_width: Image width
            img_height: Image height
        
        Returns:
            List of ground truth dictionaries
        """
        ground_truths = []
        
        if not Path(label_path).exists():
            return ground_truths
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                
                # Convert normalized to pixel coordinates
                x1 = (cx - w/2) * img_width
                y1 = (cy - h/2) * img_height
                x2 = (cx + w/2) * img_width
                y2 = (cy + h/2) * img_height
                
                ground_truths.append({
                    'box': [x1, y1, x2, y2],
                    'class': cls_id
                })
        
        return ground_truths
    
    def evaluate_frame(
        self,
        frame: np.ndarray,
        ground_truths: Optional[List[Dict]] = None,
        iou_threshold: float = 0.5
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Evaluate detection on a single frame.
        
        Args:
            frame: Input frame
            ground_truths: Optional ground truth annotations
            iou_threshold: IoU threshold for matching
        
        Returns:
            Tuple of (predictions, metrics)
        """
        start_time = time.time()
        
        results = self.model(frame, classes=TARGET_CLASSES, verbose=False)
        
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_measurements.append(fps)
        
        predictions = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                predictions.append({
                    'box': box.xyxy[0].tolist(),
                    'class': int(box.cls[0]),
                    'conf': float(box.conf[0])
                })
        
        self.all_predictions.extend(predictions)
        
        metrics = {'fps': fps}
        
        if ground_truths:
            self.all_ground_truths.extend(ground_truths)
            tp, fp, fn = match_detections(predictions, ground_truths, iou_threshold)
            metrics.update(compute_metrics(tp, fp, fn))
            
            # Update per-class stats
            for pred in predictions:
                cls_name = CLASS_NAMES.get(pred['class'], 'unknown')
                matched = any(
                    gt['class'] == pred['class'] and compute_iou(pred['box'], gt['box']) >= iou_threshold
                    for gt in ground_truths
                )
                if matched:
                    self.per_class_stats[cls_name]['tp'] += 1
                else:
                    self.per_class_stats[cls_name]['fp'] += 1
            
            for gt in ground_truths:
                cls_name = CLASS_NAMES.get(gt['class'], 'unknown')
                matched = any(
                    pred['class'] == gt['class'] and compute_iou(pred['box'], gt['box']) >= iou_threshold
                    for pred in predictions
                )
                if not matched:
                    self.per_class_stats[cls_name]['fn'] += 1
        
        return predictions, metrics
    
    def get_summary(self) -> Dict:
        """Get overall evaluation summary."""
        avg_fps = sum(self.fps_measurements) / len(self.fps_measurements) if self.fps_measurements else 0
        
        summary = {
            'total_frames': len(self.fps_measurements),
            'total_predictions': len(self.all_predictions),
            'total_ground_truths': len(self.all_ground_truths),
            'avg_fps': avg_fps,
            'min_fps': min(self.fps_measurements) if self.fps_measurements else 0,
            'max_fps': max(self.fps_measurements) if self.fps_measurements else 0,
        }
        
        # Overall metrics if we have ground truths
        if self.all_ground_truths:
            total_tp = sum(s['tp'] for s in self.per_class_stats.values())
            total_fp = sum(s['fp'] for s in self.per_class_stats.values())
            total_fn = sum(s['fn'] for s in self.per_class_stats.values())
            summary['overall_metrics'] = compute_metrics(total_tp, total_fp, total_fn)
            
            # Per-class metrics
            summary['per_class_metrics'] = {}
            for cls_name, stats in self.per_class_stats.items():
                summary['per_class_metrics'][cls_name] = compute_metrics(
                    stats['tp'], stats['fp'], stats['fn']
                )
        
        return summary


def evaluate_video(
    video_path: str,
    labels_dir: str = None,
    model_path: str = 'yolov8n.pt',
    resize_width: int = 640,
    output_json: str = None,
    show_display: bool = True
):
    """
    Evaluate detection performance on a video.
    
    Args:
        video_path: Path to input video
        labels_dir: Directory with YOLO format labels (frame_0001.txt, etc.)
        model_path: YOLO model path
        resize_width: Width to resize frames
        output_json: Path to save results as JSON
        show_display: Whether to show live display
    """
    evaluator = DetectionEvaluator(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    aspect_ratio = orig_height / orig_width
    resize_height = int(resize_width * aspect_ratio)
    
    frame_idx = 0
    
    print(f"Evaluating video: {video_path}")
    print(f"Resolution: {resize_width}x{resize_height}")
    if labels_dir:
        print(f"Labels directory: {labels_dir}")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (resize_width, resize_height))
        
        # Load ground truth if available
        ground_truths = None
        if labels_dir:
            label_file = Path(labels_dir) / f"frame_{frame_idx:04d}.txt"
            ground_truths = evaluator.load_ground_truth(
                str(label_file), resize_width, resize_height
            )
        
        predictions, metrics = evaluator.evaluate_frame(frame, ground_truths)
        
        # Draw predictions
        for pred in predictions:
            x1, y1, x2, y2 = map(int, pred['box'])
            cls_name = CLASS_NAMES.get(pred['class'], 'unknown')
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {pred['conf']:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw ground truths if available
        if ground_truths:
            for gt in ground_truths:
                x1, y1, x2, y2 = map(int, gt['box'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw metrics overlay
        cv2.putText(frame, f"FPS: {metrics['fps']:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if 'precision' in metrics:
            cv2.putText(frame, f"P:{metrics['precision']:.2f} R:{metrics['recall']:.2f} F1:{metrics['f1']:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if show_display:
            cv2.imshow('Evaluation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    summary = evaluator.get_summary()
    
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Total detections: {summary['total_predictions']}")
    print(f"\nFPS Performance:")
    print(f"  Average: {summary['avg_fps']:.1f}")
    print(f"  Min: {summary['min_fps']:.1f}")
    print(f"  Max: {summary['max_fps']:.1f}")
    
    if 'overall_metrics' in summary:
        m = summary['overall_metrics']
        print(f"\nOverall Detection Metrics:")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall: {m['recall']:.4f}")
        print(f"  F1-Score: {m['f1']:.4f}")
        print(f"  TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}")
        
        if summary.get('per_class_metrics'):
            print(f"\nPer-Class Metrics:")
            for cls_name, cls_metrics in summary['per_class_metrics'].items():
                print(f"  {cls_name}:")
                print(f"    P: {cls_metrics['precision']:.4f}, R: {cls_metrics['recall']:.4f}, F1: {cls_metrics['f1']:.4f}")
    
    # Save to JSON
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_json}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate Detection Performance')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--labels', type=str, default=None, help='Labels directory')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--width', type=int, default=640, help='Resize width')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    args = parser.parse_args()
    
    evaluate_video(
        video_path=args.video,
        labels_dir=args.labels,
        model_path=args.model,
        resize_width=args.width,
        output_json=args.output,
        show_display=not args.no_display
    )


if __name__ == '__main__':
    main()
