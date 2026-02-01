# Real-Time Object Detection and Tracking from Video Streams

A Python application that detects and tracks objects (people, vehicles) in video streams using YOLOv8 and ByteTrack, with real-time performance evaluation.

## Problem

Detect and track multiple objects across video frames in real-time, maintaining persistent identities and measuring detection accuracy.

## Approach

| Component | Technology | Purpose |
|-----------|------------|---------|
| Detection | YOLOv8n (Ultralytics) | Fast, lightweight object detection |
| Tracking | ByteTrack | Persistent ID assignment across frames |
| Video Processing | OpenCV | Frame reading, resizing, FPS control |
| Evaluation | Custom IoU-based | Precision, Recall, F1, FPS measurement |

**Target Classes**: Person, Bicycle, Car, Motorcycle

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Object Detection + Tracking

```bash
# Live display with tracking
python src/detect.py --video data/sample.mp4

# Save output video
python src/detect.py --video data/sample.mp4 --output output.mp4

# Detection only (no tracking)
python src/detect.py --video data/sample.mp4 --no-track
```

### Advanced Tracking with Trails

```bash
# Full tracking with movement trails
python src/track.py --video data/sample.mp4

# Save output
python src/track.py --video data/sample.mp4 --output tracked.mp4
```

### Evaluation

```bash
# Evaluate FPS performance
python src/evaluate.py --video data/sample.mp4

# Evaluate with ground truth labels
python src/evaluate.py --video data/sample.mp4 --labels data/labels/

# Save metrics to JSON
python src/evaluate.py --video data/sample.mp4 --output metrics.json
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | TP / (TP + FP) - How many detections are correct |
| **Recall** | TP / (TP + FN) - How many ground truths are found |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **FPS** | Frames processed per second |

## Project Structure

```
cv-object-detection-tracking/
├── data/           # Video files and datasets
├── models/         # Cached YOLO models
├── src/
│   ├── detect.py   # Detection + video ingestion
│   ├── track.py    # Object tracking with trails
│   └── evaluate.py # Metrics computation
├── requirements.txt
└── README.md
```

## Key Learnings

1. **ByteTrack > DeepSORT** for real-time applications - simpler and faster
2. **YOLOv8n** balances speed and accuracy well for CPU/GPU
3. **Frame resizing** significantly improves FPS without major accuracy loss
4. **IoU threshold of 0.5** is standard for detection evaluation
5. **Ultralytics tracking API** handles ByteTrack integration natively

## Performance Notes

- YOLOv8n achieves ~30+ FPS on CPU, 100+ FPS on GPU
- Tracking adds minimal overhead (~5%)
- 640px width provides good balance of speed and accuracy
# Real-Time-Object-Detection-Tracking-
