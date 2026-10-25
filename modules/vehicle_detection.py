"""
VisionGuard — Vehicle & Road User Detection Module
Ultralytics YOLO inference engine tuned for traffic surveillance entities.
Detects cars, motorcycles, buses, trucks, bicycles, and persons.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from config import TARGET_CLASSES, DEFAULT_CONFIDENCE_THRESHOLD, YOLO_BASE_MODEL


class VehicleDetector:
    """Performs deep learning object detection for traffic monitoring classes."""

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.model_path = model_path or str(YOLO_BASE_MODEL)
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self):
        """Loads YOLO model weights with graceful error handling."""
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"Error loading YOLO weights from {self.model_path}: {e}")
                self.model = None
        else:
            # Attempt to initialize standard YOLOv8n
            try:
                self.model = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"Failed to initialize default YOLOv8n model: {e}")
                self.model = None

    def detect(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Runs object detection on the provided video frame.
        Returns a list of detected objects:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": int,
                "class_name": str,
                "confidence": float
            }, ...
        ]
        """
        if self.model is None or frame is None:
            return []

        threshold = conf if conf is not None else self.conf_threshold
        try:
            results = self.model(frame, conf=threshold, verbose=False)[0]
        except Exception as e:
            print(f"Detection inference error: {e}")
            return []

        detections = []
        if not results.boxes:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for box, cls_id, score in zip(boxes, classes, confs):
            if cls_id in TARGET_CLASSES:
                x1, y1, x2, y2 = [int(v) for v in box]
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(cls_id),
                    "class_name": TARGET_CLASSES[cls_id],
                    "confidence": round(float(score), 3)
                })

        return detections

    def get_supported_classes(self) -> Dict[int, str]:
        return TARGET_CLASSES
